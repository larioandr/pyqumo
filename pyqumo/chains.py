import numpy as np
import enum

from pyqumo.matrix import is_infinitesimal, order_of, is_pmf, is_stochastic, \
    cached_method


def identity_row(length, k):
    return np.eye(1, length, k)


class DTMC:
    def __init__(self, matrix, check=True, tol=1e-8, dtype=None):
        if check:
            if not is_stochastic(matrix, tol, tol):
                raise ValueError("not stochastic matrix")
        self._matrix = np.asarray(matrix, dtype=dtype)
        self._order = order_of(self._matrix)
        self.__cache__ = {}

    @property
    def matrix(self):
        return self._matrix

    @property
    def order(self):
        return self._order

    @cached_method('steady_pmf')
    def steady_pmf(self):
        """Returns the steady-state probabilities distribution (mass function).

        The algorithm will attempt to use `numpy.linalg.solve()` method to
        solve the system. If it fails due to singular matrix,
        `numpy.linalg.lstsq()` method will be used.

        Notes:
            the method is cached: while the first call may take quite a
            lot of time, all the succeeding calls will require only cache
            lookups and have O(1) complexity.
        """
        n = self.order
        lside = np.vstack((self.matrix.T - np.eye(n), np.ones((1, n))))
        rside = np.zeros(n + 1)
        rside[-1] = 1.
        try:
            a_lside = lside[1:, :]
            a_rside = rside[1:]
            return np.linalg.solve(a_lside, a_rside)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lside, rside)[0]

    def trace(self, size, ends=None, state0=None, pmf0=None, tol=1e-8):
        """Generates a generator for a random path produced by the chain.

        A path is represented as tuples `(prev_state, next_state)`.
        The initial state is either given as `state0`, or is determined by
        the `pmf0` parameter. If `pmf0` is also missing, the initial state
        is selected uniformly.

        Args:
            size: number of steps
            ends: states where to finish the trace
            state0: initial state (optional)
            pmf0: initial states probability distribution (optional)
            tol: tolerance used within the pmf0 validity checks

        Returns: path generator.
        """
        if state0 is None:
            pmf0 = np.asarray(pmf0) if pmf0 is not None else \
                np.full((1, self.order), 1. / self.order).flatten()
            if not is_pmf(pmf0, tol, tol):
                raise ValueError("pmf0 is not a PMF")
            # noinspection PyTypeChecker
            state0 = np.random.choice(self.order, p=pmf0)
        if state0 < 0 or state0 >= self.order:
            raise ValueError("state0 must be in 0..N-1")

        for i in range(size):
            state1 = np.random.choice(self.order, p=self.matrix[state0])
            yield state0, state1
            if ends is not None and state1 in ends:
                return
            state0 = state1

    def generate_path(self, size, ends=None, state0=None, pmf0=None, tol=1e-8):
        generator = self.trace(size, ends=ends, state0=state0, pmf0=pmf0,
                               tol=tol)
        try:
            first = next(generator)
        except StopIteration:
            return []
        else:
            rest = [step[1] for step in generator]
            return [first[0]] + rest if len(rest) > 0 else list(first)

    def paths(self, state0, ends=None, ptol=1e-8, debug=False):
        """Generates paths from `state0` to finite states (or `ends`, if given).

        Notes:
            chains with cycles not supported now. If a cycle found, it would be
            ignored.

        Args:
            state0: all paths will start from this state
            ends: optional set of end points. If provided, only paths to these
                states will be generated and paths to any absorbing state
                otherwise. Note that `ends` could be non-absorbing.
            ptol: probability tolerance. The algorithm will ignore transitions
                with probabilities less than `ptol`
            debug: flag indicating whether to print the traces during search
        """
        if state0 >= self.order:
            raise ValueError("state0 must be a valid state number")

        class StateStatus(enum.Enum):
            NEW = 0
            PROCESSING = 1
            VISITED = 2

        # 1) Marking all states but state0 as NEW, initiating path stack
        states = [StateStatus.NEW] * self.order

        # 2) To simplify the following algorithm, building neighbours sets
        #    according to transition matrix and `ptol` value
        neighbours = [[]] * self.order
        for i in range(self.order):
            curr_state_neighbours = []
            for j in range(self.order):
                if i != j and self.matrix[i, j] >= ptol:
                    curr_state_neighbours.append(j)
            neighbours[i] = curr_state_neighbours

        def next_state(s0):
            for neigh in neighbours[s0]:
                if states[neigh] == StateStatus.NEW:
                    return neigh
            return None

        # 3) Running DFS
        path = [state0]
        states[state0] = StateStatus.PROCESSING
        if debug:
            print("(DFS) started with state0={}, ends={}, ptol={}, matrix={}"
                  "".format(state0, "" if not ends else ends, ptol,
                            self.matrix.tolist()))
        while path:
            if debug:
                print("(DFS) * states = {}, path = {}".format(
                    [str(s) for s in states], path))
            v_curr = path[-1]
            v_next = next_state(v_curr)
            if v_next is None:
                # If v_curr has no neighbours, it is an absorbing state.
                # Otherwise, a cycle found (and any path from v_next
                # completes some cycle since all neighbours were already
                # seen before)
                if debug:
                    print("(DFS) \t curr_state={} - a leaf or leads "
                          "to cycle".format(v_curr))
                if ((not ends and not neighbours[v_curr]) or
                        (ends and v_curr in ends)):
                    if debug:
                        print("(DFS) \t yielding {}".format(path))
                    yield list(path)
                states[v_curr] = StateStatus.VISITED
                path.pop()
            else:
                path.append(v_next)
                # noinspection PyTypeChecker
                states[v_next] = StateStatus.PROCESSING
        if debug:
            print("(DFS) * states = {}, path = {}".format(
                [str(s) for s in states], path))
            print("(DFS) == end ==")


class CTMC:
    def __init__(self, matrix, check=True, tol=1e-8, dtype=None):
        if check:
            if not is_infinitesimal(matrix, tol, tol):
                raise ValueError("not infinitesimal matrix")
        self._matrix = np.asarray(matrix, dtype=dtype)
        self._order = order_of(self._matrix)
        self.__cache__ = {}

    @property
    def matrix(self):
        return self._matrix

    @property
    def order(self):
        return self._order

    @cached_method('steady_pmf')
    def steady_pmf(self):
        """Returns the steady-state probabilities distribution (mass function).

        Depending on the generator matrix  the algorithm will use either
        `numpy.linalg.solve()` (if the generator matrix has rank N-1), or
        `numpy.linalg.lstsq()` (if the generator matrix has rank N-2 or less).

        Notes:
            the method is cached: while the first call may take quite a
            lot of time, all the succeeding calls will require only cache
            lookups and have O(1) complexity.
        """
        lside = np.vstack((self.matrix.T, np.ones((1, self.order))))
        rside = np.zeros(self.order + 1)
        rside[-1] = 1.
        try:
            a_lside = lside[1:, :]
            a_rside = rside[1:]
            return np.linalg.solve(a_lside, a_rside)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lside, rside)[0]

    def trace(self, size, state0=None, pmf0=None, tol=1e-8):
        """Generates a generator for a random path produced by the chain.

        A path is represented as tuples `(prev_state, next_state, interval)`.
        The initial state is either given as `state0`, or is determined by
        the `pmf0` parameter. If `pmf0` is also missing, the initial state
        is selected uniformly.

        Args:
            size: number of steps
            state0: initial state (optional)
            pmf0: initial states probability distribution (optional)
            tol: tolerance used within the pmf0 validity checks and
                in computing the absorbing state (the state is absorbing
                if its leaving rate is absolutely less than `tol`)

        Returns: path generator.
        """
        dtmc_trace = self.embedded_dtmc().trace(size, state0=state0, pmf0=pmf0)
        for state0, state1 in dtmc_trace:
            rate = -self.matrix[state0, state0]
            mean = 1. / rate if np.abs(rate) > tol else np.inf
            dt = np.random.exponential(mean)
            yield state0, state1, dt

    @cached_method('embedded_dtmc')
    def embedded_dtmc(self, tol=1e-8):
        """Get the discrete time process embedded at transition times.

        Returns: DTMC
        """
        n = self.order
        rates = self.matrix.diagonal()
        d1 = self.matrix - np.diag(rates)
        probs = np.vstack(np.eye(1, n, i) if np.abs(rates[i]) < tol
                          else d1[i] / np.abs(rates[i]) for i in range(n))
        return DTMC(probs)
