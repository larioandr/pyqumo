def test_local():
    from pyqumo import dummy
    assert dummy.welcome() == 'Welcome from pyqumo'


def test_pydesim_dep():
    from pydesim import dummy
    assert dummy.hello() == 'Hello World'
