"""pytest configuration for cuntz_bootstrap."""


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, pure-Python structural tests")
    config.addinivalue_line("markers", "integration: physics validation tests (medium runtime)")
    config.addinivalue_line("markers", "slow: long-running phase-C/D runs")
