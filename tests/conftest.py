import os

os.environ.setdefault("FASTSCIPY_STRICT", "1")  # fast-path exceptions fail the test

import pytest  # noqa: E402

import fastscipy  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def patched():
    yield fastscipy.patch()
    fastscipy.unpatch()
