import pytest


def test_placeholder(num: int = 1) -> None:
    if not num:
        pytest.fail("Test failed!")
