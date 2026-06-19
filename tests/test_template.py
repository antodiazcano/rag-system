"""This script contains the test for the sum function."""

import pytest

from src.config import config
from src.template_function import f


@pytest.fixture
def random_int() -> int:
    """Dummy fixture.

    Returns:
        Just 1.
    """

    return 1


@pytest.mark.parametrize("a, b, expected", [(1, 2, 3), (5, -1, 4)])
def test_f(a: int, b: int, expected: int) -> None:
    """Test for the sum function.

    Args:
        a: First operand.
        b: Second operand.
        expected: Expected value.
    """

    assert f(a, b) == expected + config.model.in_dim


def test_fixture(random_int: int) -> None:
    """Dummy test.

    Args:
        random_int: Pytest fixture.
    """

    assert isinstance(random_int, int)
