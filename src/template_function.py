"""This script contains the function for the sum of two numbers."""

from src.config import config


def f(a: int, b: int) -> int:
    """This function sums two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        Sum of the numbers.
    """

    return a + b + config.model.in_dim


if __name__ == "__main__":  # pragma: no cover
    print(f(1, 2))
