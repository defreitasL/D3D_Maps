import pytest


@pytest.mark.parametrize("value,expected", [(2, 4), (3, 9), (4, 16)])
def test_main(value, expected):
    result = value * value
    assert result == expected
