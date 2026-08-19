"""Hello World test for prose_metrics package."""

from prose_metrics import hello


def test_hello() -> None:
    """Test the hello() function from prose_metrics.

    Raises:
        ValueError: If the hello() function does not return the expected string.
    """
    if not hello() == "Hello from prose-metrics!":
        raise ValueError("hello() should return 'Hello from prose-metrics!'")
