"""Exception classes for tokenization errors."""

from collections.abc import Sequence


class DecodeError(Exception):
    """Raised when decoding an invalid or malformed token sequence.

    Attributes:
        message: Description of what went wrong.
        tokens: The token sequence that failed to decode.
        position: The position in the sequence where the error occurred.
    """

    def __init__(self, message: str, tokens: Sequence[int], position: int):
        self.message = message
        self.tokens = list(tokens)
        self.position = position
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format a detailed error message with context."""
        # Show tokens around the error position
        context_size = 5
        start = max(0, self.position - context_size)
        end = min(len(self.tokens), self.position + context_size + 1)

        tokens_str = " ".join(
            f"[{t}]" if i == self.position else str(t)
            for i, t in enumerate(self.tokens[start:end], start=start)
        )

        return (
            f"{self.message} at position {self.position}\n"
            f"Tokens: ...{tokens_str}...\n"
            f"Full sequence ({len(self.tokens)} tokens): {self.tokens}"
        )


class VocabularyFrozenError(Exception):
    """Raised when attempting to modify a frozen vocabulary."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(f"Cannot {operation}: vocabulary is frozen")
