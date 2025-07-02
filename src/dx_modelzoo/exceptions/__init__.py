class UnKnownError(Exception):
    """UnKownError Exception."""

    def __init__(self) -> None:
        super().__init__("An unknown error has occurred. Please report this error to DEEPX.")


class InvalidPathError(Exception):
    """Invalid Path Error.
    if path is Invalid, raise Error.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Input path is not Exist. Check your path. {path}")


__all__ = ["UnKnownError", "InvalidPathError"]
