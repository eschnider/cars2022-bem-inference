#  Copyright (c) 2022. Eva Schnider


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class NoConnectedComponentPresentError(Error):
    """Exception raised if there is no connected component.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class TooManyComponentsPresentError(Error):
    """Exception raised if there are too many connected components

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
