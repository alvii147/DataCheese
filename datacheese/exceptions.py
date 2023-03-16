class ArrayShapeError(BaseException):
    """
    Exception raised for invalid array shape.
    """

    pass


class NotFittedError(BaseException):
    """
    Exception raised for running prediction for model that has no been fitted.
    """

    pass
