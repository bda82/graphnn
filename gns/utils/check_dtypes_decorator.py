from functools import wraps

from gns.utils.check_dtypes import check_dtypes


def check_dtypes_decorator(call):
    """
    Decorator for automatic type checking.

    Args:
        call: param function call
        
    Returns:
    """

    @wraps(call)
    def _inner_check_dtypes(inputs, **kwargs):
        inputs = check_dtypes(inputs)

        return call(inputs, **kwargs)

    return _inner_check_dtypes
