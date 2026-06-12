class OrmiError(Exception):
    pass

###
###
###

class DeviceMismatchError(OrmiError):
    pass

def raise_device_mismatch(msg: str, got: str, expected: str | None = None) -> None:
    err_msg = f'{msg} | Got: "{got}"'
    if expected:
        err_msg += f' | Expected: "{expected}"'
    raise DeviceMismatchError(err_msg)

class ShapeMismatchError(OrmiError):
    pass

def raise_shape_error(msg: str, shape_a: tuple, shape_b: tuple) -> None:
    raise ShapeMismatchError(f"{msg} | Shape A: {shape_a}, Shape B: {shape_b}")