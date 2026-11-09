import pytest
import numpy as np
import ormi_core as oc

###
###
###

def _cuda_available() -> bool:
    try:
        oc.Tensor([0.0], device="cuda").numpy()
    except Exception:
        return False

    return True

CUDA_AVAILABLE = _cuda_available()

###
###
###

def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Run device tests on CPU and, when available, CUDA.

    Parameters
    ----------
    metafunc : pytest.Metafunc
        Test definition. Add device cases if it requests the ``device`` fixture.
    """

    if "device" in metafunc.fixturenames:
        devices = ["cpu"]

        if CUDA_AVAILABLE:
            devices.append("cuda")

        metafunc.parametrize("device", devices)

@pytest.fixture
def assert_allclose():
    """Provide an approximate array comparison function.

    Returns
    -------
    callable
        Function accepting ``actual``, ``expected``, ``rtol=1e-5``,
        ``atol=1e-8``, and ``err_msg=""``. Convert inputs to NumPy arrays and
        raise AssertionError when ``numpy.testing.assert_allclose`` fails.
    """

    def _to_numpy(array):
        if hasattr(array, "get"):
            return array.get()

        if isinstance(array, oc.Tensor):
            return _to_numpy(array.data)

        return np.asarray(array)

    def _assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, err_msg=""):
        actual_np = _to_numpy(actual)
        expected_np = _to_numpy(expected)

        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg or "Tensor data mismatch exceeding tolerance",
        )

    return _assert_allclose
