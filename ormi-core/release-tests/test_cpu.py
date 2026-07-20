import os

import numpy as np

import ormi_core as oc


def test_installed_release_version():
    assert oc.__version__ == os.environ["ORMI_RELEASE_VERSION"]


def test_cpu_works_without_a_jit_compiler():
    oc.jit.clear()

    lhs_data = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    rhs_data = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    result = oc.Tensor(lhs_data) @ oc.Tensor(rhs_data)

    np.testing.assert_allclose(
        result.numpy(),
        lhs_data @ rhs_data,
        rtol=1e-4,
        atol=1e-3,
    )
    assert oc.jit.info().entries == 0
