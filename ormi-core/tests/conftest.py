import pytest

@pytest.fixture(params=["cpu", "cuda"])
def compute_device(request):
    return request.param