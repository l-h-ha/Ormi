# ORMI CUDA

CUDA 12.4 backend package for ORMI Core.

Install the matching core and CUDA backend from the ORMI package index:

```bash
pip install "ormi-core[cuda]" --extra-index-url https://l-h-ha.github.io/Ormi/whl/
```

The wheel includes the CUDA runtime libraries used by ORMI. Users still need a supported NVIDIA GPU and driver. The CUDA Toolkit is required only to build this package from source. The backend is loaded only when a CUDA device is requested.
