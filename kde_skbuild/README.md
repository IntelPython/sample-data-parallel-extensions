# Building this extension

Assuming `numpy`, `Cython`, `pybind11`, `pytest`, `scikit-build`, `cmake >=3.21`, and `ninja` are installed
and DCP++ has been activated:

```bash
CC=icx CXX=icpx  python setup.py develop -G Ninja -- -DDCPTL_MODULE_PATH=$(python -m dpctl --cmakedir)
pytest -m tests
```