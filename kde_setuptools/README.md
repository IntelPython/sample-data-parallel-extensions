# Building this extension

Assuming `Cython` and `pybind11` are installed and DCP++ has been activated:

```bash
CC=dpcpp LDSHARED="dpcpp --shared" python setup.py develop
pytest -m tests
```