# Building this extension

Assuming `Cython` and `pybind11` are installed and DCP++ has been activated:

```bash
CC=dpcpp LDSHARED="dpcpp --shared" python setup.py develop
pytest -m tests
```

**Note**: Building this package requires the using latest development version of `dpctl`.
This can be installed using conda package manager using

```bash
conda install -c dppy/label/dev dpctl
```