# Building this extension

Assuming `numpy`, `Cython`, `pybind11`, `pytest`, `scikit-build`, `cmake >=3.21`, `dpctl`, and `ninja` are
installed and DCP++ has been activated:

```bash
CC=icx CXX=icpx python setup.py develop -G Ninja
pytest -m tests
```

Scikit-build enables building Python native extensions using CMake. This package leverage integration DPC++
with CMake [dpcpp-cmake-integration] as well as integration of `dpctl` with CMake.

The latest version of `dpctl` can be installed using conda package manager using

```bash
conda install -c dppy/label/dev dpctl
```

[dpcpp-cmake-integration]: https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compiler-setup/use-the-command-line/use-cmake-with-the-intel-oneapi-dpc-c-compiler.html