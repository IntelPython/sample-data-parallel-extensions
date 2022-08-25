![Building and testing](https://github.com/IntelPython/sample-data-parallel-extensions/actions/workflows/ci.yaml/badge.svg)

# Collection of sample oneAPI Python extensions

oneAPI Python extensions is a native Python extension compiled with DPC++ and
targeting various devices programmable by DPC++, e.g. GPUs, multi-core CPUs or
accelerators such as FPGA.

This collection of examples is part of "[oneAPI for Scientific Python Community][poster]"
virtual poster by @oleksandr-pavlyk and @diptorupd presented at [SciPy 2022][scipy22] conference.

Two packages `kde_setuptools` and `kde_pybind11` implement the same Kernel Density Estimation
code executable on SYCL devices supported by DPC++, e.g. Intel CPUs, Intel GPUs.

They only differ in how they are built. One is built with plain `setuptools` while other
is build with `scikit-build` and uses DPC++ integration with CMake.

[poster]: https://intelpython.github.io/oneAPI-for-SciPy
[scipy22]: https://www.scipy2022.scipy.org/