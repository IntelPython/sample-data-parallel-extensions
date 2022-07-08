from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import dpctl

ext_modules = [
    Pybind11Extension(
        "kde_setuptools._pybind11_kde",
        ["./src/pybind11_kde.cpp",],
        include_dirs=["./src", dpctl.get_include(),]
    ),
    Extension(
        "kde_setuptools._cython_kde",
        ["./src/cy_kde.pyx",],
        include_dirs=["./src", dpctl.get_include(),],
        language="c++"
    ),
]

setup(
    name="kde_setuptools",
    author="Intel Corporation",
    version="0.0.1",
    description="An example of data-parallel extensions built with oneAPI",
    long_description="""
    Example of using oneAPI to build data-parallel extension using setuptools.

    Part of oneAPI for Scientific Python community virtual poster.
    Also see README.md
    """,
    license="Apache 2.0",
    url="https://intelpython.github.io/oneAPI-for-SciPy",
    ext_modules=ext_modules
)
