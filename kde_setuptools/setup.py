#  Copyright 2022 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
import dpctl

from setuptools.command.build_ext import build_ext

class custom_build_ext(build_ext):
    def build_extensions(self):
        self.compiler.set_executable("compiler_so", "dpcpp -fPIC")
        self.compiler.set_executable("compiler_cxx", "dpcpp -fPIC")
        self.compiler.set_executable("linker_so", "dpcpp -shared -fpic -fsycl-device-code-split=per_kernel")
        build_ext.build_extensions(self)

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
    description="An example of data-parallel Python extensions built with setuptools and oneAPI DPC++",
    long_description="""
    Example of using oneAPI to build data-parallel extension using setuptools.

    Part of oneAPI for Scientific Python community virtual poster.
    Also see README.md
    """,
    license="Apache 2.0",
    url="https://intelpython.github.io/oneAPI-for-SciPy",
    ext_modules=ext_modules,
    cmdclass={"build_ext": custom_build_ext}
)
