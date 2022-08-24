// Copyright 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File implementing pybind11-driven native Python extension exposing
// data-parallel function to compute Kernel Density Estimation using oneAPI

#include <CL/sycl.hpp>
#include <cstdint>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "dpctl4pybind11.hpp"
#include "kde.hpp"

namespace py = pybind11;

template <typename T>
py::array_t<T>
py_kde_eval_t(
    sycl::queue q,
    py::array_t<T, py::array::c_style|py::array::forcecast> x,
    py::array_t<T, py::array::c_style|py::array::forcecast> data,
    T h)
{
    py::buffer_info x_pybuf = x.request();
    py::buffer_info data_pybuf = data.request();

    if (x_pybuf.ndim != 2 || data_pybuf.ndim != 2) {
	throw py::value_error("Input arrays must be matrices");
    }

    auto dim = x_pybuf.shape[1];
    constexpr auto dim_max = static_cast<decltype(dim)>(
	std::numeric_limits<std::uint16_t>::max());

    if (dim != data_pybuf.shape[1] || dim > dim_max) {
	throw py::value_error(
	    "Inputs have inconsistent functionality or too large a width"
	    );
    }

    auto n = x_pybuf.shape[0];
    auto n_data = data_pybuf.shape[0];

    T *x_ptr = reinterpret_cast<T *>(x_pybuf.ptr);
    T *data_ptr = reinterpret_cast<T *>(data_pybuf.ptr);

    py::array_t<T, py::array::c_style> f({n}, {sizeof(T)});

    example::kernel_density_estimate<T>(
	q, static_cast<std::uint16_t>(dim),
	const_cast<const T*>(x_ptr),
	f.mutable_data(0),
	n,
	const_cast<const T*>(data_ptr),
	n_data,
	h);

    return f;
}

py::array
py_kde_eval(
    sycl::queue exec_q,
    py::array x,
    py::array data,
    py::object h)
{
    if (py::isinstance<py::array_t<double>>(x) && py::isinstance<py::array_t<double>>(data)) {
	return py_kde_eval_t<double>(
	    exec_q,
	    py::cast<py::array_t<double>>(x),
	    py::cast<py::array_t<double>>(data),
	    py::cast<double>(h)
	    );
    } else if (py::isinstance<py::array_t<float>>(x) && py::isinstance<py::array_t<float>>(data)) {
	return py_kde_eval_t<float>(
	    exec_q,
	    py::cast<py::array_t<float>>(x),
	    py::cast<py::array_t<float>>(data),
	    py::cast<float>(h)
	    );
    } else {
	throw py::type_error("Both arrays must be either single or double precision floating point types");
    }
}


PYBIND11_MODULE(_pybind11_kde, m) {
    import_dpctl();

    m.def("kde_eval", &py_kde_eval,
	  "Evaluate kernel density estimation function for every argument for the dataset",
	  py::arg("exec_q"),
	  py::arg("x"),
	  py::arg("data"),
	  py::arg("h") );
}
