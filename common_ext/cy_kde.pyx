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

# distutils: language = c++
# cython: language_level=3

cimport cython
from libc.stdint cimport uint16_t

cdef extern from "CL/sycl.hpp" namespace "sycl":
    cdef cppclass dpcpp_queue "sycl::queue":
        pass


cdef extern from "kde.hpp" namespace "example":
    void kernel_density_estimate[T](
        dpcpp_queue,   # execution queue
        uint16_t,      # datadimensionality
        const T *,     # evaluation points
        T *,           # output for kde values
        size_t,        # number of evaluation points
        const T *,     # observation data
        size_t,        # number of observations
        T              # smoothing parameter
    )

cimport dpctl as c_dpctl
import numpy as np

def kde_eval(
        c_dpctl.SyclQueue py_q,
        cython.floating[:, :] x,
        cython.floating[:, :] x_data,
        cython.floating h
):
    cdef cython.floating[:] f

    cdef c_dpctl.DPCTLSyclQueueRef qref = py_q.get_queue_ref()
    cdef dpcpp_queue* sycl_queue = <dpcpp_queue *>qref

    if x.shape[1] != x_data.shape[1]:
        raise ValueError("Evaluation data and observation data have different dimensions")

    if (x.shape[0] == 0 or x.shape[1] == 0 or
        x_data.shape[0] == 0 or x_data.shape[1] == 0):
        raise ValueError("Evaluation and observation data must be non-empty")

    if cython.floating is float:
        f = np.empty(x.shape[0], dtype=np.float)
    else:
        f = np.empty(x.shape[0], dtype=np.double)

    kernel_density_estimate(
        sycl_queue[0],
        <uint16_t>x.shape[1],
        &x[0, 0],
        &f[0],
        f.size,
        &x_data[0,0],
        x_data.shape[0],
        h
    )

    return np.asarray(f)
