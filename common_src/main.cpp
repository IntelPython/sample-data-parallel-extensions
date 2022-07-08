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

// This is auxiliary file used to test kde.hpp

#include <CL/sycl.hpp>
#include <random>
#include "kde.hpp"

int main(void) {
    sycl::queue q;

    using T = double;

    size_t n = 4;
    size_t n_data = 16253;

    T *x = new T[n];
    T *f = new T[n];
    T *x_data = new T[n_data];

    for(size_t i=0; i < n; ++i) {
	x[i] = T(i+2)/T(n+4);
    }

    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<T> uniform_d(T(0), T(1));

    for(size_t i=0; i < n_data; ++i) {
	x_data[i] = uniform_d(e);
    }

    example::kernel_density_estimate<double>(
	q,
	static_cast<std::uint32_t>(1),
	const_cast<const T *>(x),
	f, n,
	const_cast<const T *>(x_data),
	n_data, 0.2
    );

    for(size_t i=0; i < n; ++i) {
	std::cout << "f(" << x[i] << ")=" << f[i] << std::endl;
    }

    return 0;
}
