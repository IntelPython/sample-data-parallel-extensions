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

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>

namespace example {

namespace detail {
template <typename T>
T upper_quotient_of(T n, T wg)
{
    return ((n + wg - 1) / wg);
}

template <typename T1, typename T2>
T1 upper_quotient_of(T1 n, T2 wg) {
    return upper_quotient_of(n, static_cast<T1>(wg));
}

} // namespace detail


/*
    Evaluates

     f(x, h) = sum(
        1/(sqrt(2*pi)*h)**dim * exp( - dist_squared(x, x_data[j])/(2*h*h)),
        0 <= j < n_data)

     writes out f(x, h) for every x.

     Execution target is specified with sycl::queue argument.

     All pointers are presumed allocated with system allocator (i.e. malloc, or new)
 */
template <typename T>
void
kernel_density_estimate(
    sycl::queue &q,
    std::uint16_t dim, // dimensionality of the data
    const T* x,        // points at which KDE is evaluated
    T *f,              // where values of kde(x, h) are written to
    size_t n,          // number of points to evaluate
    const T* data,     // data-set
    size_t n_data,     // size of data-set
    T h                // smoothing parameter
    )
{
    const sycl::property_list buf_props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<T, 2> buf_X(x, sycl::range<2>(n, dim), buf_props);
    sycl::buffer<T, 2> buf_D(data, sycl::range<2>(n_data, dim), buf_props);
    sycl::buffer<T, 1> buf_F(f, sycl::range<1>(n), buf_props);

    // initialize buffer of function values with zeros
    try {
    q.submit(
	[&](sycl::handler &cgh) {
	    sycl::accessor acc_F(buf_F, cgh, sycl::write_only, sycl::no_init);
	    cgh.fill(acc_F, T(0));
	});
    } catch (const std::exception &e){
      std::cout << e.what() << std::endl;
      std::rethrow_exception(std::current_exception());
    }

    // populate function values buffer
    //   perform 2D loop in parallel
    try{
    q.submit(
	[&](sycl::handler &cgh) {
	    sycl::accessor acc_X(buf_X, cgh, sycl::read_only);
	    sycl::accessor acc_D(buf_D, cgh, sycl::read_only);
	    sycl::accessor acc_F(buf_F, cgh, sycl::write_only, sycl::no_init);

	    T two_pi = T(8) * sycl::atan(T(1));
	    T gaussian_norm;
	    if (dim == 1) {
		// use reciprocal sqrt function for efficiency
		gaussian_norm = sycl::rsqrt(two_pi) / h;
	    } else if (dim % 2 == 1) {
		gaussian_norm = sycl::rsqrt(two_pi) / h / sycl::pown(two_pi * h * h, dim / 2);
	    } else {
		gaussian_norm = T(1) / sycl::pown(two_pi * h * h, dim / 2);
	    }

	    size_t wg = 256;
	    size_t n_data_per_wi = 16;

	    auto gRange = sycl::range<2>(n, detail::upper_quotient_of(n_data, wg * n_data_per_wi) * wg);
	    auto lRange = sycl::range<2>(1, wg);

	    cgh.parallel_for(
		sycl::nd_range<2>(gRange, lRange),
		[=](sycl::nd_item<2> it) {
		    auto x_id = it.get_global_id(0);
		    auto x_data_batch_id = it.get_group(1);

		    auto x_data_local_id = it.get_local_id(1);

		    // work-items sums over data-points with indices
		    //   x_data_id = x_data_batch_id * wg * n_data_per_wi + m * wg + x_data_local_id
		    // for 0 <= m < n_wi
		    T local_sum(0);

		    for(size_t m = 0; m < n_data_per_wi; ++m) {
			T dist_sq(0);
			size_t x_data_id = x_data_local_id + m * wg + x_data_batch_id * wg * n_data_per_wi;
			if (x_data_id < n_data) {
			    for(std::uint8_t k=0; k < dim; ++k) {
				T diff =
				    acc_X[sycl::id<2>(     x_id, k)] -
				    acc_D[sycl::id<2>(x_data_id, k)];
				dist_sq += diff * diff;
			    }
			    local_sum += (gaussian_norm / n_data) * sycl::exp(-dist_sq / T(2) / (h*h));
			}
		    }

		    auto work_group = it.get_group();
		    T sum_over_wg = sycl::reduce_over_group(work_group, local_sum, sycl::plus<T>());

		    if (work_group.leader()) {
			sycl::atomic_ref<T, sycl::memory_order::relaxed,
					 sycl::memory_scope::device,
					 sycl::access::address_space::global_space> f_ref(acc_F[x_id]);
			f_ref += sum_over_wg;
		    }
		});
	});
    } catch (const std::exception &e) {
      std::cout << e.what() << std::endl;
      std::rethrow_exception(std::current_exception());
    }

    return;
}

} // namespace example
