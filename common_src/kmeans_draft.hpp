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

//  WARNING:
//    This file is a work in progress, it is currently unused

#include <CL/sycl.hpp>
#include <limits>
#include <cstdint>

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

template <typename T>
void
kmeans_host_data(
    sycl::queue q,       // execution queue
    const T *data,       // pointer to training data, size at least n*dim*sizeof(T)
    size_t n,            // number of data points
    unsigned int dim,    // dimensionality of data points
    T *centroids,        // initial centroids, changed by the algorithms
    unsigned int k,      // number of centroids, same dimensionality
    size_t num_iters,    // number of iterations of Lloyd to perform
    unsigned int *labels // optional pointer to array where to give centroid assignments
    )
{
    const sycl::property_list buf_props {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<T, 2> buf_X(data, sycl::range<2>(n, dim), buf_props);
    sycl::buffer<T, 2> buf_C(centroids, sycl::range<2>(k, dim), buf_props);

    // temporary space for new centroid coordinaties
    sycl::buffer<T, 2> buf_Cn{ sycl::range<2>(k, dim) };
    // temporary space for point labels (index of centroid it is closest to)
    sycl::buffer<unsigned int, 1> buf_L {sycl::range<1>(n)};
    // temporary space for new centroid sizes
    sycl::buffer<size_t, 1> buf_Cs{ sycl::range<1>(k) };

    if (labels != nullptr) {
	buf_L = sycl::buffer<unsigned int, 1>(labels, sycl::range<1>(n), buf_props);
    }

    for(size_t iter=0; iter < num_iters; ++iter) {

    // assign each data-point to one of current centroids
    q.submit(
	[&](sycl::handler &cgh) {

	    // this kernel only reads data-points and centroids
	    //   hence no reason to copy back data from device to host
	    sycl::accessor acc_X(buf_X, cgh, sycl::read_only);
	    sycl::accessor acc_C(buf_C, cgh, sycl::read_only);
	    // it only assigns labels and overwrites the buffer completely
	    //   hence no reason to copy data from host to device, i.e. no_init
	    sycl::accessor acc_L(buf_L, cgh, sycl::write_only, sycl::no_init);

	    cgh.parallel_for(
		sycl::range<1>(n),
		[=](sycl::id<1> id) {
		    auto i = id[0]; // data-point index

		    T min_dsq = std::numeric_limits<T>::max();
		    unsigned int min_j = 0;
		    for (unsigned int j=0; j < k; ++j) {
			T dsq = T(0);
			for(unsigned int s=0; s < dim; ++s) {
			    T diff = acc_X[sycl::id<2>(i, s)] - acc_C[sycl::id<2>(j, s)];
			    dsq += diff * diff;
			}
			if (dsq < min_dsq) {
			    min_dsq = dsq;
			    min_j = j;
			}
		    }
		    acc_L[i] = min_j;
		});
	});

    // zero out buffer where we accumulate data to compute positions of new centeroids
    q.submit(
	[&](sycl::handler &cgh) {
	    auto flat_buf_Cn = buf_Cn.template reinterpret<T, 1>(k*dim);

	    sycl::accessor acc_Cn(flat_buf_Cn, cgh, sycl::write_only, sycl::no_init);
	    cgh.fill(acc_Cn, T(0));
	});

    // zero out sizes for new centroids
    q.submit(
	[&](sycl::handler &cgh) {
	    sycl::accessor acc_Cs(buf_Cs, cgh, sycl::write_only, sycl::no_init);
	    cgh.fill(acc_Cs, size_t(0));
	});

    // compute new centroids
    q.submit(
	[&](sycl::handler &cgh) {
	    sycl::accessor acc_X(buf_X, cgh, sycl::read_only);
	    sycl::accessor acc_L(buf_L, cgh, sycl::read_only);

	    // we read and write new centroids and counts
	    sycl::accessor acc_Cn(buf_Cn, cgh, sycl::read_write);
	    sycl::accessor acc_Cs(buf_Cs, cgh, sycl::read_write);

	    int wg_n = 256;  // work-group size
	    constexpr int n_wi = 16; // Number of coordinates reduced in each work-item
	    constexpr int wg_dim = 2;

	    // local space used to reduce coordinates for 2 componets
	    sycl::accessor<T, 2, sycl::access::mode::read_write,
			   sycl::access::target::local> slm_C(sycl::range<2>(wg_n, wg_dim), cgh);
	    // local space used to reduce cluster sizes
	    sycl::accessor<size_t, 1, sycl::access::mode::read_write,
			   sycl::access::target::local> slm_Cs(sycl::range<1>(wg_n), cgh);

	    // ensure sizeof(size_t) * wg_n + sizeof(T) * wg_n * wg_dim  < max_local_mem_size

	    auto gRange = sycl::range<3>(
		k,
		detail::upper_quotient_of(n, wg_n * n_wi) * wg_n,
		detail::upper_quotient_of(dim, wg_dim)
	        );
	    auto lRange = sycl::range<3>(1, wg_n, 1);
            cgh.parallel_for(
		sycl::nd_range<3>(gRange, lRange),
		[=](sycl::nd_item<3> it) {
                    // each work item processes n_wi data-points
		    // if they are associated with cluster gl_k,
		    // we add their corrdinates up in acc_Cn,
		    // increment counter in acc_Cs
		    // We do some of that locally in each work-item,
		    // then work-items aggregate their partial result using SLM
		    // the SLM then updates global accessor using atomic_ref

		    auto gl_k = it.get_global_id(0);   // cluster id processed
		    auto gl_i = it.get_group(1);       // global batch id
		    auto gl_s = it.get_global_id(2);   // global id

		    auto lo_i = it.get_local_id(1);

		    // we are processing data point with
		    //  index i = gl_i * wg_n * n_wi + m * wg_n + lo_i
		    // where we iterate over 0 <= m < n_wi within the work-item

		    // do the reduction in each work-item using private variable
		    std::array<T, wg_dim> s;
		    for(int p=0; p < wg_dim; ++p) {s[p] = 0;}
		    size_t wi_count = 0;

		    size_t i0 = gl_i * wg_n * n_wi + lo_i;
		    for(int m=0; m < n_wi; ++m) {
			size_t i = i0 + m * wg_n;
			bool mask = (acc_L[i] == k);
			wi_count += ((mask) ? 1 : 0);

			int gp = gl_s * wg_dim;
			for(int p=0; p < wg_dim; ++p, ++gp) {
			    s[p] += (mask && gp < dim) ?  acc_X[sycl::id<2>(i, gp)] : T(0);
			}
		    }

		    size_t group_total = sycl::reduce_over_group(it.get_group(), wi_count, sycl::plus<size_t>());
		    it.barrier(sycl::access::fence_space::local_space);

		    for(int p=0; p < wg_dim; ++p) {
			T sum = sycl::reduce_over_group(it.get_group(), s[p], sycl::plus<T>());
			it.barrier(sycl::access::fence_space::local_space);
			s[p] = sum;
		    }

		    it.barrier(sycl::access::fence_space::local_space);

		    if (it.get_group().leader()) {
			sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
					 sycl::memory_scope::device,
					 sycl::access::address_space::global_space> ar_count(acc_Cs[gl_k]);
			ar_count += group_total;

			int q = gl_s * wg_dim;
			for(int p=0; p < wg_dim && q < dim; ++p, ++q) {
			    sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
					     sycl::memory_scope::device,
					     sycl::access::address_space::global_space> ar_coord_sum(acc_Cn[sycl::id<2>(gl_k, q)]);
			    ar_coord_sum += s[p];
			}
		    }
		});

	});
    };

    return;
}
} // namespace example
