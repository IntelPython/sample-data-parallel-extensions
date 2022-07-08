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
