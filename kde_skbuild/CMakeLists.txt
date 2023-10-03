cmake_minimum_required(VERSION 3.22...3.27 FATAL_ERROR)

project(oneapi_kde_skbuild VERSION 0.1 LANGUAGES C CXX DESCRIPTION "Example of DPC++-based Python extension")

if (NOT DEFINED DPCTL_MODULE_PATH)
  if (DEFINED ENV{DPCTL_MODULE_PATH})
    set(DPCTL_MODULE_PATH $ENV{DPCTL_MODULE_PATH})
  else ()
    message(FATAL_ERROR "Specify DPCTL_MODULE_PATH, either via cmake or as environment varibale")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})
find_package(IntelSYCL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
  URL_HASH SHA256=d475978da0cdc2d43b73f30910786759d593a9d8ee05b1b6846d1eb16c6d2e0c
)
FetchContent_MakeAvailable(pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module)
set(CYTHON_FLAGS "-t -w ${CMAKE_SOURCE_DIR}")
find_package(Cython REQUIRED)
find_package(Dpctl REQUIRED)


# Pybind11 extension module
set(py_module_name _pybind11_kde)
pybind11_add_module(${py_module_name}
    MODULE
    ${CMAKE_SOURCE_DIR}/src/pybind11_kde.cpp
)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${CMAKE_SOURCE_DIR}/src/pybind11_kde.cpp)
target_link_options(${py_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
target_include_directories(${py_module_name}
    PUBLIC ${CMAKE_SOURCE_DIR}/src ${Dpctl_INCLUDE_DIR}
)
install(TARGETS ${py_module_name} DESTINATION kde_skbuild)

# Cython extension


set(cy_module_name _cython_kde)

# subdirectory must be known for add_cython_target bails out
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_cython_target(${cy_module_name} ${CMAKE_SOURCE_DIR}/src/_cython_kde.pyx CXX PY3 OUTPUT_VAR _generated_src)
Python_add_library(${cy_module_name} MODULE WITH_SOABI ${_generated_src})
add_sycl_to_target(TARGET ${cy_module_name} SOURCES ${_generated_src})

target_include_directories(${cy_module_name} PRIVATE ${NumPy_INCLUDE_DIR} ${Dpctl_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src)
target_link_options(${cy_module_name} PRIVATE -fsycl-device-code-split=per_kernel)
install(TARGETS ${cy_module_name} LIBRARY DESTINATION kde_skbuild)

set(ignoreMe "${SKBUILD}")
