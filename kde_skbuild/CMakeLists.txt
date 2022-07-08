cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(oneapi_kde_skbuild LANGUAGES C CXX)

if (NOT DEFINED DPCTL_MODULE_PATH)
  if (DEFINED ENV{DPCTL_MODULE_PATH})
    set(DPCTL_MODULE_PATH $ENV{DPCTL_MODULE_PATH})
  else ()
    mesage(FATAL_ERROR "Specify DPCTL_MODULE_PATH, either via cmake or as environment varibale")
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${DPCTL_MODULE_PATH})
find_package(IntelDPCPP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.9.2.tar.gz
  URL_HASH SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1
)
FetchContent_MakeAvailable(pybind11)

find_package(PythonExtensions REQUIRED)
set(CYTHON_FLAGS "-t -w ${CMAKE_SOURCE_DIR}")
find_package(Cython REQUIRED)
find_package(Dpctl REQUIRED)


# Pybind11 extension module
set(py_module_name _pybind11_kde)
pybind11_add_module(${py_module_name}
    MODULE
    ${CMAKE_SOURCE_DIR}/src/pybind11_kde.cpp
)
target_include_directories(${py_module_name}
    PUBLIC ${CMAKE_SOURCE_DIR}/src ${Dpctl_INCLUDE_DIR}
)
install(TARGETS ${py_module_name} DESTINATION kde_skbuild)

# Cython extension


set(cy_module_name _cython_kde)

# subdirectory must be known for add_cython_target bails out
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_cython_target(${cy_module_name} ${CMAKE_SOURCE_DIR}/src/_cython_kde.pyx CXX PY3 OUTPUT_VAR _generated_src)
add_library(${cy_module_name} MODULE ${_generated_src})

target_include_directories(${cy_module_name} PRIVATE ${NumPy_INCLUDE_DIR} ${Dpctl_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src)
python_extension_module(${cy_module_name})
install(TARGETS ${cy_module_name} LIBRARY DESTINATION kde_skbuild)

set(ignoreMe "${SKBUILD}")
