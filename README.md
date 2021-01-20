# libgwmodel --- a C++ library for Geographically Weighted models

This library consist of C++ implementations of some geographically weighted models. Currently, implemented models are:

- [Geographically Weighted Regression][gwr]
- [Geographically Weighted Summary Statistics][gwss]

Comparing with other libraries, this library has these features:

- Pure C++ implementation which promise high efficiency
- Can be linked ether as a static library or shared library
- Use OpenMP and CUDA (to be supported in next stage) to reach very high efficiency
- Cross platform, configure and generate by CMake
- A python wrapper based on Cython

With OOP style codes, this library is easy to be integrated in any C++ project.

## Installation

You can use CMake to build and install. These dependencies are required:

- [Armadillo][arma], together with a BLAS and LAPACK implementation. Such as OpenBLAS and Intel MKL.
- OpenMP (if `ENABLE_OPENMP` is defined), usually provided by complier. On macOS, OpenMP support is disabled by default, but clang support OpenMP.
- CUDA (if `ENABLE_CUDA` is defined, to be supported in the next stage) and cuBLAS.
- Cython (if `WITH_PYTHON` is defined)

Then configure project with cmake, for example

```bash
mkdir build
cd build
cmake ..
```

If you want to build a shared library, configure the project with:

```bash
cmake .. -DBUILD_SHARED_LIBS
```

Then build the projects, for example

```bash
cmake --build . --config Release -T host=x64
```

Currently, auto install is not enabled. It will be finished in the next stage.

## Usage

Usually, include the `gwmodel.h` header file in your project to use this library.
If this library is build as a shared library, `GWMODEL_SHARED_LIB` should be defined in your project.

For more details, check `static.cpp` or `shared.cpp` fils in each test project.

Examples for python is coming.

[gwr]:https://www.onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.2003.tb01114.x
[gwss]:https://www.sciencedirect.com/science/article/pii/S0198971501000096
[arma]:http://arma.sourceforge.net/
