# libgwmodel --- a C++ library for Geographically Weighted models

[Documentation](https://libgwmodel.readthedocs.io/)

This library consist of C++ implementations of some geographically weighted models. Currently, implemented models are:

- [Geographically Weighted Regression (GWR)][gwr] and its extensions
  - [Geographically weighted regression with parameter-specific distance metrics (PSDM-GWR, MGWR)][psdm-gwr]
  - [Geographically and Temporally Weighted Regression (GTWR)][gtwr]
  - [Generalized GWR (GGWR)][ggwr]
  - Robust GWR
  - Local-collinearity GWR
  - [Scalable GWR][scgwr]
- [Geographically Weighted Summary Statistics (GWSS)][gwss]
- [Geographically Weighted Principal Component Analysis (GWPCA)][gwpca]
- Geographically Weighted Density Regression (SDR)

Comparing with other libraries, this library has these features:

- [x] Pure C++ implementation which promise high efficiency
- [x] Can be linked ether as a static library or shared library
- [x] Use OpenMP to make algorithms faster
- [x] Use CUDA to reach very high efficiency (only for GWR and MGWR)
- [x] Cross platform, configure and generate by CMake

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

> Bindings for Python are archived.
> Please find it at tag [v0.2.0](https://github.com/GWmodel-Lab/libgwmodel/releases/tag/v0.2.0).

## Development

This repositry provides VSCode container development configurations.
Developers can create a container with ready-to-use development environment.
To do so,

1.  Install Docker on your PC.
2.  Open this repositry with VSCode.
3.  Press `Ctrl+P` to open command panel, then choose `Open in Container`.
4.  Wait for the building of container.

When the container is ready, make some changes then build and test.
Happy coding!

# Related Projects

- [GWmodel3](https://github.com/GWmodel-Lab/GWmodel3): The next-generation R package for geographically weighted modeling
- [GWmodelS](https://github.com/GWmodel-Lab/GWmodelS): Desktop studio for geographically weighted modeling

[gwr]:https://www.onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.2003.tb01114.x
[psdm-gwr]:https://www.tandfonline.com/doi/abs/10.1080/13658816.2016.1263731
[gtwr]:https://onlinelibrary.wiley.com/doi/abs/10.1111/gean.12071
[ggwr]:https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2129
[scgwr]:https://www.tandfonline.com/doi/full/10.1080/24694452.2020.1774350
[gwss]:https://www.sciencedirect.com/science/article/pii/S0198971501000096
[gwpca]:https://www.tandfonline.com/doi/full/10.1080/13658816.2011.554838
[arma]:http://arma.sourceforge.net/
