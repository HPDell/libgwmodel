
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("pygwmodel", [
        "gwss.pyx",
        "mat_interface.pyx",
        "name_list_interface.pyx",
        "simple_layer.pyx",
        "spatial_weight.pyx",
        "variable_interface.pyx"
    ], libraries=["libgwmodel"])])
)