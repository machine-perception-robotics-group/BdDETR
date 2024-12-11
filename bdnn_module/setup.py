from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy
import sys

ext = Extension("binaryfunc_cython", sources=["binaryfunc_cython.pyx"], include_dirs=sys.path,
                extra_compile_args=['-fopenmp', '-march=native'], extra_link_args=['-fopenmp', '-march=native'], libraries=['m'])

setup(
    name="binaryfunc_cython",
    ext_modules=cythonize([ext]),
    include_dirs=[numpy.get_include()]
)