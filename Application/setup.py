from setuptools import setup
from Cython.Distutils import build_ext
import numpy as np
from distutils.extension import Extension

ext_modules = [
    Extension("process_signal_cython",
              ["process_signal_cython.pyx"],
              extra_compile_args=["-ffast-math"])
]

setup(
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)
