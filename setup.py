from multiprocessing import cpu_count

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.fast_fail = True

setup(
    name="raster-tools",
    ext_modules=cythonize(
        [
            Extension(
                "raster_tools.distance._heap",
                ["raster_tools/distance/_heap.pyx"],
                extra_compile_args=[
                    "-O3",
                    "-march=native",
                    "-g0",
                ],
            ),
            Extension(
                "raster_tools.distance._core",
                ["raster_tools/distance/_core.pyx"],
                extra_compile_args=[
                    "-O3",
                    "-march=native",
                    "-g0",
                ],
            ),
        ],
        nthreads=cpu_count(),
    ),
    include_dirs=[np.get_include()],
)
