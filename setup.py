import os
from multiprocessing import cpu_count

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.fast_fail = True


def get_version():
    with open(os.path.join("raster_tools", "_version.py")) as fd:
        # contents are __version__ = "<vstring>"
        return fd.read().split("=")[1].strip().strip('"')


setup(
    name="raster-tools",
    version=get_version(),
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
