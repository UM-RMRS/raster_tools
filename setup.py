import os
from multiprocessing import cpu_count

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, find_packages, setup

Options.fast_fail = True


def read_file(fname):
    with open(fname, encoding="utf-8") as fd:
        return fd.read()


def get_version():
    with open(os.path.join("raster_tools", "_version.py")) as fd:
        # contents are __version__ = "<vstring>"
        return fd.read().split("=")[1].strip().strip('"')


def get_requirements(fname):
    with open(fname, encoding="utf-8") as fd:
        reqs = [line.strip() for line in fd if line]
    return reqs


NAME = "raster-tools"
DESCRIPTION = "Tools for processing geospatial data"
LONG_DESCRIPTION = read_file("README.md")
VERSION = get_version()
LICENSE = read_file("LICENSE")
URL = "https://github.com/UM-RMRS/raster_tools"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/UM-RMRS/raster_tools/issues"
}
INSTALL_REQUIRES = get_requirements("requirements/default.txt")
EXT_MODULES = cythonize(
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
)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    project_urls=PROJECT_URLS,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["docs", "tests"]),
    package_data={"": ["*.pyx", "*.pxd"]},
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
    ext_modules=EXT_MODULES,
    include_dirs=[np.get_include()],
)
