import os

from setuptools import find_packages, setup


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
LICENSE = "GPL-3.0"
URL = "https://github.com/UM-RMRS/raster_tools"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/UM-RMRS/raster_tools/issues"
}
INSTALL_REQUIRES = get_requirements("requirements/default.txt")


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
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.8",
)
