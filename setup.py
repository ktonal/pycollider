import sys

from glob import glob
from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("pycollider._pycollider",
        sources = sorted(glob("src/*.cpp")),
        include_dirs=["src/", "include/plugin_interface", "include/common", "include/server"],
        # Example: passing in the version to the compiled code
        libraries=['dl'],
        define_macros = [('VERSION_INFO', __version__)],
        ),
]


setup(
    name="pycollider",
    version=__version__,
    author="Bjoern Erlach",
    author_email="berlach@ccrma.stanford.edu",
    packages=find_packages(),
    #url="https://github.com/pybind/python_example",
    description="PyCollider is a library for loading SuperCollider plugins.",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    zip_safe=False,
    python_requires=">=3.6",
)
