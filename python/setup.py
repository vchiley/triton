import distutils
import distutils.spawn
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from distutils.version import LooseVersion
from typing import NamedTuple

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_build_type():
    if check_env_flag("DEBUG"):
        return "Debug"
    elif check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    else:
        return "Release"


def use_system_llvm():
    if platform.system() == "Windows":
        return True
    versions = ['-11.0', '-11', '-11-64']
    supported = ['llvm-config{v}'.format(v=v) for v in versions]
    paths = [distutils.spawn.find_executable(cfg) for cfg in supported]
    return any(p is not None for p in paths)


def get_thirdparty_packages(triton_cache_path):
    class Package(NamedTuple):
        package: str
        name: str
        url: str
        test_file: str
        include_flag: str
        lib_flag: str

    packages = [
        Package("pybind11", "pybind11-2.10.0", "https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.tar.gz", "include/pybind11/pybind11.h", "PYBIND11_INCLUDE_DIR", "")
    ]
    if not use_system_llvm():
        # download LLVM if no suitable system LLVM is installed
        packages.append(
            Package("llvm", "clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04", "https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz", "lib", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR")
        )

    thirdparty_cmake_args = []
    for p in packages:
        package_root_dir = os.path.join(triton_cache_path, p.package)
        package_dir = os.path.join(package_root_dir, p.name)
        test_file_path = os.path.join(package_dir, p.test_file)
        if not os.path.exists(test_file_path):
            try:
                shutil.rmtree(package_root_dir)
            except Exception:
                pass
            os.makedirs(package_root_dir, exist_ok=True)
            print('downloading and extracting {} ...'.format(p.url))
            ftpstream = urllib.request.urlopen(p.url)
            file = tarfile.open(fileobj=ftpstream, mode="r|*")
            file.extractall(path=package_root_dir)
        if p.include_flag:
            thirdparty_cmake_args.append("-D{}={}/include".format(p.include_flag, package_dir))
        if p.lib_flag:
            thirdparty_cmake_args.append("-D{}={}/lib".format(p.lib_flag, package_dir))
    return thirdparty_cmake_args


class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    user_options = build_ext.user_options + [('base-dir=', None, 'base directory of Triton')]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        triton_cache_path = os.path.join(os.environ["HOME"], ".triton_pre_mlir")
        thirdparty_cmake_args = get_thirdparty_packages(triton_cache_path)
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.path)))
        # create build directories
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        # python directories
        python_include_dirs = [distutils.sysconfig.get_python_inc()]
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DBUILD_TUTORIALS=OFF",
            "-DBUILD_PYTHON_MODULE=ON",
            # '-DPYTHON_EXECUTABLE=' + sys.executable,
            # '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON',
            "-DPYTHON_INCLUDE_DIRS=" + ";".join(python_include_dirs)
        ] + thirdparty_cmake_args
        # configuration
        cfg = get_build_type()
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            import multiprocessing
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", '-j' + str(2 * multiprocessing.cpu_count())]

        env = os.environ.copy()
        subprocess.check_call(["cmake", self.base_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setup(
    name="triton_pre_mlir",
    version="2.0.0",
    author="Philippe Tillet",
    author_email="phil@openai.com",
    description="A language and compiler for custom Deep Learning operations",
    long_description="",
    packages=["triton_pre_mlir", "triton_pre_mlir/_C", "triton_pre_mlir/language", "triton_pre_mlir/runtime", "triton_pre_mlir/tools", "triton_pre_mlir/ops", "triton_pre_mlir/ops/blocksparse"],
    install_requires=[
        "cmake",
        "filelock",
        "torch",
    ],
    package_data={
        "triton_pre_mlir/ops": ["*.c"],
        "triton_pre_mlir/ops/blocksparse": ["*.c"],
        "triton_pre_mlir/language": ["*.bc"],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("triton_pre_mlir", "triton_pre_mlir/_C/")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    # for PyPI
    keywords=["Compiler", "Deep Learning"],
    url="https://github.com/openai/triton/",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    extras_require={
        "tests": [
            "autopep8",
            "flake8",
            "isort",
            "numpy",
            "pytest",
            "scipy>=1.7.1",
        ],
        "tutorials": [
            "matplotlib",
            "pandas",
            "tabulate",
        ],
    },
)
