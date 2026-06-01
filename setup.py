from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import platform

class get_pybind_include:
    def __init__(self, user=False):
        self.user = user
    
    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def has_flag(compiler, flagname):
    import tempfile
    from distutils.errors import CompileError
    
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main() { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
        return True
    except CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass

class BuildExt(build_ext):
    def build_extensions(self):
        # 检测编译器并设置优化标志
        extra_compile_args = []
        extra_link_args = []
        
        # 通用优化标志
        extra_compile_args.append('-O3')
        extra_compile_args.append('-ffast-math')
        extra_compile_args.append('-funroll-loops')
        
        # 平台特定优化
        if platform.system() == 'Windows':
            # Windows编译标志
            extra_compile_args.append('/O2')
            extra_compile_args.append('/arch:AVX2')
            extra_compile_args.append('/fp:fast')
            extra_compile_args.append('/openmp')
            extra_link_args.append('/openmp')
        else:
            # 类Unix系统优化
            if has_flag(self.compiler, '-mavx2'):
                extra_compile_args.append('-mavx2')
            elif has_flag(self.compiler, '-msse4.2'):
                extra_compile_args.append('-msse4.2')
            
            # OpenMP支持检测
            if has_flag(self.compiler, '-fopenmp'):
                extra_compile_args.append('-fopenmp')
                if platform.system() == 'Darwin':
                    # macOS需要特殊处理
                    extra_link_args.append('-lomp')
                else:
                    extra_link_args.append('-fopenmp')
            
            # macOS特定优化
            if platform.system() == 'Darwin':
                extra_compile_args.append('-stdlib=libc++')
                extra_compile_args.append('-mmacosx-version-min=10.13')
                extra_link_args.append('-stdlib=libc++')
                extra_link_args.append('-mmacosx-version-min=10.13')
        
        # C++标准版本
        if has_flag(self.compiler, '-std=c++17'):
            extra_compile_args.append('-std=c++17')
        elif has_flag(self.compiler, '-std=c++14'):
            extra_compile_args.append('-std=c++14')
        elif has_flag(self.compiler, '-std=c++11'):
            extra_compile_args.append('-std=c++11')
        
        # 应用标志到所有扩展
        for ext in self.extensions:
            ext.extra_compile_args = extra_compile_args
            ext.extra_link_args = extra_link_args
        
        build_ext.build_extensions(self)

# 检查是否可以构建C++扩展
try:
    import pybind11
    can_build_ext = True
except ImportError:
    can_build_ext = False
    print("Warning: pybind11 not found. C++ extension will not be built.")

ext_modules = []

if can_build_ext:
    try:
        # 尝试构建C++扩展
        ext_modules = [
            Extension(
                'autoproj._projection_cpp',
                ['autoproj/_projection_cpp.cpp'],
                include_dirs=[
                    get_pybind_include(),
                    get_pybind_include(user=True),
                ],
                language='c++',
            ),
        ]
        cmdclass = {'build_ext': BuildExt}
    except Exception as e:
        print(f"Warning: Failed to configure C++ extension: {e}")
        print("Falling back to pure Python implementation.")
        ext_modules = []
        cmdclass = {}
else:
    cmdclass = {}

setup(
    name="autoproj",
    version="0.5.0",
    author="AutoProj Development Team",
    description="High-precision 3D-to-2D projection engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "numpy>=1.20.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "cuda": ["cupy>=12.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pybind11>=2.10.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords="projection camera 3d 2d robotics computer-vision",
    license="MIT",
    url="https://github.com/autoproj/autoproj",
    project_urls={
        "Documentation": "https://autoproj.readthedocs.io/",
        "Bug Reports": "https://github.com/autoproj/autoproj/issues",
        "Source": "https://github.com/autoproj/autoproj",
    },
)
