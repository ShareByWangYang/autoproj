from setuptools import setup, Extension
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
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
