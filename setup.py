from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys

class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'autoproj._projection_cpp',
        ['autoproj/_projection_cpp.cpp'],
        include_dirs=[
            get_pybind_include(),
        ],
        language='c++',
        extra_compile_args=['-O3', '-ffast-math', '-std=c++11'],
    ),
]

setup(
    name='autoproj',
    version='0.1.0',
    description='High-precision 3D-to-2D projection engine for autonomous driving and robotics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ShareByWangYang',
    author_email='your.email@example.com',
    url='https://github.com/ShareByWangYang/autoproj',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.24',
        'pyyaml>=6.0',
        'pybind11>=2.10'
    ],
    extras_require={
        'cuda': ['cupy>=12.0'],
        'dev': ['pytest>=7.0', 'pytest-cov>=4.0']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Computer Vision',
    ],
    python_requires='>=3.8',
    cmdclass={'build_ext': build_ext},
)