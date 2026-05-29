from setuptools import setup, find_packages

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
    install_requires=[
        'numpy>=1.24',
        'pyyaml>=6.0'
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
)