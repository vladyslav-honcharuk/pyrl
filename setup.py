"""
Setup script for pyrl package.
"""
from setuptools import setup, find_packages

setup(
    name='pyrl_torch',
    version='0.1.0',
    description='PyTorch implementation of PyRL - Reward-based training of RNNs for cognitive tasks',
    author='Vladyslav Honcharuk',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.9.0',
        'matplotlib>=3.3.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'jupyter>=1.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
