from setuptools import find_packages, setup

setup(
    name='sevn_model',
    packages=find_packages(),
    version='1.0.0',
    install_requires=['gym', 'matplotlib', 'pybullet', 'torch', 'tensorflow==1.14'])
