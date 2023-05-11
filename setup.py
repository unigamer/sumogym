from setuptools import setup, find_packages

setup(
    name="sumogym",
    version="0.0.1",
    install_requires=["gymnasium", "pybullet", "numpy", "simple-pid", "pillow"],
    packages=find_packages(include=['sumogym'])
)
