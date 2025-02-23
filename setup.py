from setuptools import setup


setup(
    name='neural_fdm',
    version='0.1.0',
    description='Combining differentiable mechanics simulations with neural networks',
    author='Rafael Pastrana',
    license='MIT',
    packages=["neural_fdm"],
    package_dir={"": "src"}
)
