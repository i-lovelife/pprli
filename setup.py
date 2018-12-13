from setuptools import find_packages, setup
import glob

setup(
    name='src',
    packages=find_packages(),
    scripts=glob.glob('scripts/*'),
    version='0.1.0',
    description='A short description of the project.',
    author='i-lovelife',
    license='',
)
