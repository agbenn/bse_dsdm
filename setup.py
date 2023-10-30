try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import library_hw4


def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='library_hw4',
    version=library_hw4.__version__,
    description='homework4',
    author='Andrew, Mariam,Daniela',
    packages=find_packages(where='', exclude=['tests']),
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/danielavelez1997/hw4.git',
    classifiers=[
        'Programming Language :: Python :: 3.11.4'
    ]
)
