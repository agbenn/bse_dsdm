try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import bse_dsdm


def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='bse_dsdm',
    version="0.1.0",
    description='homework4',
    author='Andrew, Mariam,Julia,Daniela',
    packages=find_packages(where='', exclude=['tests']),
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/agbenn/bse_dsdm.git',
    classifiers=[
        'Programming Language :: Python :: 3.11.4'
    ]
)
