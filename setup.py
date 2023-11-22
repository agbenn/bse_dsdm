try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import bse_dsdm
print(dir(bse_dsdm))


def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='bse_dsdm',
    version="0.1.0",
    description='Barcelona School of Economics Data Science for Decision Making Library',
    author='Andrew Bennett',
    packages=find_packages(where='', exclude=['tests']),
    include_package_data=True,
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/danielavelez1997/hw4.git',
    classifiers=[
        'Programming Language :: Python :: 3.11.4'
    ]
)
