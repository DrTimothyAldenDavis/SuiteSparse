from setuptools import find_packages, setup

setup(
    name='SPEXpy',
    packages=find_packages(include=['SPEXpy']),
    install_requires=['numpy','scipy'],
    version='0.1.0',
    description='Python interface for SPEX',
    author='Lorena Mejia Domenzain',
    license='SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later',
)

