import bz2
import imp
import os
import sys

import pkg_resources
from setuptools import setup, find_packages

with open('README.md', 'r',encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='harmof0',
    version='0.0.1',
    description='HarmoF0 pitch tracker',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/WX-Wei/HarmoF0',
    author='weiweixing',
    author_email='wxwei20@fudan.edu.cn',
    entry_points = {
        'console_scripts': ['harmof0=harmof0.main:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='pitch tracking',
    project_urls={
        'Source': 'https://github.com/WX-Wei/HarmoF0',
        'Tracker': 'https://github.com/WX-Wei/HarmoF0/issues'
    },
    install_requires=[
        str(requirement)
        for requirement in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    # package_data={
    #     'checkpoints': ['mdb-stem-synth.pth'],
    # },
    packages=find_packages(),
    # packages=["HarmoF0",],
    # package_dir={'harmof0': 'harmof0'},
    include_package_data=True,
    data_files=['harmof0/checkpoints/mdb-stem-synth.pth',]
)