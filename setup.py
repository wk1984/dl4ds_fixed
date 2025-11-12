# !/usr/bin/env python

import os
import re
from setuptools import setup

def resource(*args):
    return os.path.join(os.path.abspath(os.path.join(__file__, os.pardir)),
                        *args)


with open(resource('dl4ds', '__init__.py')) as version_file:
    version_file = version_file.read()
    VERSION = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                        version_file, re.M)
    VERSION = VERSION.group(1)

with open(resource('README.md')) as readme_file:
    README = readme_file.read()

setup(
    name='dl4ds',
    packages=['dl4ds',
              'dl4ds.models',
              'dl4ds.training'],
    version=VERSION,
    description='Deep Learning for empirical DownScaling',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Carlos Alberto Gomez Gonzalez',
    license='Apache v2.0',
    author_email='carlos.gomez@bsc.es',
    url='https://github.com/carlgogo/dl4ds',
    keywords=[
        'deep learning', 
        'downscaling', 
        'super-resolution', 
        'neural networks',
        'Earth data',
        'EO' 
        ],
    install_requires=[
    'numpy>1,<2',
    'pandas<2',
    'scipy',
    'matplotlib<3.9',
    'seaborn',
    'opencv-python-headless',
    'xarray',
    'ecubevis',
    'tensorflow>2,<2.17',
    'keras>2,<2.17',
    'scikit-learn',
    'cartopy',
    'climetlab',
    'climetlab_maelstrom_downscaling'
    ],
    extras_require={
        'horovod':['horovod'] 
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        ],
)