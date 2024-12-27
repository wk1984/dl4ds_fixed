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
    'numpy==1.19.5',
    'pandas==1.3.5',
    'protobuf==3.19.6',
    'scipy==1.7.1',
    'matplotlib==3.4.3',
    'xarray==0.19.0',
    'ecubevis==1.0.1',
    'tensorflow==2.6.0', 
    'tensorflow-gpu==2.6.0',
    'tensorflow-estimator==2.6.0',
    'keras==2.6.0',
    'scikit-learn==1.0',
    'opencv-python-headless==4.5.4.58',
    'joblib==1.1.1',
    'seaborn==0.11.2',
    'absl-py==0.14.1'
    ],
    extras_require={
        'horovod':['horovod'] 
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        ],
)