"""
  Adaptive Divergence for rapid Adversarial Optimization
"""

from setuptools import setup, find_packages

from codecs import open
import os.path as osp


here = osp.abspath(osp.dirname(__file__))

with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


setup(
  name='advopt',

  version='1.0.0',

  description="""Adversarial Optimization.""",

  long_description=long_description,

  url='https://github.com/HSE-LAMBDA/rapid-ao',

  author='Maxim Borisyak',
  author_email='mborisyak at hse dot ru',

  maintainer = 'Maxim Borisyak',
  maintainer_email = 'mborisyak at hse dot ru',

  license='MIT',

  classifiers=[
    'Development Status :: 4 - Beta',

    'Intended Audience :: Science/Research',

    'License :: OSI Approved :: MIT License',

    'Programming Language :: Python :: 3',
  ],

  keywords=['Adversarial Optimization', 'Bayesian Optimization'],

  packages=find_packages('.'),
  package_dir={'': '.'},

  extras_require={
    'dev': ['check-manifest'],
    'test': ['nose>=1.3.0'],
  },

  install_requires=[
    'numpy==1.17.4',
    'scikit-optimize==0.5.2',
    'scipy==1.4.1',
    'tqdm==4.40.2',
    'torch==1.3.1',
    'scikit-learn==0.22',
    'catboost==0.21'
  ],
)
