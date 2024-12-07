from setuptools import setup

setup(
    name="sliceduot",
    distname="",
    version='0.1.0',
    description="Fast computation of sliced-unbalanced-OT problems",
    packages=['sliceduot'],
    install_requires=[
              'numpy',
              'torch',
              'matplotlib',
              'tqdm',
              'pytest'
    ],
    license="MIT",
)
