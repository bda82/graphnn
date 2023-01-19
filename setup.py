from setuptools import find_packages, setup

from versions import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gns",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "tensorflow",
        "pylint",
        "black",
        "mypy",
        "isort",
        "tqdm",
    ],
    url="https://gitlab.actcognitive.org/itmo-sai-code/graph-nn",
    license="GPL",
    author="SFU",
    author_email="dabespalov@sfedu.ru",
    description="Graph Neural Networks Basic Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
