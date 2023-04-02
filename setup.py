from setuptools import setup, find_packages
from pathlib import Path


def read_file(filename: str) -> str:
    with open(Path(__file__).parent / filename, mode='r', encoding='utf-8') as f:
        return f.read()


def get_package_info(package_name: str) -> dict:
    package_info_dict = {}
    with open(str(Path(__file__).parent / package_name / 'package_info.py'), 'r') as f:
        exec(f.read(), {}, package_info_dict)

    return package_info_dict


PACKAGE_NAME = 'torchmcee'

package_info = get_package_info(PACKAGE_NAME)

__version__ = package_info['__version__']
__license__ = package_info['__license__']
__author__ = package_info['__author__']
__email__ = package_info['__email__']

classifiers = package_info['classifiers']
description = package_info['description']

long_description = read_file('README.md')
long_description_content_type = 'text/markdown'

python_requires = '>=3.6'
install_requires = read_file('requirements.txt').splitlines()

setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    version=__version__,
    author=__author__,
    author_email=__email__,
    license=__license__,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    python_requires=python_requires,
    classifiers=classifiers,
    install_requires=install_requires,
)
