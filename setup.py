import io
from setuptools import setup, find_packages 
import pathlib
import pkg_resources

with pathlib.Path('requirement.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name='less',
    packages=["less"],
    version='0.1',
    description='LESS',
    author='Mengzhou Xia',
    url='https://github.com/princeton-nlp/LESS',
    install_requires=install_requires,
    entry_points={
        "console_scripts": [],
    },
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)
