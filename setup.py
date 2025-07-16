from setuptools import setup, find_packages

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='falco',
    version=get_version('falco/__init__.py'),
    author='Author name',
    author_email='author@gmail.com',
    license='LGPLv3+',
    keywords='python project template repository package',
    url='https://github.com/Idopt-Lab/falco',
    download_url='', #TODO: Add download URL
    # download_url='http://pypi.python.org/pypi/aircraft-flight
    description='An aircraft flight simulator for MDAO',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9,<3.11',
    platforms=['any'],
    install_requires=[
        'numpy==1.26.4',
        'vedo',
        'pint',
        'pandas',
        'scipy',
        'matplotlib',
        'sympy',
        'pytest',
        'cython==0.29.28',
        'graphviz',
        'jax',
        'casadi',
        'CSDL_alpha @ git+https://github.com/LSDOlab/CSDL_alpha.git',
        'modopt @ git+https://github.com/LSDOlab/modopt.git',
        'lsdo_geo @ git+https://github.com/LSDOlab/lsdo_geo.git',
        'lsdo_b_splines_cython @ git+https://github.com/LSDOlab/lsdo_b_splines_cython.git',
        'lsdo_function_spaces @ git+https://github.com/LSDOlab/lsdo_function_spaces.git',
        'NRLMSIS2 @ git+https://github.com/nichco/NRLMSIS2.git'
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
)
