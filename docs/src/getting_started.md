# Getting started
This page provides instructions for installing FALCO.

## Installation
FALCO has been tested on Windows platforms and may not work on other operating systems. Future implementations will be tested on Linux.

The minimum python version required is 3.9, and the recommended version is Python 3.10. These are the primary versions used for development and testing.

Before installation of FALCO, it is recommended to create a new anaconda virtual environment to avoid conflicts with other packages. To create a new virtual environment, you can use the following command in the terminal or command line:
```sh
$ conda create -n falco_env python=3.10
```
Then activate the virtual environment in your preferred python IDE or terminal.

```sh
$ conda activate falco_env
```
FALCO has a number of dependencies that need to be installed before using the package. The installation process can be automated using pip, which will install all required packages and their dependencies. If the automatic installation process does not automatically work, the recommended installation order of packages in a clean environment is provided at the bottom of the installation instructions section.

### Installation instructions for users
For direct installation with all dependencies in your new virtual environment, run on the terminal or command line

```sh
$ pip install git+https://github.com/Idopt-Lab/falco.git
```

### Installation instructions for developers
To install `FALCO`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
$ git clone https://github.com/Idopt-Lab/falco.git
$ pip install -e ./falco
```

### Package Dependencies Installation Order
The following is the recommended order of package installation in a clean environment if the automatic installation process does not work or leads to errors:

| Package                                                                 | Branch | Short Description                                                                                                                                                                                                                           |
|-------------------------------------------------------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [CSDL_alpha](https://github.com/LSDOlab/CSDL_alpha)                     | main   | Python-Based algebraic modeling language enabling automated adjoint sensitivity analysis                                                                      |
| [modopt](https://github.com/LSDOlab/modopt)                             | main   | A MODular development environment and library for OPTimization algorithms                                                                                      |
| [lsdo_geo](https://github.com/LSDOlab/lsdo_geo)                         | main   | Geometry engine for efficient manipulation of geometries via free-form deformation techniques                                                                  |
| [cython](https://cython.org/)                                           | 0.29.28    | Required for compiling Cython-based packages (lsdo_b_splines).                                                                                                |
| [lsdo_b_splines](https://github.com/LSDOlab/lsdo_b_splines_cython)      | main   | Cython-based package for efficient B-spline evaluation and manipulation                                                                                       |
| [lsdo_function_spaces](https://github.com/LSDOlab/lsdo_function_spaces) | main   | Package that enables the solver-independent field representation of field quantities via a host of functions that can be fit to solver data                    |
| [NRLMSIS2](https://github.com/nichco/NRLMSIS2)                       | main   | Python wrapper for the NRLMSIS 2.0 atmospheric model,<br> used for atmospheric density calculations                                                               |

Installation in order (can be copied and pasted into the terminal or command line):
```sh
pip install git+https://github.com/LSDOlab/CSDL_alpha.git
pip install git+https://github.com/LSDOlab/modopt.git
pip install git+https://github.com/LSDOlab/lsdo_geo.git
pip install cython==0.29.28
pip install git+https://github.com/LSDOlab/lsdo_b_splines_cython.git
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
pip install git+https://github.com/Idopt-Lab/nrlmsis2.git
pip install git+https://github.com/Idopt-Lab/falco.git
```
or, including the creation of a new conda environment, you can run the following commands in the terminal or command line:

```sh
conda create -n falco_env python=3.10
conda activate falco_env
pip install git+https://github.com/LSDOlab/CSDL_alpha.git
pip install git+https://github.com/LSDOlab/modopt.git
pip install git+https://github.com/LSDOlab/lsdo_geo.git
pip install cython==0.29.28
pip install git+https://github.com/LSDOlab/lsdo_b_splines_cython.git
pip install git+https://github.com/LSDOlab/lsdo_function_spaces.git
pip install git+https://github.com/nichco/NRLMSIS2.git
pip install git+https://github.com/Idopt-Lab/falco.git@dev-base-classes
```

If you are interested in manually editing any of the above packages, visit their documentation for developer installation instructions.

## Updating FALCO
When a new version of FALCO is released, you can update your installation by running the following command in the terminal or command line:

```sh
$ pip install --upgrade git+https://github.com/Idopt-Lab/falco.git
```

## Testing
To run all tests for this repository, navigate to the falco directory at `./falco` and, in the terminal, run
```sh
$ pytest
```

## Optimizers
FALCO is built using `csdl_alpha` and `modopt`, which are required for optimization. MODOPT provides a set of optimization algorithms, while CSDL provides the ability to create computational graphs. General purpose optimizers can be found in the `modopt` package, which is a dependency of FALCO. For more information on MODOPT, refer to the [MODOPT documentation](https://modopt.readthedocs.io/en/latest/).

MODOPT includes educational algorithms like Gradient Descent, Newton, Nelder-Mead, SQP, and more in addition to performant algorithms like IPOPT, SNOPT, and SLSQP. It is recommended to refer to the MODOPT documentation for more information on the available optimizers for your use case.