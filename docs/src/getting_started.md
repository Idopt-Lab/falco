# Getting started
This page provides instructions for installing FALCO.

## Before Installing
FALCO requires Python 3.9 or 3.10, and it is recommended to use Anaconda for package management. After installing Anaconda, make sure to add the anaconda folder to your system's PATH environment variable. This will allow you to use the `conda` command in your terminal or command line to create a new virtual environment and install packages.

Graphviz is required as part of the installation process, it is recommended to install Graphviz before installing FALCO. You can download Graphviz from the [Graphviz website](https://graphviz.org/download/). After downloading, make sure to add the Graphviz bin directory to your system's PATH environment variable.

The `lsdo_b_splines_cython` package is also required for FALCO. To ensure proper installation of a cython package, it is necessary to install cython and Visual Studio Build Tools 14+ (Current version is 17). You can download Visual Studio Build Tools from the [Visual Studio website](https://visualstudio.microsoft.com/downloads/). Make sure to select the "Desktop development with C++" workload during installation or modify the installed build tools afterwards to include it.

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

or, including the creation of a new conda environment, you can run the following commands in the terminal or command line:

```sh
conda create -n falco_env python=3.10
conda activate falco_env
pip install -e git+https://github.com/Idopt-Lab/falco.git
```

### Installation instructions for developers
To install `FALCO`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
$ git clone https://github.com/Idopt-Lab/falco.git
$ pip install -e ./falco
```

or, including the creation of a new conda environment, you can run the following commands in the terminal or command line:

```sh
conda create -n falco_env python=3.10
conda activate falco_env
git clone https://github.com/Idopt-Lab/falco.git
pip install -e ./falco
```

### Common Installation Issues
If you encounter issues during the installation process, it is recommended to check the following:
- Ensure that you have the correct version of Python installed (3.9 or 3.10).
- Make sure that you have added the anaconda and Graphviz bin directories to your system's PATH environment variable.
- If you are using Windows, ensure that you have installed Visual Studio Build Tools 14+ (Current version is 17) and selected the "Desktop development with C++" workload during installation.
- If you encounter issues with the `lsdo_b_splines_cython` package, ensure that you have installed Cython and Visual Studio Build Tools correctly.
- For NRLMSIS2, if issues are encountered in the usage of the package, it is recommended to visit the [NRLMSIS2 GitHub repository](https://github.com/nichco/NRLMSIS2), clone the repository, and install it using pip:
```sh
$ git clone https://github.com/nichco/NRLMSIS2.git
$ pip install -e ./NRLMSIS2
```
- If you encounter issues stemming from `lsdo_geo`, it is recommended to similarly clone the repository and install it using pip:
```sh
$ git clone https://github.com/LSDOlab/lsdo_geo.git
$ pip install -e ./lsdo_geo
```


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