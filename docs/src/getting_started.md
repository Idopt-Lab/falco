# Getting started
This page provides instructions for installing FALCO.

## Installation
FALCO has been tested on Windows platforms and may not work on other operating systems. Future implementations will be tested on Linux.

The minimum python version required is 3.9, and the recommended version is Python 3.10. These are the primary versions used for development and testing.

Before installation of FALCO, it is recommended to create a new virtual environment to avoid conflicts with other packages. To create a new virtual environment, you can use the following command in the terminal or command line:

```sh
$ python -m venv falco_env
```
Then activate the virtual environment in your preferred python IDE or terminal.

If the automatic installation process does not automatically work, the recommended installation order of packages in a clean environment is provided at the bottom of the getting started page.

### Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line

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

### Updating FALCO
When a new version of FALCO is released, you can update your installation by running the following command in the terminal or command line:

```sh
$ pip install --upgrade git+https://github.com/Idopt-Lab/falco.git@<main>
```

### Testing
To run all tests for this repository, navigate to the falco directory at `./falco` and, in the terminal, run
```sh
$ pytest
```

### Optimizers
FALCO is built using `csdl_alpha` and `modopt`, which are required for optimization. MODOPT provides a set of optimization algorithms, while CSDL provides the ability to create computational graphs. General purpose optimizers can be found in the `modopt` package, which is a dependency of FALCO. For more information on MODOPT, refer to the [MODOPT documentation](https://modopt.readthedocs.io/en/latest/).

MODOPT includes educational algorithms like Gradient Descent, Newton, Nelder-Mead, SQP, and more in addition to performant algorithms like IPOPT, SNOPT, and SLSQP. It is recommended to refer to the MODOPT documentation for more information on the available optimizers for your use case.

### Package Dependencies Installation Order
The following is the recommended order of package installation in a clean environment if the automatic installation process does not work or leads to errors:

```markdown
| Package      | Branch   | Required | Description                | Notes                      |
|--------------|----------|----------|----------------------------|----------------------------|
| numpy        | >=1.21   | Yes      | Core numerical library     | Install first              |
| csdl_alpha   | latest   | Yes      | Computational graphs       | Dependency for FALCO       |
| modopt       | latest   | Yes      | Optimization library       | Dependency for FALCO       |
| pytest       | latest   | Optional | For running tests          | Only needed for testing    |
| matplotlib   | latest   | Optional | For plotting results       | Useful for visualization   |
```