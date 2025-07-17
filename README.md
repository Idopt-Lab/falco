# Framework for Aircraft-Level Configuration and Optimization (FALCO)

[![GitHub Actions Test Badge](https://github.com/Idopt-Lab/falco/actions/workflows/actions.yml/badge.svg)](https://github.com/falco/flight-simulator/.github)
[![Coverage Status](https://coveralls.io/repos/github/Idopt-Lab/falco/badge.svg?branch=main&kill_cache=1)](https://coveralls.io/github/Idopt-Lab/falco?branch=main)
[![Forks](https://img.shields.io/github/forks/Idopt-Lab/falco.svg)](https://github.com/Idopt-Lab/falco/network)
[![Issues](https://img.shields.io/github/issues/Idopt-Lab/falco.svg)](https://github.com/Idopt-Lab/falco/issues)


## Description
FALCO is an in-development framework that enables researchers and engineers to simulate various aircraft configurations, flight conditions, and control methodologies through a modular, python-based framework built using [csdl_alpha](https://github.com/LSDOlab/CSDL_alpha) (Computational System Design Language). The framework is capable of integrating multi-fidelity aerodynamic models, propulsion models, and control systems to evaluate flight performance. This enables rapid gradient-based aircraft design optimization.

Key features include:
- Six-degree-of-freedom aircraft dynamics modeling
- Easily configurable aerodynamic, propulsion, and stability models capable of integration with existing software
- Modular architecture supporting custom aircraft configurations


FALCO is under active development. Contributions are welcome, and we encourage users to report issues or suggest features. Expect frequent updates and changes as we continue to enhance the framework's capabilities.

<!---
[![Python](https://img.shields.io/pypi/pyversions/csdl_alpha)](https://img.shields.io/pypi/pyversions/csdl_alpha)
[![Pypi](https://img.shields.io/pypi/v/csdl_alpha)](https://pypi.org/project/csdl_alpha/)
[![Coveralls Badge][13]][14]
[![PyPI version][10]][11]
[![PyPI Monthly Downloads][12]][11]
-->

## Installation instructions for users
FALCO can be installed using the following instructions: 

For direct installation with all dependencies, run on the terminal or command line

```sh
$ pip install git+https://github.com/Idopt-Lab/falco.git
```

## Installation instructions for developers
To install `FALCO`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
$ git clone https://github.com/Idopt-Lab/falco.git
$ pip install -e ./falco
```
More details instructions are available on the documentation page "Getting Started with FALCO" including what to do with a clean virtual environment. The package is available for Python 3.9 and later versions.

## Documentation
The documentation for FALCO is available at [DOCS LINK]() \[Need to fill link\].
The documentation includes:
- Getting started with FALCO
- Basic examples to familiarize users with the framework's core functionalities (Geometry, Solvers, Mass Properties, Condition Setup)
- Advanced examples (Trim Optimization, Geometry Parameterization, Coupled Design Optimization) \[WIP\]
- API reference for developers


## License
FALCO is licensed under GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](https://github.com/Idopt-Lab/falco/LICENSE.txt) file for more details.