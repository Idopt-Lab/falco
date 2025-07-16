# Getting started
This page provides instructions for installing FALCO.

## Installation
FALCO has been tested on Windows platforms and may not work on other operating systems. Future implementations will be tested on Linux.

### Installation instructions for users
For direct installation with all dependencies, run on the terminal or command line

```sh
$ pip install git+https://github.com/Idopt-Lab/aircraft-flight-simulator.git
```

### Installation instructions for developers
To install `FALCO`, first clone the repository and install using pip.
On the terminal or command line, run
```sh
$ git clone https://github.com/Idopt-Lab/aircraft-flight-simulator.git
$ pip install -e ./aircraft-flight-simulator
```

### Testing
To run all tests for this repository, navigate to the aircraft-flight-simulator directory at `./aircraft-flight-simulator` and, in the terminal, run
```sh
$ pytest
```
