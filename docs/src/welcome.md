

![alt text](/src/images/FALCO.jpg "FALCO Logo")


This documentation is intended to provide a basis for understanding the Framework for Aircraft-Level Configuration and Optimization (FALCO), an in-development framework for modeling and analyzing aircraft dynamics, control systems, flight performance, and more.

FALCO enables researchers and engineers to simulate various aircraft configurations, flight conditions, and control methodologies through a modular, python-based framework built using [csdl_alpha](https://github.com/LSDOlab/CSDL_alpha) (Computational System Design Language). The framework is capable of integrating multi-fidelity aerodynamic models, propulsion models, and control systems to evaluate flight performance. This enables rapid gradient-based aircraft design optimization.

Key features include:
- Six-degree-of-freedom aircraft dynamics modeling
- Easily configurable aerodynamic, propulsion, and stability models capable of integration with existing software
- Modular architecture supporting custom aircraft configurations

# Cite us
```none
@article{idopt2023,
        author = {Joseph Gould and Nathan Shune and David Solano Sarmiento and Darshan Sarojini},
        title = {Performance and Regulatory Assessment Trim Formulations for Conceptual Aircraft Design},
        booktitle = {AIAA AVIATION FORUM AND ASCEND 2025},
        year = {2025},
        doi = {10.2514/6.2025-3624},
        URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2025-3624},
        eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2025-3624}
}
```


# Table of Contents
```{toctree}
:maxdepth: 4

src/getting_started
src/examples
src/api
```