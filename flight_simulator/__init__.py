__version__ = '0.0.1'

from pint import UnitRegistry
from pathlib import Path


ureg = UnitRegistry()
Q_ = ureg.Quantity

REPO_ROOT_FOLDER = Path(__file__).parent.parent
