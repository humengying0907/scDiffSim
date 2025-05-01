import logging
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

from ._mymodel import MyModel, MyModule
from ._mypyromodel import MyPyroModel, MyPyroModule

from importlib.metadata import version, PackageNotFoundError

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("simple_scvi: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["MyModel", "MyModule", "MyPyroModel", "MyPyroModule"]

# __version__ = version("simple-scvi")

try:
    __version__ = version("simple-scvi")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default version if not installed
