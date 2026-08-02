import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from .Model          import Model
from .ModelConfig    import ModelConfig

__all__ = ["Model", "ModelConfig"]