"""Teacher CNN package: `import teacher` from workspace/ exposes the API in teacher/teacher.py."""
from .teacher import *  # noqa: F401,F403
from .teacher import load  # explicit, in case __all__ is not defined
