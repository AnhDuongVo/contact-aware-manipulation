from .base import GROUPS, ContactEnv, RobotConfig
from .insertion import InsertionEnv
from .press import PressEnv
from .surface import SurfaceEnv
from .nuisance import Nuisance, sample as sample_nuisance

ENVS = {"insertion": InsertionEnv, "press": PressEnv, "surface": SurfaceEnv}


def make(task: str) -> ContactEnv:
    return ENVS[task]()
