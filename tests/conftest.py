import numpy as np
import pytest

from imulab.experiment import get_plan
from imulab.sim import KinematicSim, ReplayConfig
from imulab.arm import Arm


@pytest.fixture(scope="session")
def arm():
    return Arm()


@pytest.fixture(scope="session")
def truth(arm):
    """10 s circular replay (ground-truth tool trajectory) shared by many tests."""
    plan, _ = get_plan("circular", 10.0, 0, arm)
    return KinematicSim(arm, ReplayConfig(seed=0)).replay(plan).truth
