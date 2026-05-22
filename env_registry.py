"""Registry mapping env names to (EnvClass, CfgClass) pairs.

IMPORTANT: importing this module imports the env modules, which import
``isaaclab.*`` / ``isaaclab_assets.*``. Those imports only succeed AFTER
``AppLauncher`` has booted the Isaac Sim runtime. So import this module
inside the post-AppLauncher section of server.py / train.py / play.py,
never at the top of a script.
"""

from cartpole_env import CartpoleEnv, CartpoleEnvCfg
from cart_double_pendulum_env import CartDoublePendulumEnv, CartDoublePendulumEnvCfg

# Keys here MUST match the argparse choices in server.py / train.py / play.py.
ENVS = {
    "cartpole": (CartpoleEnv, CartpoleEnvCfg),
    "double": (CartDoublePendulumEnv, CartDoublePendulumEnvCfg),
}
