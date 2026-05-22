"""Registry mapping env names to (EnvClass, CfgClass, ppo_module) triples.

Each env carries its own PPO module (``ppo_cartpole`` / ``ppo_double``), so the
two envs can be tuned independently. A ppo_module exposes ``PPO`` (the agent
class) and ``HYPERPARAMS`` (a dict of that env's training hyperparameters).

IMPORTANT: importing this module imports the env modules, which import
``isaaclab.*`` / ``isaaclab_assets.*``. Those imports only succeed AFTER
``AppLauncher`` has booted the Isaac Sim runtime. So import this module
inside the post-AppLauncher section of server.py / train.py / play.py,
never at the top of a script.
"""

import ppo_cartpole
import ppo_double
from cartpole_env import CartpoleEnv, CartpoleEnvCfg
from cart_double_pendulum_env import CartDoublePendulumEnv, CartDoublePendulumEnvCfg

# Keys here MUST match the argparse choices in server.py / train.py / play.py.
ENVS = {
    "cartpole": (CartpoleEnv, CartpoleEnvCfg, ppo_cartpole),
    "double": (CartDoublePendulumEnv, CartDoublePendulumEnvCfg, ppo_double),
}
