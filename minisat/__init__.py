from .minisat.gym.MiniSATEnv import gym_sat_Env

import gym


def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point
    )


# Graph-Q-SAT UPD: register the sat environment
register(id="sat-v0", entry_point="minisat.minisat.gym.MiniSATEnv:gym_sat_Env")
