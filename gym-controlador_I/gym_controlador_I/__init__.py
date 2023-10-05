from gymnasium.envs.registration import register

register(
    id="gym_controlador_I/GridControlador-V0",
    entry_point="gym_controlador_I.envs:GridControladorEnv",
    max_episode_steps=1000,
)