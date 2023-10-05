""" Este código utiliza el entorno gym_controlador_I, con acciones aleatorias y discretas, 
 debido a que falta añadir una Policy para control del agente"""

import gymnasium as gym 
import gym_controlador_I # Importamos el paquete que hemos creado

# Creamos el entorno 
env =gym.make('gym_controlador_I/GridControlador-V0',render_mode="human") 
observation, info= env.reset(seed=42) # Reseteamos y obtenemos posición agente y error.

for _ in range(30):
    print(_)
    ## Poner la política aqui.
    action = env.action_space.sample() # Generamos una acción valida.
    observation, reward, terminated, truncated,info = env.step(action=action)
    if terminated or truncated: 
        observation, info = env.reset()
        print(observation, info)
env.close()

 