import gymnasium as gym # Importamos la libreria gymnasium
import gym_controlador_I
import numpy as np
from Gym_controlador import GridControladorEnv
from collections import defaultdict
from email.policy import default
from ssl import ALERT_DESCRIPTION_PROTOCOL_VERSION
import itertools
import matplotlib
import matplotlib.style
import pandas as pd
import sys
import random as rd
import time
#from os import stat

#import plotting ## Script original del windyworld.
#matplotlib.style.use('ggplot')

num_actions=4 ## Acciones posibles del agente.

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities= np.ones(num_actions, dtype=float) * epsilon/num_actions
        #print('Action_probabilities', Action_probabilities) 
        best_action = np.argmax(Q[state]) # Indice donde se encuentra el valor maximo de Q[state]

        Action_probabilities[best_action]+=(1.0-epsilon)  ## Distribución de probabilidad original
        return Action_probabilities, best_action # Retorna probabilidades p/cada acción y el indice con la acción de mayor probabilidad.
    return policyFunction


def qLearning(env, num_episodios, discount_factor=0.2, alpha=0.6, epsilon=0.5):
    Q = defaultdict(lambda: np.zeros(4)) #Inicializamos a cero la tabla Q.
    
    # Se crea una politica epsilon-greedy
    policy= createEpsilonGreedyPolicy(Q, epsilon, num_actions) 

    # Iteramos sobre el numero de episodios
    for ith_episode in range(num_episodios):
        posicion, info = env.reset(seed=42) # Obtenemos una posición inicial (state) e info del errror porcentual.
        shape=(env.size,env.size)  #shape=(10,10), dimensiones del grid para esta implementación.
        state = np.ravel_multi_index(tuple(posicion),shape) ## Devuelve el estado de "posición inicial" dentro de shape.
        
        time.sleep(0.1)
        # State generado en reset, inicializa la politica en la siguiente iteración.

        for t in itertools.count(): # .Count, itera indefinidamente, salvo que se agregue una condición.
            action_probabilities, best_action = policy(state) ## Obtememos probabilidades de acción y el indice de la mejor acción.

            ##------->>> Def. tipo acción: aleatoria (exploración) o e-greedy (explotación)
            if rd.random()<epsilon:
                #print('acción aleatoria ')
                action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
            else:
                #print('acción greedy')
                action= np.argmax(action_probabilities) # Valor maximo dentro de la lista de probabilidades
                action= best_action
         
            ##------->>>> Hacemos step con la acción generada.
            next_posicion, reward, terminated,truncated, info = env.step(action=action)


            next_state =np.ravel_multi_index(tuple(next_posicion),shape)
            #print('Estado siguiente', next_state)
            
            # No renderizamos aquí. Render viene implicito en el step de GridControladorEnv.
            time.sleep(0.05)

            ## ---->>> No se adapto el codigo de estadisticas de windiworld para esta implementación.
            # stats.episode_lengths[ith_episode] = t  
            # stats.episode_rewards[ith_episode] += reward

            best_next_action = np.argmax(Q[next_state]) #Calculamos la mejor acción del sig. state.

            ###---->>>>>> Aplicando la diferencia temporal
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta


            if terminated: ## Reemplazar por terminated o truncated?
                print("Se alcanzo un estado terminal!!")
                print(" Con un error porcentual de:", info)
                #env.render()
                time.sleep(1)
                break

            state = next_state #Actualizamos state
    env.close()
    return Q


env=GridControladorEnv(render_mode="human")

Q=qLearning(env,3) #Entrenamiento para 3 episodios.