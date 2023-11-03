import gymnasium as gym # Importamos la libreria gymnasium
#import gym_controlador_I
import numpy as np
from Gym_controlador_vII import GridControladorEnvII
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

## Implementación de Q learnig en sistema controladora motor, en este entorno las acciones aún son 
## discretas, pero se ha logrado actualizar acciones (valor de ganancias), para cada paso de tiempo.
## Es necesario modificar:
# ---- Acciones a valores o incrementos más pequeños
#----- El umbral esta en 40 para forzar el estado terminal, debe ser mucho menor
#----- Quiza definer una estrategia que permita cuantificar mejor el error, es decir, cuando
#------  un valor de error por debajo del umbral se repita por un periódo de tiempo decir entonces
##------ que tenemos un estado terminal.

num_actions=4 ## Acciones posibles del agente.

def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities= np.ones(num_actions, dtype=float) * epsilon/num_actions
        #print('Action_probabilities', Action_probabilities) 
        best_action = np.argmax(Q[state]) # Indice donde se encuentra el valor maximo de Q[state]

        Action_probabilities[best_action]+=(1.0-epsilon)  ## Distribución de probabilidad original
        return Action_probabilities, best_action # Retorna probabilidades p/cada acción y el indice con la acción de mayor probabilidad.
    return policyFunction

def data_in():
    print("Teclee alguna de las siguientes opciones, seguida de enter ")
    print("1 si utilizamos tabla Q vacia, 2 si cargaremos condiciones iniciales de tabla Q")
    dato=input()

    if int(dato)==1:
        Q = defaultdict(lambda: np.zeros(4)) #Inicializamos a cero la tabla Q.
    else:
        print("Ingrese el nombre del archivo + .npy")
        archivo_r=input()
        P=np.load(archivo_r, allow_pickle=True) ## Cargamos el objeto array que contiene al diccionario
        Q = defaultdict(lambda: np.zeros(4)) # Inicializamos todo a cero.
        Q.update(P.item()) #Recreamos el defaultdict uniendo los dos diccionarios.
        #print(Q)
    print("Introduzca el nombre del archivo donde se guardara la tabala Q, seguido de .npy")
    archivo_w=input()

    return Q, archivo_w

def action_type(policy,state, epsilon ):
    action_probabilities, best_action = policy(state) ## Probabilidades de acción y el indice de la mejor acción.
    ##->>> Def. tipo acción: aleatoria (exploración) o e-greedy (explotación)
    if rd.random()<epsilon:
          #print('acción aleatoria ')
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    else:
        #print('acción greedy')
        #action= np.argmax(action_probabilities) # Valor maximo dentro probabilidades
        #action= best_action
        action=best_action

    return action

def qLearning(env,num_actions, num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,x0,Tl):
    
    Q,archivo_w=data_in()
    #Politica epsilon-greedy
    policy= createEpsilonGreedyPolicy(Q, epsilon, num_actions)

    for ith_episode in range(num_episodios):
        print("Episodio número: ",ith_episode)
       
        posicion= env.reset(seed=42)                 ## Posición inicial
        shape=(env.size,env.size)                    ## Dimensión del grid
        state = np.ravel_multi_index(tuple(posicion),shape) ## Position in shape
        #print('state',state)
        ## State inicializa la politica en el siguiente loop.
        time.sleep(0.1)

                    ### Vector solución (lista) y condiciones iniciales.
        x_sol_q=[]
        x_sol_q.append(x0)
        E_integral=0 #Cada episodio se resetea a 0 el error integral
        
        for t in itertools.count(): 
            i=t+1 # ya que i=0 corresponde a condiciones iniciales
            action=action_type(policy, state, epsilon)
            #print('action',action)

            x_sol=x_sol_q[i-1][:] #<<-vector 1x3, i,pos, vel o tambien x_sol=x_sol_q[0] toda la fila 0
            I_error= I_reference - x_sol_q[i-1][0] #Error en corriente
            E_integral=E_integral+ (I_error*h)

            next_posicion, reward, terminated,truncated,x_next, info = env.step(I_error, E_integral, h, x_sol,Tl,I_reference,error_umbral,action)

            x_sol_q.append(x_next)
            next_state =np.ravel_multi_index(tuple(next_posicion),shape)
            time.sleep(0.001)

            best_next_action = np.argmax(Q[next_state]) #Calculamos la mejor acción del sig. state.

            ###---->>>>>> Aplicando la diferencia temporal
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if t%50==0 :
                np.save(archivo_w, np.array(dict(Q)))
            

            if terminated: ## Reemplazar por terminated o truncated?
                print("Se alcanzo un estado terminal!!!")
                print(" Con un error porcentual de:", info)
                #env.render()
                time.sleep(1)
                break
            elif truncated:
                print("Truncado por rebasar Imax u obtener un valor NAN")
                break
                

            state = next_state #Actualizamos state
    return Q

## Argumentos que serian pasados a la función Qlearning
env=GridControladorEnvII() ## Falta revisar la parte de renderizado.

num_actions=4
num_episodios=10
discount_factor=0.2
alpha=0.6
epsilon=0.5

error_umbral=40
h=0.001 #Tamaño del paso
I_reference=0.2 # En amperes

##Condiciones iniciales p/ecuaciones del motor.
In_current= 0          # Initial current 
In_position= 0         # Initial angular position 
In_velocity= 0         # Initial angular velocity

x0=np.array([ In_current, In_position, In_velocity ])  
Tl=0 # Entradas externas, podría cambiar para simular una interacción o perturbación.



Q=qLearning(env,num_actions,num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,x0,Tl) 

print(Q) 

## Revisar esta última parte y tratar de manener el trabajo con csv y dataframe.