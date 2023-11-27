import gymnasium as gym # Importamos la libreria gymnasium
#import gym_controlador_I
import numpy as np
from Gym_controlador_vIII import GridControladorEnvIII
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
import matplotlib.pyplot as plt

## Implementación de Q learnig en sistema controladora motor, en esta versión se busco aplicar 
# correcciones respecto a la señal que devuelve el step como estado u observación, en este caso corriente. 
# Ya no se manejara un arreglo cuadrado de ganancias, por el contrario se manejara un vector de 7801 
# estados, comenzado en 0  y terminando en 7800. # El estado_0=-3.9A, estado_3900=0A y estado 7800=3.9A, 
# la distancia entre estados sera de 1 mA.


#Función que define la politica a seguir.
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    def policyFunction(state):
        Action_probabilities= np.ones(num_actions, dtype=float) * epsilon/num_actions
        #print('Action_probabilities', Action_probabilities) 
        best_action = np.argmax(Q[state]) # Indice donde se encuentra el valor maximo de Q[state]

        Action_probabilities[best_action]+=(1.0-epsilon)  ## Distribución de probabilidad original
        return Action_probabilities, best_action # Retorna probabilidades p/cada acción y el indice con la acción de mayor probabilidad.
    return policyFunction

#Función que permite decidir si comenzar de cero o ingresar una tabla Q previa.
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
    print("Introduzca el nombre del archivo donde se guardara la tabala Q, sin ninguna extensión")
    archivo_w=input()

    return Q, archivo_w

#Función que permite decidir si aplicar acción aletoria o ambiciosa.
def action_type(policy,state, epsilon ):
    action_probabilities, best_action = policy(state) ## Probabilidades de acción y el indice de la mejor acción.

    if rd.random()<epsilon:
          #print('acción aleatoria ')
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    else:
        action=best_action
    return action

#Función que realiza el proceso de aprendizaje.
def qLearning(env,num_actions, num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,Tl):
    
    Q,archivo_w=data_in()
    policy= createEpsilonGreedyPolicy(Q, epsilon, num_actions)

    for ith_episode in range(num_episodios):  
       
        v_solucion, ganancias= env.reset() # Se genera vector de condiciones iniciales así como
        # un arreglo de ganancia aleatorias de entre 0 y 100, con paso de 0.001
        
        #Convertimos un valor de corriente a un estado dentro de un intervalo de 0 a 7800.
        state= int(round((v_solucion[0]*1000)+3900,3)) 
        Retorno=0  #Icialización acumulado de recompensa por episodio.    
        
        for t in itertools.count(): 
            action=action_type(policy, state, epsilon) #Generamos una acción, aleatoria o ambiciosa.
            #Procedemos a realizar un step y recibimos vector solución, recompensa, estado, error, ganancias.
            v_solucion_next, reward, terminated,truncated,info, ganancias = env.step(h,Tl,I_reference,error_umbral,action)
            
            Retorno+=reward 
            next_state=int(round((v_solucion_next[0]*1000)+3900,3)) #Definimos estado siguiente.
            best_next_action = np.argmax(Q[next_state]) #Calculamos la mejor acción del sig. estado.

            ###--->>>>>> Aplicando la diferencia temporal
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            #En función de la condición del estado, decidimos continuar iterando o pasar al sig. episodio.
            if truncated:
                print("Truncado por rebasar Imax o valor NAN")
                break

            if terminated:
                print(" Estado terminal, con error y ganancias:", info, ganancias/100)
                print("Retorno",Retorno, "  Episodio", ith_episode )
                time.sleep(3)
                break
            #Actualizamos el estado
            state = next_state 
        #Condicionales para respaldar la información generada.
        if ith_episode>0 and ith_episode%50==0:
            print('Guardo en episodio #', ith_episode)
            print('Guardado')
            np.save(archivo_w+".npy", np.array(dict(Q)))
        elif ith_episode% 10==0:
            print("Episodio #:",ith_episode)
        
    return Q,archivo_w #Se retorna la tabla Q y el nombre del archivo.

## ------->> configuraciń previa al aprendizaje con Qlearning <<<<---------
env=GridControladorEnvIII()

num_actions=4         #  Número de acciones posibles
num_episodios=100000  ## <<<<<-----------------------Número de episodios
discount_factor=0.5   #  Factor de descuento
alpha=0.9             #  Factor de aprendizaje 
epsilon=0.5           #  Tasa de exploración
error_umbral=5
h=0.001               #  Tamaño del paso
I_reference=0.2       #  Amperes
Tl=0                  # Entradas externas
#podría cambiar para simular una interacción o perturbación.


###--------------- >>>>  Estructura principal   <<<<<<<----------------
###                      Aprendizaje 
Q, nombre_archivo=qLearning(env,num_actions,num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,Tl) 

###------->>>>Guardado de la tabla Q y creación de tabla excel para visualización de datos.
D=dict(Q)
    
def s_a(estado,accion):
    if accion==0:
        return estado[0]
    elif accion==1:
        return estado[1]
    elif accion==2:
        return estado[2]
    else:
        return estado[3]
#Guardamos en excel para visualización de la información
df2 = pd.DataFrame([[key, s_a(D[key],0), s_a(D[key],1), s_a(D[key],2), s_a(D[key],3) ] for key in D.keys()], columns=['Estado', 'sa1','sa2','sa3', 'sa4'])
nombre2=nombre_archivo+".xlsx"
df2.to_excel(nombre2) # Guardamos en excel