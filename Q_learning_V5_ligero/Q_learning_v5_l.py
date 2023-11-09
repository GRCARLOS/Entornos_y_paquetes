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
import matplotlib.pyplot as plt

## Implementación de Q learnig en sistema controladora motor, en este entorno las acciones 
# aún son discretas, pero se ha creado un espacio de size*size, donde size=100 0 1000 según
# sea el caso, esto permite ampliar el espacio de busqueda.

# Este codigo trabaja relativamente igual que la versión Qlearning_v5, sin embargo ha sido 
## modificado  para que solo proporcione # de la iteración cada 10. Así mismo se desabilito la
## la graficación en tiempo real, y solo se grafica el comportamiento si se alcanza un estado 
# terminal.

# Esta versión esta configurada para trabajar en un grid de 50*50= 2500 estados
## --    ->>> IMPORTANTE  IMPORTANTE IMPORTANTE <<<<<<----------
# para cambiar estas dimensiones, debe modifcarse size  en el metodo __init__ de
#  GymControlador_vII.
#  Tambien cambiar el entero utilizado para calcular GX=self._agent_location/10, en función de 
# los incrementos deseados.




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
    print("Introduzca el nombre del archivo donde se guardara la tabala Q, sin ninguna extensión")
    archivo_w=input()

    return Q, archivo_w

def action_type(policy,state, epsilon ):
    action_probabilities, best_action = policy(state) ## Probabilidades de acción y el indice de la mejor acción.

    if rd.random()<epsilon:
          #print('acción aleatoria ')
        action = np.random.choice(np.arange(len(action_probabilities)), p = action_probabilities)
    else:
        action=best_action
    return action

def qLearning(env,num_actions, num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,x0,Tl):
    
    Q,archivo_w=data_in()
    policy= createEpsilonGreedyPolicy(Q, epsilon, num_actions)

    for ith_episode in range(num_episodios):
        #print("Episodio número: ",ith_episode)
       
        posicion= env.reset()  
    
        shape=(env.size,env.size)                    ## Dimensión del grid
        state = np.ravel_multi_index(tuple(posicion),shape) ## Position in shape
        time.sleep(0.01)

                    ### Vector solución (lista) y condiciones iniciales.
        x_sol_q=[]
        x_sol_q.append(x0)
        E_integral=0 #Cada episodio se resetea a 0.

        ## Conf. figura para graficar
        fig, ax = plt.subplots()
        fig.suptitle('Estimated Current in the motor')
        ax.set_ylabel('Current (A)')
        ax.set_xlabel('Time (ms)')
        xdata, ydata = [0], [x0[0]] # Lista que almacena los valores a graficar.
        Retorno=0       # Acumulado de recompensa
        
        for t in itertools.count(): 
            i=t+1 # ya que i=0 corresponde a condiciones iniciales
            action=action_type(policy, state, epsilon)

            x_sol=x_sol_q[i-1][:] #<<-vector 1x3, i,pos, vel 
            I_error= I_reference - x_sol_q[i-1][0] #Error en corriente
            E_integral=E_integral+ (I_error*h)

            next_posicion, reward, terminated,truncated,x_next, info, ganancias = env.step(I_error, E_integral, h, x_sol,Tl,I_reference,error_umbral,action)

            x_sol_q.append(x_next)
            Retorno+=reward

            next_state =np.ravel_multi_index(tuple(next_posicion),shape)
            best_next_action = np.argmax(Q[next_state]) #Calculamos la mejor acción del sig. state.

            ###---->>>>>> Aplicando la diferencia temporal
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if truncated:
                #print("Truncado por rebasar Imax o valor NAN")
                plt.close()
                break

            xdata.append(i) ## Vector de tiempo
            ydata.append(x_next[0]) ##Vector solución
            #if t>3:
                #plt.scatter(xdata,ydata,color = "green")
                #plt.pause(0.001)

            if terminated:
                #print("Se alcanzo un estado terminal!!!")
                print(" Estado terminal, con error y ganancias:", info, ganancias)
                print("Retorno",Retorno, "  Episodio", ith_episode )
                plt.scatter(xdata, ydata,color = "red")
                plt.pause(1)
                plt.close()
                break
            
                

            state = next_state #Actualizamos state
            #time.sleep(0.1)
        if ith_episode>0 and ith_episode%50==0:
            print('Guardo en episodio #', ith_episode)
            print('Guardado')
            np.save(archivo_w+".npy", np.array(dict(Q)))
        elif ith_episode% 10==0:
            print("Episodio #:",ith_episode)
        
    return Q,archivo_w

## ------->>  Argumentos que configuran el aprendizaje Qlearning <<<<---------
env=GridControladorEnvII()

num_actions=4
num_episodios=110  ## <<<<<-----Número de episodios
discount_factor=0.5
alpha=0.9
epsilon=0.5
error_umbral=10
h=0.001 #Tamaño del paso
I_reference=0.2 # Amperes

##Condiciones iniciales p/ecuaciones del motor.
In_current= 0          # Initial current 
In_position= 0         # Initial angular position 
In_velocity= 0         # Initial angular velocity

x0=np.array([ In_current, In_position, In_velocity ])  
Tl=0 # Entradas externas, podría cambiar para simular una interacción o perturbación.


###------ >>>>  Estructura Main, Aprendizaje y guradado de los datos.
Q, nombre_archivo=qLearning(env,num_actions,num_episodios, discount_factor, alpha, epsilon,error_umbral,h,I_reference,x0,Tl) 

D=dict(Q)
#df = pd.DataFrame([[key, D[key]] for key in D.keys()], columns=['Estado', 'Valor estado/acción']) 
#nombre = nombre_archivo+".xlsx"
#df.to_excel(nombre) # Gurdamos en excel

shape=(env.size,env.size) 
def gain(state, shape,k):
    ganancia=np.unravel_index(state, shape)
    if k==0:
        return ganancia[0]
    else:
        return ganancia[1]
    
def s_a(estado,accion):
    if accion==0:
        return estado[0]
    elif accion==1:
        return estado[1]
    elif accion==2:
        return estado[2]
    else:
        return estado[3]
#Guradamos en excel para visualización de la información
df2 = pd.DataFrame([[key, s_a(D[key],0), s_a(D[key],1), s_a(D[key],2), s_a(D[key],3), gain(key,shape,0), gain(key, shape, 1) ] for key in D.keys()], columns=['Estado', 'sa1','sa2','sa3', 'sa4', 'Kp', 'Kd'])
nombre2=nombre_archivo+".xlsx"
df2.to_excel(nombre2) # Guardamos en excel