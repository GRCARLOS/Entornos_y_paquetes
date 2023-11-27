import numpy as np            ## Importamos librerias
import pygame
import gymnasium as gym
from gymnasium import spaces
###----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
###----------------------------------------------

class GridControladorEnvIII(gym.Env): ##Heredamos de la clase gym

###------------>>>> Metodos init <<<<-----------------
    def __init__(self,size=7081,size2=100000):
        self.size= size #Tamaño del grid
        self.size2=size2   
        
        #Def. observaciones como diccionario con la ubicación del agente.
        self.observation_space= spaces.Dict(
            {
                "vector solución": spaces.Box(0, size-1, shape=(3,), dtype=float),
                "ganancias": spaces.Box(0, size2-1, shape=(2,), dtype=int),
            }
        )

        self.action_space=spaces.Discrete(4)
        
        """ El siguiente diccionario mapea las acciones de'self.action_space' en la dirección
        en la que debe moverse el agente si determinada acción es tomada. I.e. 0 corresponde a
        moverse a la derecha (right), 1 moverse hacia arriba (up), etc. """

        self._action_to_direction = {
            0: np.array([1, 0]),    # Derecha
            1: np.array([0, 1]),    # Arriba
            2: np.array([-1, 0]),   # Izquierda
            3: np.array([0, -1]),    # Abajo
        }


##------------>>>> métodos  para calcular salidas del motor y señal de control <<<<-------------------------
    ### Controlador
    def _PI_control(self,ganancias, error, E_integral): 
        Kp,Ki=ganancias
        Pout=Kp*error
        Iout=Ki*E_integral
        return Pout+Iout
    
    ###Modelo motor
    def _model_motor(self,x, u, Tl): ## u señal control, Tl torque externo, x vector de estado
        R=0.343            # Impedance
        L = 0.00018        # Inductance
        kb=0.0167          # contraelectromotriz constant
        Jm=2.42*10**-6     # Motor inertia
        kt=0.0167          # Torque constant
        B=5.589458*10**-6
        current=x[0]       # Current
        theta=x[1]         # Angular postion
        theta_dot=x[2]     # Angular velocity

        current_dot=(1/L)*u -(R/L)*current -(kb/L)*theta_dot       # Derivative of current
        theta_ddot= (kt/Jm)*current -(B/Jm)*theta_dot - (1/Jm)*Tl  # Angular aceleration
        return np.array([ current_dot, theta_dot, theta_ddot])  # ret
    
    def error_porcentual(self, i_referencia, i_estimada):
        return abs((i_referencia-i_estimada)/i_referencia)*100

##---------->>>> Método Reset <<<------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        ##Condiciones iniciales p/ecuaciones del motor.
        In_current= 0          # Initial current 
        In_position= 0         # Initial angular position 
        In_velocity= 0         # Initial angular velocity
        x0=np.array([ In_current, In_position, In_velocity ]) 

        self._vector_solucion = x0 #Contien los 3 estados iniciales
        ##Generamos ganancias aleatorias
        self._ganancias=self.np_random.integers(0, self.size2, size=2,dtype=int)
        return self._vector_solucion, self._ganancias

##---------->>>> Método Reset <<<------------------------------------------------------------------
    def step(self,h, Tl, I_reference,error_umbral, action): 
        direction=self._action_to_direction[action] 
        self._ganancias = np.clip(           
        self._ganancias + direction, 0, self.size2 -1 #Limitamos ganancias
        )

        GX=self._ganancias/1000 ## Las ganancias pueden variar de 0 a 100, con incrementos de 0.001
        print('Ganancia aplicada ',GX)

        ###<-----Config. antes de iterar para hayar el estado estacionario
        E_integral=0 #Se resetea a 0 antes de iterar
        x_sol=[self._vector_solucion] # Icializamos vector solución temporal antes de cada prueba
        ##Configuración de graficado
        fig, ax = plt.subplots()
        fig.suptitle('Estimated Current in the motor')
        ax.set_ylabel('Current (A)')
        ax.set_xlabel('Time (ms)')
        xdata, ydata = [0], [self._vector_solucion[0]] 
        cespecial=False

        for t in itertools.count(): 
            i=t+1 # i=0 corresponde a condiciones iniciales

            I_error= I_reference - x_sol[i-1][0] #Error en corriente
            E_integral=E_integral+ (I_error*h)
            
            u=self._PI_control(GX,I_error, E_integral)    # Control signal

            ## Calculamos los terminos de RK-DP para el paso de tiempo actual
            k1=h*self._model_motor(x_sol[i-1], u, Tl)
            k2=h*self._model_motor(x_sol[i-1]+ (k1/5), u,Tl)
            k3=h*self._model_motor(x_sol[i-1]+ (3/40)*k1 + (9/40)*k2, u,Tl)
            k4=h*self._model_motor(x_sol[i-1]+ (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u,Tl)
            k5=h*self._model_motor(x_sol[i-1]+ (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u,Tl)
            k6=h*self._model_motor(x_sol[i-1]+ (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u,Tl)
            k7=h*self._model_motor(x_sol[i-1]+ (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6,u,Tl)
            ##Vector solución para el paso i
            x_sol.append(x_sol[i-1]+ (35/384)*k1 + (500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 +(11/84)*k6)

            ### Condiciones para el caso especial
            if x_sol[i][0]>3.9 or x_sol[i][0]< -3.9 or np.isnan(x_sol[i][0]) or i>120000:
                #Mas de 120 segundos para alcanzar estado estable es una respuesta muy lenta
                terminated=False
                truncated=True
                cespecial=True
                self._ep=self.error_porcentual(I_reference, x_sol[i][0])
                reward= -1
                print('Episodio truncado', x_sol[i][0], 'ep', self._ep, 'iteraciones', i)
                break
            #Vectores para graficar
            xdata.append(i) ## Vector de tiempo
            ydata.append(x_sol[i][0]) ##Vector solución
            if t%1000==0:
                plt.plot(xdata,ydata,color = "green")
                plt.pause(0.001)

             #Condiciones para evaluar estado estacionario I
            if t==6000:
               data1=ydata[3500:] #Analisamos del segundo 2 al 4
               ds1=np.std(data1) # Desviación standard
               #Empiricamente si ds1 es menor a 0.013, consideramos E. Estacionario.
               media1=np.mean(data1)
               if ds1<0.013 and abs(media1)>0.03:
                   self._ep=self.error_porcentual(I_reference, x_sol[i][0])
                   reward1= 1
                   self._vector_solucion = x_sol[i] #Actualizamos el vector de estados
                   time.sleep(3)
                   print("Tenemos estado estable 2s a 4s", media1)
                   break
                       
            ##Condiciones para evaluar estado estacionario II
            elif t==100000:
                data2=ydata[95000:] #Analisamos del segundo 4 al 6
                ds2=np.std(data2) #Desviación standard
                media2=np.mean(data2)
                if ds2<0.013:
                   self._ep=self.error_porcentual(I_reference, x_sol[i][0])
                   reward1= 0
                   self._vector_solucion = x_sol[i] #Actualizamos el vector de estados
                   print(data2[99995:])
                   print("Tenemos estado estable 4s a 6s")
                   print('media1 ',media1,'media2 ',media2)
                   time.sleep(3)
                   break
        plt.close()
        #Evaluamos si se alcanza el umbral del error
        if self._ep <= error_umbral and cespecial==False:
            terminated=True
            truncated=False
            reward= reward1+1
            print('Recompensa edo. Terminal', reward)
        elif self._ep > error_umbral and cespecial==False:
            terminated=False
            truncated=False
            reward= 1

        return self._vector_solucion, reward, terminated, truncated, self._ep,self._ganancias

        