import numpy as np            ## Importamos librerias
import pygame
import gymnasium as gym
from gymnasium import spaces
###----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
###----------------------------------------------

class GridControladorEnvII(gym.Env): ##Heredamos de la clase gym
    ## Def. los modos de renderizado # El valor de render_mode se asigna en el script de entrenamiento.
    #metadata= {"render_modes": ["human", "rgb_array"], "render_fps":4}

###------------>>>> Metodos init <<<<-----------------
    #def __init__(self, render_mode= None, size=10):
    def __init__(self,size=100):
        self.size= size #Tamaño del grid
        #self.window_size=512 # Tamaño de la ventana
        
        #Def. observaciones como diccionario con la ubicación del agente.
        # self.observation_space= spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
        #     }
        # )

        #Def. 4 acciones, "right","left", "up","down" para moverse en el plano kp-ki.

        self.action_space=spaces.Discrete(4)
        
        """ El siguiente diccionario mapea las acciones de'self.action_space' en la dirección
        en la que debe moverse el agente si determinada acción es tomada. I.e. 0 corresponde a
        moverse a la derecha (right), 1 moverse hacia arriba (up), etc. """

        # self._action_to_direction = {
        #     0: np.array([1, 0]),    # Derecha
        #     1: np.array([0, 1]),    # Arriba
        #     2: np.array([-1, 0]),   # Izquierda
        #     3: np.array([0,-1]),    # Abajo
        # }

        self._action_to_direction = {
            0: np.array([1, 0]),    # Derecha
            1: np.array([0, 1]),    # Arriba
            2: np.array([-1, 0]),   # Izquierda
            3: np.array([0, -1]),    # Abajo
        }

       # assert render_mode is None or render_mode in self.metadata["render_modes"]
        #self.render_mode= render_mode ## El uso de assert nos permite realizar comprobaciones. 
        ## Si la expresión es False, se lanzara una excepción,  AssertionError

        """Si se usa el modo de renderizado humano, `self.window` será una referencia 
        a la ventana que dibujamos. `self.clock` será un reloj que se utiliza para
        asegurar que el entorno se renderiza a la velocidad de fotogramas correcta
        en el modo humano. Permanecerán `None` hasta que se utilice el modo humano 
        por primera vez."""

        #self.window=None
        #self.clock= None

###------------>>>> Metodos para obtener ubicación del agente y error porcentual <<<<-----------------
    def _get_obs(self):
        """Devuelve una observación del entorno. En este caso, la localización del agente."""
        return {"agent": self._agent_location}
    
    def _get_info(self):
        """Devuelve información  sobre el error resultante, al comparar la referencia vs
        la salida obtnenida"""
        return {"error_porcentual": self._agent_error_porcentual }

##------------>>>> métodos  para calcular salidas del motor y señal de control <<<<-------------------------
    def _PI_control(self,ganancias, error, E_integral):
        Kp,Ki=ganancias
        Pout=Kp*error
        Iout=Ki*E_integral
        return Pout+Iout
    
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
    
##---------->>>> Método Reset <<<------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2,dtype=int)
        #self._agent_location=np.array([9, 5]) # Esta ubicación inicial rompia el programa
        print("Reset. Ubicación inicial ",self._agent_location)
        
      

        return self._agent_location # Mi observación es la posición del agente en el grid.
        #return observation, info Config.original del gridworld.

    
    ##---->>>Pendiente revisar y arregal el step <<----------
    def step(self,I_error, E_integral,h,x_sol, Tl, I_reference,error_umbral, action): # Conf. provisional p/desarrollo de la sulución
        direction=self._action_to_direction[action] 
        #print('agent location', self._agent_location)
        #print('step direction', direction)
        
        self._agent_location = np.clip(           
        self._agent_location + direction, 0, self.size -1
        )


        #print('step acción', direction)
        #print('step, acción- location actualizada', direction, self._agent_location)
    #-->>>En esta parte estaba el for
        GX=self._agent_location/10
        print('Gananci aplicada ',GX)
        #ganancias=self._agent_location
        ganancias=GX
        u=self._PI_control(ganancias,I_error, E_integral)    # Control signal

    #Calc. terminos de RK-DP para el paso de tiempo actual
        k1=h*self._model_motor(x_sol, u, Tl)
        k2=h*self._model_motor(x_sol + (k1/5), u, Tl)
        k3=h*self._model_motor(x_sol + (3/40)*k1 + (9/40)*k2, u, Tl)
        k4=h*self._model_motor(x_sol + (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u, Tl)
        k5=h*self._model_motor(x_sol + (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u, Tl)
        k6=h*self._model_motor(x_sol + (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u, Tl)
        k7=h*self._model_motor(x_sol + (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6, u, Tl)
                
    ## Calc. resp. del sistema para el paso de tiempo actual
    
        x_sol_next=x_sol+ (35/384)*k1 + (500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 +(11/84)*k6
        #print("vector solución un paso", x_sol_next)
                #print(x_sol_next[0])

        self._agent_error_porcentual = abs((I_reference-x_sol_next[0])/I_reference)*100

    ## Si la corriente estimada es mayor que 3.9A o "nan", salimos del bucle for.
        if x_sol_next[0]>3.9 or x_sol_next[0]< -3.9 or np.isnan(x_sol_next[0]):
            #Aqui consideramos limite superior e inferior para la corriente
            terminated=False
            truncated=True
            reward= -1
            print("Caso especial, error", self._agent_error_porcentual)
        else:
            # Aqui calculaba el error originalmente
            if self._agent_error_porcentual <= error_umbral:
                terminated=True
                truncated=False
                reward=10
                print("Step CII. Terminado. Ep debajo de umbral", self._agent_error_porcentual)

            else:
                terminated=False
                truncated= False
                reward=1
                #print("Step CIII, Ep",self._agent_error_porcentual) ##-- REVISAR función de recompensa.

    # observation =agent_location
        return self._agent_location, reward,terminated, truncated,x_sol_next, self._agent_error_porcentual