import numpy as np            ## Importamos librerias
import pygame
import gymnasium as gym
from gymnasium import spaces
###----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
###----------------------------------------------

class GridControladorEnv(gym.Env): ##Heredamos de la clase gym
    ## Def. los modos de renderizado # El valor de render_mode se asigna en el script de entrenamiento.
    metadata= {"render_modes": ["human", "rgb_array"], "render_fps":4}

###------------>>>> Metodos init <<<<-----------------
    def __init__(self, render_mode= None, size=10):
        self.size= size #Tamaño del grid
        self.window_size=512 # Tamaño de la ventana
        
        #Def. observaciones como diccionario con la ubicación del agente.
        self.observation_space= spaces.Dict(
            {
                "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
            }
        )

        #Def. 4 acciones, "right","left", "up","down" para moverse en el plano kp-ki.
        self.action_space=spaces.Discrete(4)

        """ El siguiente diccionario mapea las acciones de'self.action_space' en la dirección
        en la que debe moverse el agente si determinada acción es tomada. I.e. 0 corresponde a
        moverse a la derecha (right), 1 moverse hacia arriba (up), etc. """

        self._action_to_direction = {
            0: np.array([1, 0]),    # Derecha
            1: np.array([0, 1]),    # Arriba
            2: np.array([-1, 0]),   # Izquierda
            3: np.array([0,-1]),    # Abajo
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode= render_mode ## El uso de assert nos permite realizar comprobaciones. 
        ## Si la expresión es False, se lanzara una excepción,  AssertionError

        """Si se usa el modo de renderizado humano, `self.window` será una referencia 
        a la ventana que dibujamos. `self.clock` será un reloj que se utiliza para
        asegurar que el entorno se renderiza a la velocidad de fotogramas correcta
        en el modo humano. Permanecerán `None` hasta que se utilice el modo humano 
        por primera vez."""

        self.window=None
        self.clock= None

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
        print("Reset. Ubicación inicial del agente",self._agent_location, type(self._agent_location))
        ganancias=self._agent_location ## Asig. ganancias controlador PI
        error_umbral=0.5 ## Def. un umbral de error para el reset.

###->>>>Insertamos el loop ctrl-motor para obtener ganancias y e.porcentual iniciales-aleatorios <<---

        ## Def. Tiempo, ciclos, paso, etc.
        t_simulation=5.000      # Tiempo de simulación en segundos.  
        Step_size=0.001         # Paso de tiempo.
        dt=h=Step_size          # Def. identificadores para el paso de tiempo.
        t_sim=np.arange(0, t_simulation, dt)  # Vector tiempo simulación.
        N=len(t_sim)            # Número de pasos

        ## En def. no config. figuras, por que no graficamos.

        ## Condiciones iniciales para las ec.motor.
        In_current= 0          # Define Initial current 
        In_position= 0         # Define Initial angular position 
        In_velocity= 0         # Define Initial angular velocity
        x0=np.array([ In_current, In_position, In_velocity ])  

        Tl=np.zeros_like(t_sim) # Entradas externas

        ### Creación vector solución y asignación de condiciones iniciales.
        x_sol=np.zeros((len(t_sim), len(x0)))
        x_sol[0]=x0

        ## Def. I. referencia
        Corriente_referencia=0.2
        I_reference= Corriente_referencia*np.ones(len(t_sim)) ##Vector  I. referencia.

        ### Def. Cond. Inical. para la integral del error en ctrl. PI
        E_integral=0

        ## Def. bandera para salir del ciclo cuando la corriente estimada sea mayor a 3.9A o "nan"
        self._caso_especial=False

        
        for i in range (1,N):
            I_error=  I_reference[i-1]- x_sol[i-1,0]              # Calculamos error seguimiento
            E_integral=E_integral+ (I_error*dt)
            u=self._PI_control(ganancias,I_error, E_integral)     # Calculamos señal de control

            ## Calculamos los terminos de RK-DP para el paso de tiempo actual
            k1=h*self._model_motor(x_sol[i-1], u, Tl[i-1])
            k2=h*self._model_motor(x_sol[i-1]+ (k1/5), u,Tl[i-1])
            k3=h*self._model_motor(x_sol[i-1]+ (3/40)*k1 + (9/40)*k2, u,Tl[i-1])
            k4=h*self._model_motor(x_sol[i-1]+ (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u,Tl[i-1])
            k5=h*self._model_motor(x_sol[i-1]+ (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u,Tl[i-1])
            k6=h*self._model_motor(x_sol[i-1]+ (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u,Tl[i-1])
            k7=h*self._model_motor(x_sol[i-1]+ (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6,u,Tl[i-1])

            ## Con los terminos de RK-DP calculamos respuest del sistema en el paso de tiempo actual
            x_sol[i]=x_sol[i-1]+ (35/384)*k1 + (500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 +(11/84)*k6

            ## Si la corriente estimada es mayor que 3.9A o "nan", salimos del bucle for.
            if x_sol[i,0]>3.9 or np.isnan(x_sol[i,0]):
                self._caso_especial=True
                break

        
            # Def.terminado, truncado y error porcentual para el caso especial.
        if self._caso_especial==True:
            terminated=False
            truncated= True
            self._agent_error_porcentual = 100 #Definimos un error al 100%.
            print("Reset. Iteración truncada debido a valores fuera de rango")

        else:
            ## Calc. error porcentual y lo asignamos a la variable self._error_porcentual
            ep=np.mean(abs((I_reference-x_sol[:,0])/I_reference)*100)
            self._agent_error_porcentual = ep
            
            ## Condiciones para estado terminada y truncado.
            if self._agent_error_porcentual <= error_umbral: 
                terminated=True
                truncated=False
                print("Reset. Iteración completada")
            else:
                terminated=False
                truncated= True
                print("Reset. Iteración truncada")

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        ## Retornamos posición del agente y error porcentual.
        return self._agent_location, self._agent_error_porcentual
        #return observation, info

    
    
####--------->>>> Método STEP <<<<<-----------------------------------------------------------------------
    def step(self, action): # Recibe la acción (a), definida Policy.
        # direction mapea (a) (elemento de {0,1,2,3}) a la dirección que dicta Policy.
        direction=self._action_to_direction[action] 
        self._agent_location = np.clip(             # Clip asegur aque no salgamos del grid.
            self._agent_location + direction, 0, self.size -1
        )
        #Modificamos el umbral a 40% para forzar el estado terminated.
        #error_umbral=0.5 ## Def. un umbral de error para STEP.
        error_umbral=40 ##<<<----Dato modificado

        ## Conf. figura para graficars
        fig, ax = plt.subplots()
        fig.suptitle('Estimated Current in the motor')
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 0.22)
        ax.set_ylabel('Currnet (A)')
        ax.set_xlabel('Time (ms)')
        xdata, ydata = [], [] # Lista que almacena los valores a graficar.

        ### Tiempo, ciclos, paso, etc.
        t_simulation=5.000                      # Tiempo simulación en segundos
        Step_size=0.001                         # Tamaño paso (s)
        dt=h=Step_size                          # Identificadores para el paso.
        t_sim=np.arange(0, t_simulation, dt)    # Vector tiempo simulación
        N=len(t_sim)                            # Number of steps  

        ### Condiciones iniciales para las ec.motor.
        In_current= 0          # Define Initial current 
        In_position= 0         # Define Initial angular position 
        In_velocity= 0         # Define Initial angular velocity
        x0=np.array([ In_current, In_position, In_velocity ])  

            ### Def. entradad externas
        Tl=np.zeros_like(t_sim)

            ### Vector solución y condiciones iniciales.
        x_sol=np.zeros((len(t_sim), len(x0)))
        x_sol[0]=x0

        Corriente_referencia=0.2  ## Def. I. referencia
        I_reference= Corriente_referencia*np.ones(len(t_sim)) ##Vector  I. referencia.
    

    # ---------------------------Definimos ganancias del controlador PI-----------------------------------
        ganancias=self._agent_location
        print("ganancias", ganancias)
    #---------------------------------------------------------------------------------------------------
        ### Def. condición inicial para la integral del error controlador PI
        E_integral= 0 

        ## Def. bandera para salir del ciclo cuando la corriente estimada sea mayor a 3.9A o "nan"
        self._caso_especial=False

        #self._agent_error_porcentual= 0
        
        for i in range (1,N):
            I_error=  I_reference[i-1]- x_sol[i-1,0]            # Error 
            E_integral=E_integral+ (I_error*dt)
            u=self._PI_control(ganancias,I_error, E_integral)    # Control signal

                #Calculamos los terminos de RK-DP para el paso de tiempo actual
            k1=h*self._model_motor(x_sol[i-1], u, Tl[i-1])
            k2=h*self._model_motor(x_sol[i-1]+ (k1/5), u,Tl[i-1])
            k3=h*self._model_motor(x_sol[i-1]+ (3/40)*k1 + (9/40)*k2, u,Tl[i-1])
            k4=h*self._model_motor(x_sol[i-1]+ (44/45)*k1 - (56/15)*k2 + (32/9)*k3, u,Tl[i-1])
            k5=h*self._model_motor(x_sol[i-1]+ (19372/6561)*k1 - (25360/2187)*k2 +(64448/6561)*k3 - (212/729)*k4, u,Tl[i-1])
            k6=h*self._model_motor(x_sol[i-1]+ (9017/3168)*k1 -(355/33)*k2 +(46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5, u,Tl[i-1])
            k7=h*self._model_motor(x_sol[i-1]+ (35/384)*k1 +(500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 + (11/84)*k6,u,Tl[i-1])
                
            ## Con los terminos de RK-DP calculamos respuest del sistema en el paso de tiempo actual
            x_sol[i]=x_sol[i-1]+ (35/384)*k1 + (500/1113)*k3 +(125/192)*k4 -(2187/6784)*k5 +(11/84)*k6

            ## Si la corriente estimada es mayor que 3.9A o "nan", salimos del bucle for.
            if x_sol[i,0]>3.9 or np.isnan(x_sol[i,0]):
                self._caso_especial=True 
                plt.close()
                break
            else:
                # Si el caso especial no se cumple, agregamos los valores a las listas y graficamos.
                xdata.append(i)
                ydata.append(x_sol[i,0])
                if i%70==0:
                    plt.plot(xdata,ydata,color = "green")
                    plt.pause(0.001)

        ## Def.terminado, truncado y error porcentual para el caso especial.
        if self._caso_especial==True:
            terminated=False
            truncated= True
            self._agent_error_porcentual = 100 #Definimos un error al 100%.
            print("Step.Iteración truncada debido a valores fuera de rango")

        else: # Si el caso especial no se cumple, calculamos el error porcentual.
            ep=np.mean(abs((I_reference-x_sol[:,0])/I_reference)*100)
            self._agent_error_porcentual = ep

            ## Condiciones para estado terminado y truncado.
            if self._agent_error_porcentual <= error_umbral:
                terminated=True
                truncated=False
                plt.close()
                print("Step. Iteración completada")
            else:
                terminated=False
                truncated= True
                plt.close() #Linea nueva
                #print("Step. Iteración truncada debido a error porcentual mayor a 0.5%")
                print(self._agent_error_porcentual)
            


        ## Definimos la recompensa en base al estado lógico de terminated.
        reward= 1 if terminated else 0 # Recompensa binaria

        ## Establecemos la observación (valores de kp y ki).
        observation=self._get_obs()

        ## Establecemos la información (error porcentual del agente).
        info=self._get_info()

        if self.render_mode=="human":
            self._render_frame()

        ## Retornamos observación, recompensa, estado terminado, truncado e información.
        #return observation, reward, terminated, truncated, info
        return self._agent_location, reward,terminated, truncated, self._agent_error_porcentual
    
    
####----->>>> Método RENDER <<<<----------------------------------------------------------------------
### Utilizamos casi la misma configuración que el caso del gridworld.
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size)) # Crea una superficie de dibujo
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  ## El tamaño de un solo cuadro de la cuadrícula en pixeles.


        ## No se dibuja objetivo, por que no lo necesitamos.
        # Ahora se dibuja el agente, originalmente era un circulo, pero se cambia por una elipse
        
        pygame.draw.ellipse(            # pygame.draw.ellipse(surface, color, rect)
            canvas,      #surface
            (0,255,255), #color
            pygame.Rect( #rect
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
        )

        #Se  agregan las lineas que forman la cuadricula.
        for x in range(self.size +1):
            ## pygame.draw.line(surface, color, start_pos, end_pos, width(optional))
            pygame.draw.line(               # linea 1
                canvas,                     # surface
                0,                          # color
                (0, pix_square_size * x),   # start_pos
                (self.window_size, pix_square_size * x), #end_pos
                width = 3, #width
            )

            pygame.draw.line(                            # linea 2
                canvas,                                  # surface
                0,                                       # color
                (pix_square_size * x, 0),                # start_pos
                (pix_square_size * x, self.window_size), # end_pos
                width = 3, #width
            )

        if self.render_mode == "human":
        # Las siguiente linea copia nuestros dibujos de 'canvas' a la ventana visible.
            self.window.blit(canvas, canvas.get_rect()) # blit(source, dest, area=None, special_flags=0) 
            # Copia una superficie a otra, con la posibilidad de escalarla, rotarla y hacer un recorte.
            pygame.event.pump() # Procesa los eventos de la cola de eventos de pygame.
            pygame.display.update() ## Permite actualizar solo una parte de la pantalla, en lugar de toda 
                                    ## el área. Si no se pasa ningún argumento, actualiza toda el área de 
                                    # la superficie como pygame.display.flip ().

        #Se agrega un retardo para mantener estable la taza de fotogramas.
            self.clock.tick(self.metadata["render_fps"])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                # pygame.surfarray.pixels3d(surface) Copia los pixeles de la superficie a un array 3D.
            )



        
