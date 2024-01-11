import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import copy

class SPP_Env(Env):

    metadata = {"render_modes": [None], "render_fps": 4}

    def __init__(self, T = [], N = []):

        # Inicializa las estructuras de datos que almacenan en la clase las tareas a asignar y los nodos existentes
        self.Tasks= self._format_Tasks(T)
        self.Nodes= self._format_Nodes(N)

        # Creacion de las estructuras de datos que guardan los dispositivos de los que dispone cada nodo y los requeridos por cada tarea
        self._Task_req = self._Task_req_format(T)
        self._Nodes_Resources = self._Nodes_Resources_format(N)

        # Copia de seguridad de los nodos originales en caso de que necesiten ser reestablecidos
        self.Tasks_Original = copy.deepcopy(T)
        self.Nodes_Original = copy.deepcopy(N)

        self.Full_Task = copy.deepcopy(self._format_Tasks(T))
        self.Full_Nodes = copy.deepcopy(self._format_Nodes(N))

        # Esta variable es el indice de la siguiente tarea a asignar
        self._target_task=0

        #Este diccionario almacena las tareas que estan siendo ejecutadas en cada paso de la iteracion
        self._current_Tasks = {}

        #Esta es una variable auxiliar utilizada para calcular que recursos deben ser liberados al principio de cada iteracion
        self._Termina = 0

        #Realmente lo siguiente no se utiliza, pues, pese a que la clase hereda de gymnasium.Env al final las funcionalidades de 
        #esta biblioteca no fueron utilizadas

        #Definimos el espacio de estados del problema a tratar
        self.observation_space = spaces.Dict({
            # [0] CPU cycles Task [0]
            "Task_CPUt": spaces.Box(low=100, high=np.inf, shape=(1,), dtype=np.float32),
            # [1] RAM required (mb) Task[1]
            "Task_RAM": spaces.Box(low=10, high=np.inf, shape=(1,), dtype=np.float32),
            # [2] User Task[2]
            "user": spaces.Discrete(11),
            # [3] Minimal transmission Task[3]
            "MinimTrans": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # [4] Disk Required Task[9]
            "DReq": spaces.Box(low=10, high=np.inf, shape=(1,), dtype=np.float32)
        })

        #Definimos las acciones que estarán disponibles en cada estado, por el momento ignoramos la posibilidad de las restricciones 
        self.action_space= spaces.Dict({
            "NumNodos": spaces.Discrete(len(self.Nodes)+1)
        })


    def close(self):
        super().close()

    def reset(self):

        """
        Este metodo resetea el contador _target_task y reinicia el estado para afrontar una nueva asignacion nodos-tareas.

        El metodo devuelve la primera tarea sobre la que se itera, la mascara de accion para dicha tarea y un booleano que indica si se han
        consumido todos los recursos.
        """

        self._target_task=0
        state = self.Tasks[0,:]

        mask, reject = self.get_action_mask(state, self.Nodes)

        return state, mask, reject

    # Esta funcion es equivalente a step de los environment en gym
    def execute(self, action):


        """
        Este método ejecuta un paso en el episodio actual, pasando de una tarea a la siguiente en orden de asignación.

        El método proporciona una implementación para la interzaz de entorno general proporcionada por la librería Tensorforce. 
        De acuerdo a la declaración de dicha libería este método toma por entrada la acción realizada en este paso de ejecución 
        y modifica el estado del entorno de cara a la toma de la siguiente decisión. Ademas, execute determina si el estado actual 
        es o no un estado terminal y calcula la recompensa para la acción dada en este paso de ejecución.

        El método devuelve el próximo estado, un booleano que indica si (1) el estado es terminal o (0) no lo es, la recompensa 
        para la acción realizada en el estado actual, la mascara de acciones para el siguiente estado y un booleano que indica si los
        recursos son insuficientes para realizar dicha tarea y por tanto no se puede ejecutar la tarea.
        """      

        # Comprobar si el estado en el que nos encontramos es un estado final y asigna la tarea 
        terminal = 1 if self._target_task == self.Tasks.shape[0] - 2 else 0


        # Calcula la recompensa para la accion seleccionada esta es el negativo de la energía computacional.
        reward = -(1-(self.Full_Nodes[action][9]/100))*(self.Full_Nodes[action][3]*(self.Full_Nodes[action][9]/100)*self.Full_Nodes[action][3])*(self.Tasks[self._target_task][0]/self.Full_Nodes[action][0]*(self.Tasks[self._target_task][0]/self.Full_Nodes[action][0]))

        #Para la accion seleccionada reserva los recursos pertinentes para la ejecucion de la tarea 
        self.sustract_resources(action)

        #Incrementamos el contador y preparamos la siguiente tarea
        self._target_task = self._target_task + 1
        task = self.Tasks[self._target_task,:]

        #Calculamos la mascara de acciones, esta nos indica las acciones que pueden ser tomadas en el siguiente paso de ejecucion
        mask, reject = self.get_action_mask(self.Tasks[self._target_task,:], self.Nodes)

        #node = self.Nodes[action:]


        return task, terminal, reward, mask, reject

        
    def sustract_resources(self, action):

        """
        Este método es un método auxiliar pensado para realizar la actualización de los nodos dada la acción realizada. 
        Es decir, retirar del nodo seleccionado para la tarea ejecutada los recursos que este utilizará.

        action -> Nodo que va a ejecutar la tarea actual (self._tarjet_task)
        """

        # Este método tambien debería actualizar la lista de perifericos de los que dispone un nodo, dado que dos periféricos no pueden ser usados
        # por una misma aplicación. Sin embargo, por ahora simplemente acutalizaremos los recursos de los nodos asignados de esta manera

        self.Nodes[action][0] = self.Nodes[action][0] - self.Tasks[self._target_task][0]
        self.Nodes[action][4] = self.Nodes[action][4] - self.Tasks[self._target_task][1]

        for elemento in self._Nodes_Resources[action]:
            if elemento in self._Task_req[self._target_task] and elemento !="":
                self._Task_req[self._target_task].remove(elemento)

        In_execution = {
            "Node" : action,
            "Task" : self._target_task
        }

        if self._Termina in self._current_Tasks:
            self._current_Tasks[self._Termina].append(In_execution)

        else:
            self._current_Tasks[self._Termina] = [In_execution]

    
    def release_resources(self, time):

        """
        Complementando al método sustract_resources este método libera los recursos antes reservados. Este método es llamado 
        antes de que llegue una petición para liberar los recursos de las tareas que hubieran acabado antes de su tiempo de inicio.

        Time -> Día y hora exactas del comienzo de una petición. Time es un objeto datetime de la biblioteca pandas.
        """

        _list = list(self._current_Tasks)

        for _each in _list:
            #print(_each)
            if time > _each:

                for pack in self._current_Tasks[_each]:

                    #print(pack)
                    Node = pack["Node"]
                    Task = pack["Task"]

                    self.Nodes[Node][0] = self.Nodes[Node][0] + self.Tasks[Task][0]
                    self.Nodes[Node][4] = self.Nodes[Node][4] + self.Tasks[Task][1]

                    for elemento in self._Task_req[Task]:
                        if not (elemento in self._Nodes_Resources[Node]) and elemento !="":
                            self._Nodes_Resources[Node].append(elemento)
                
                self._current_Tasks.pop(_each)
    
    def get_action_mask(self, task, Nodes):
        """
        Calcula, para una tarea dada, los nodos que son compatibles con dicha tarea en el momento en que el método es llamado.

        task -> Tarea almacenada en un ndarray flotante unidimensional de tamaño 5.

        Nodes -> Nodos contenidos en un ndarray flotante bidimensional de tamaño nx10  donde n es el número de nodos.

        Devuelve un ndarray unidimensional booleano en el que cada índice corresponde a la validez del nodo asociada a él para la tarea en cuestión.
        """
                #Complejidad n
        mask = np.zeros(len(Nodes), dtype=bool)
        reject = False

                #Complejidad lineal para n (Si ignoramos los requisitos de dispositivos será menor a 10n)
        for x in range(len(Nodes)):
            
            compatible = self._calc_cond(task, Nodes, x)
            # Aquí se calcula el booleano
            
            reject = reject or compatible
            
            mask[x] = compatible 

        return mask, not reject
    
    def _calc_cond(self, task, Nodes, x):
        """
        Método auxiliar que se utiliza para comprobar si un par Nodo Tarea son compatibles en un momento dado.

        task -> Tarea almacenada en un ndarray flotante unidimensional de tamaño 5.

        Nodes -> Nodos contenidos en un ndarray flotante bidimensional de tamaño nx10  donde n es el número de nodos.

        x -> Entero correspondiente a la acción realizada. Es necesario comprobar tambien los dispositivos de los que dispone el nodo,
        por lo que no nos valdría simplemente con pasar el nodo en lugar de la matriz de nodos.

        Devuelve un booleano.
        """
                # La complejidad de este método es constante para n y t

        cond1 = True
        for index in range(len(self._Task_req[self._target_task])):
            cond1 = cond1 and self._Task_req[self._target_task][index] in self._Nodes_Resources[x]
 
        cond2 = task[0] < Nodes[x][0] and task[1] < Nodes[x][4]


        return cond1 and cond2
    
    def _format_Nodes(Self, N):
        """
        Dota de formato a las tareas del entorno, cambiando la estructura de datos recibida, el diccionario a un ndarray de numpy.
        La matriz es más eficiente a la hora de operar, dado que es el formato que esperan como entrada las redes neuronales.
        Este método solo es llamado en la creación del entorno para la traducción, luego no se utiliza más.

        N -> Diccionario en el que se contienen los nodos (Cosultar data).
        """

        
                #Complejidad 10n
        nodos = np.zeros((len(N), 10))

                #Complejidad 10n
        for x in range(len(N)):
            nodos[x][0] = float(N[x]["cpu"])
            nodos[x][1] = float(N[x]["bwup"])
            nodos[x][2] = float(N[x]["pwup"])
            nodos[x][3] = float(N[x]["maxenergy"])
            nodos[x][4] = float(N[x]["ram"])
            nodos[x][5] = float(N[x]["importance"])
            nodos[x][6] = float(N[x]["pwdown"])
            nodos[x][7] = float(N[x]["bwdown"])
            nodos[x][8] = float(N[x]["cores"])
            nodos[x][9] = float(N[x]["percnormal"])

        return nodos
    
    def _Nodes_Resources_format(self,N):
        """
        Crea una lista de listas que contiene en cada caso las claves de recursos que posee cada nodo. Esta lista se utiliza
        en el calculo de la máscara de acción. De ella se sustraen los recursos para que estos no sean accesibles a más de una 
        tarea a la vez.

        N -> Diccionario en el que se contienen los nodos (Cosultar data).
        """
                
        _format = []

                #complejidad (3 + r1 + r2)n   donde ri es la cantidad de elementos de las listas de requisitos 
        for x in range(len(N)):
            _format.append(N[x]["sensingunits"] + N[x]["peripherials"]) 

        return _format

    def _Task_req_format(self,T):
        """
        Crea una lista de listas que contiene en cada caso las claves de requisitos demandados por cada tarea. Esta lista se utiliza
        en el calculo de la máscara de acción. 

        T -> Diccionario en el que se contienen las tareas (Cosultar data).
        """
        _format = []

                #complejidad (3 + r1 + r2)t   donde ri es la cantidad de elementos de las listas de requisitos 
        for x in range(len(T)):
            _format.append(T[x]["sensreq"] + T[x]["periphreq"])
        
        return _format

    def _format_Tasks(self, T):
        """
        Dota de formato a las tareas del entorno, cambiando la estructura de datos recibida, el diccionario a un ndarray de numpy.
        La matriz es más eficiente a la hora de operar, dado que es el formato que esperan como entrada las redes neuronales
        Este método solo es llamado en la creación del entorno para la traducción, luego no se utiliza más.

        T -> Diccionario en el que se contienen las tareas (Cosultar data).
        """

                #Complejidad 5t
        tasks = np.zeros((len(T), 5))

                #Complejidad 5t
        for x in range(len(T)):
            tasks[x][0] = float(T[x]["Task_CPUt"])
            tasks[x][1] = float(T[x]["Task_RAM"])
            tasks[x][2] = float(T[x]["user"])
            tasks[x][3] = float(T[x]["MinimTrans"])
            tasks[x][4] = float(T[x]["DReq"])

        return tasks
    

        """
    def release_all(self):
        ""
        Libera todos los recursos asignados a nuestra arquitectura de red.
        ""

        self.Tasks = self.Tasks_Original
        self.Nodes = self._format_Nodes(self.Nodes_Original)
        self._current_Tasks = []"""