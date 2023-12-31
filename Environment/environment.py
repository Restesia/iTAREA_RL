import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np

class SPP_Env(Env):

    metadata = {"render_modes": [None], "render_fps": 4}

    def __init__(self, T = [], N = []):
        self.Tasks=T
        self.Nodes= self._format_Nodes(N)
        self.Tasks_Original = T
        self.Nodes_Original = self.Nodes
        self._target_task=0
        self._current_Tasks = []

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

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._target_task=0
        task = self.Tasks[0]
        self.release_all()
        mask = self.get_action_mask(self.Tasks[self._target_task], self.Nodes)
        state = {
            "Task_CPUt": float(task["Task_CPUt"]),
            "Task_RAM": float(task["Task_RAM"]),
            "user": float(task["user"]),
            "MinimTrans": float(task["MinimTrans"]),
            "DReq": float(task["DReq"])
        }

        return state, mask

    # Esta funcion es equivalente a step de los environment en gym
    def execute(self, action):


        """
        Este método ejecuta un paso en el episodio actual, pasando de una tarea a la siguiente en orden de asignación.

        El método proporciona una implementación para la interzaz de entorno general proporcionada por la librería Tensorforce. 
        De acuerdo a la declaración de dicha libería este método toma por entrada la acción realizada en este paso de ejecución 
        y modifica el estado del entorno de cara a la toma de la siguiente decisión. Además, execute determina si el estado actual 
        es o no un estado terminal y calcula la recompensa para la acción dada en este paso de ejecución.

        El método devuelve el próximo estado, un booleano que indica si (1) el estado es terminal o (0) no lo es y la recompensa 
        para la acción realizada en el estado actual.
        """      


        reward = 0

        # Comprobar si el estado en el que nos encontramos es un estado final y asigna la tarea 
        terminal = 1 if self._target_task == len(self.Tasks) - 2 else 0

        # Asignar la tarea siguiente a asignar como la tarea siguiente a la esperada
        task = self.Tasks[self._target_task]

        reward = -(1-(self.Nodes[action]["percnormal"]/100))*(self.Nodes[action]["maxenergy"]*(self.Nodes[action]["percnormal"]/100)*self.Nodes[action]["maxenergy"])*(self.Tasks[self._target_task]["Task_CPUt"]/self.Nodes[action]["cpu"]*(self.Tasks[self._target_task]["Task_CPUt"]/self.Nodes[action]["cpu"]))

        self.sustract_resources(action)

        self._target_task = self._target_task + 1

        #Calculamos la mascara de acciones, esta nos indica las acciones que pueden ser tomadas en el siguiente paso de ejecucion
        mask = self.get_action_mask(self.Tasks[self._target_task], self.Nodes)

        node = self.Nodes[action]

        state = {
            "Task_CPUt": float(task["Task_CPUt"]),
            "Task_RAM": float(task["Task_RAM"]),
            "user": float(task["user"]),
            "MinimTrans": float(task["MinimTrans"]),
            "DReq": float(task["DReq"])
        }


        return state, terminal, reward, mask

        
    def sustract_resources(self, action):

        """
        Este método es un método auxiliar pensado para realizar la actualización de los nodos dada la acción realizada. 
        Es decir, retirar del nodo seleccionado para la tarea ejecutada los recursos que este utilizará.
        """

        # Este método tambien debería actualizar la lista de perifericos de los que dispone un nodo, dado que dos periféricos no pueden ser usados
        # por una misma aplicación. Sin embargo, por ahora simplemente acutalizaremos los recursos de los nodos asignados de esta manera

        self.Nodes[action]["cpu"] = self.Nodes[action]["cpu"] - self.Tasks[self._target_task]["Task_CPUt"]
        self.Nodes[action]["ram"] = self.Nodes[action]["ram"] - self.Tasks[self._target_task]["Task_RAM"]

        In_execution = {
            "Node" : action,
            "Task" : self._target_task
        }

        self._current_Tasks.append(In_execution)
    
    def release_resources(self, _current_execution):

        """
        Complementando al método sustract_resources este método libera los recursos antes reservados una vez la ejecución ha concluido
        """

        Node = _current_execution["Node"]
        Task = _current_execution["Task"]

        self.Nodes[action]["cpu"] = self.Nodes[action]["cpu"] + self.Tasks[self._target_task]["Task_CPUt"]
        self.Nodes[action]["ram"] = self.Nodes[action]["ram"] + self.Tasks[self._target_task]["Task_RAM"]

    def release_all(self):
        """
        Libera todos los recursos asignados a nuestra arquitectura de red.
        """

        self.Tasks = self.Tasks_Original
        self.Nodes = self.Nodes_Original
        self._current_Tasks = []
    
    def get_action_mask(self, task, Nodes):

        mask = []

        for x in Nodes:
            
            compatible = False
            # Aquí se calcula el booleano

            if task["Task_CPUt"] < Nodes[x]["cpu"] and task["Task_RAM"] < Nodes[x]["ram"]:
                compatible = True
            
            mask.append(compatible)
        mask = np.array(mask)


        return mask.flatten()
    
    def get_nodes(self):
        return self.Nodes
    
    def _format_Nodes(Self, N):
        
        nodos = {}

        for x in N:
            nodo = {
                "cpu" : float(N[x]["cpu"]),
                "bwup": float(N[x]["bwup"]),
                "pwup": float(N[x]["pwup"]),
                "maxenergy": float(N[x]["maxenergy"]),
                "ram": float(N[x]["ram"]),
                "importance": float(N[x]["importance"]),
                "pwdown": float(N[x]["pwdown"]),
                "bwdown": float(N[x]["bwdown"]),
                "cores": float(N[x]["cores"]),
                "percnormal": float(N[x]["percnormal"])
            }
            nodos[x] = nodo

        return nodos