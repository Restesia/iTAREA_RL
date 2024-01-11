from Environment import environment as Env
from Model import model, model_DDQN, model_SQN_SIM 
import numpy as np
import pandas as pd
import copy

class runner():

    def __init__(self, environment, modelo = "SQN", ruta = ""):

        # Asignacion del entorno
        self.environment = environment

        # Creacion de los modelos según el tipo de modelo con el que se va a trabajar
        self.type_model = modelo
        if modelo == "SQN":
            self.model = model.SPP_model_SQN(ruta)
        elif modelo == "DDQN":
            self.model = model_DDQN.SPP_model_DDQN(ruta, environment.Nodes.shape[0])
        elif modelo =="SQN_SIM":
            self.model = model_SQN_SIM.SIM_model_SQN(ruta)  
        
        #Carga el dataset para poder trabajar con el 
        self.df = pd.read_csv('/home/caosd/dcaosd/Pruebas/RL/dataset_shangai.csv')

        #Divide el dataset en dos conjuntos diferentes, el conjunto de entrenamiento y el conjunto de prueba
        self.test_set = self.df.sample(frac=0.1, random_state=42)
        self.train_set = self.df.drop(self.test_set.index)
        self.test_set.to_csv("datos_test"+modelo+".csv", index=False)


    def train(self):
        """
        Este metodo lleva acabo la labor de entrenamiento, recorre el dataset y pone en conjunto el entorno con el modelo
        """
        # Inicialización de variables necesarias durante el bucle de entrenamiento
        Reward_over_the_training = []
        steps = []
        updates = 0

        # Hiperparámetros de enetrenamiento
        gamma = 0.9  # Factor de descuento
        epsilon = 0.1  # Probabilidad de exploración vs. explotación

        #Bucle de entrenamiento
        for episode, row in self.train_set.iterrows():
            
            #Reinicio del entorno
            state, mask, reject = self.environment.reset()
            terminal = False
            sum_rewards = 0.0


            #Liberar recursos
            self.environment._Termina = pd.to_datetime(row["end time"])
            self.environment.release_resources(pd.to_datetime(row["start time"]))

            # Guarda el estado de los nodos para reestablecerlos en caso de rechazo
            _nodes = copy.deepcopy(self.environment.Nodes)

            #Parte del bucle de asigncaicon, fuera del while para ganar eficiencia y ahorrarnos inferencias a la red
            #Decidimos la accion a tomar (El nodo a asignar) para la primera tarea
            action, q_value = self.model.act(state, self.environment.Nodes, mask)

            #Bucle de asignacion
            while not (terminal or reject):

                #Clausula Epsilon-greedy
                if np.random.rand() < epsilon:
                    
                    #Aplica la mascara de accion para no violar las restricciones
                    indices_true = np.where(mask)[0]
                    action = np.random.choice(indices_true)

                #Llamada al metodo execute para acutalizar el entorno con la acción decidida
                next_state, terminal, reward, next_mask, reject = self.environment.execute(action=action)


                #Decidimos la accion a tomar para la siguiente tarea, necesaria para el calculo del Q_valor objetivo y para la siguiente iteracion del bucle
                next_action, q_value = self.model.act(next_state, self.environment.Nodes, next_mask)

                #Actualizamos los pesos de la red, será cada modelo el que aplique la formula del Q_learning
                self.model.train(action, q_value, gamma, reward, state, self.environment.Nodes)

                #Preparacion para la siguiente iteracion
                state = next_state
                mask = next_mask
                action = next_action
                sum_rewards += reward
            
            if reject:
                #En caso de rechazo de la peticion por falta de recuros reestablecemos los recursos 
                self.environment.Nodes = _nodes
            else:
                #En caso de actualizacion de los parámetros mostramos la información requerida para el seguimiento del entrenamiento
                Reward_over_the_training.append(sum_rewards)
                steps.append(episode)
                print('Episode {}: return={}, updates={}'.format(episode, sum_rewards, updates))
                updates = updates + 1

            # Cada 10.000 epocas guardamos el modelo por si el entrenamiento se ve interrumpido
            if (episode % 10000) == 0:
                self.model.model.save("Model_" + self.type_model + "2.h5")

        return Reward_over_the_training, steps


    def inference(self):
        """
        Este método recibirá por entrada un environment y calculará la asignación de nodos a tareas
        """
        state, mask, reject = self.environment.reset()

        asignacion = np.zeros((len(self.environment.Tasks), len(self.environment.Nodes)))

        terminal = False
        sum_rewards = 0.0

        action, q_value = self.model.act(state, self.environment.Nodes, mask)
        asignacion[self.environment._target_task,action] = 1
        while not (terminal or reject):
            state, terminal, reward, mask, reject = self.environment.execute(action=action)

            action, q_value = self.model.act(state, self.environment.Nodes, mask)
            asignacion[self.environment._target_task,action] = 1

            sum_rewards += reward

        return sum_rewards, asignacion
    
    def print_asignacion(self, asignacion):
        """
        Dada una matriz de asignación imprime a que nodo se ha asignado cada tarea
        """

        filas, columnas = asignacion.shape

        for x in range(filas):
            for y in range(columnas):
                if asignacion[x, y] == 1:
                    print("Tarea {} asignada al Nodo {}".format(x, y))