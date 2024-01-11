import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import contextlib
from io import StringIO
from tensorflow.keras.models import load_model


class SPP_model_DDQN():

    def __init__(self, ruta = "", num_nodos = 1):
        
        if ruta == "":
            #Declaración de la arquitectura
            print("modelo creado de 0")
            #Entrada de la red
            print(num_nodos*10 + 5)
            input_layer = layers.Input(shape=(num_nodos*10 + 5,))

            #Capas ocultas
            hidden_layer1 = layers.Dense(512, activation='relu')(input_layer)
            hidden_layer2 = layers.Dense(units=512, activation="relu")(hidden_layer1)
            
            #salidas separadas
            output_layer1 = layers.Dense(1)(hidden_layer2) 
            output_layer2 = layers.Dense(num_nodos)(hidden_layer2)

            #salida final de la red
            mean_output_layer2 = tf.reduce_mean(output_layer2, axis=1, keepdims=True)   
            final_output = layers.Add()([output_layer1, layers.Subtract()([output_layer2, mean_output_layer2])])

            # Crear el modelo
            self.model = models.Model(inputs=input_layer, outputs=final_output)

            #Compilación del modelo
            self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            
        else:
            print("Modelo cargado")
            self.model = load_model(ruta)

    def train(self, action, q_value, gamma, reward, state, Nodes):
        """
        Este método lleva acabo la actualizacion de los pesos para una accion tomada.

        action -> Accion tomada, entero
         
        q_value -> Valor estimado por la red de la accion tomada, numero real float
         
        gamma -> Tasa de descuento del sistema de aprendizaje por refuerzo, numero real float

        reward -> Recompensa asignada por el entorno a la accion realizada, numero real float

        state -> Estado del entorno en el que se realiza la accion, ndarray undimensional de 5 elementos dtype = float

        Nodes -> Lista de nodos, ndarray bidimensional de n * 10 dtype = float
        """

        _target = reward + gamma * q_value[0,action]
        q_value[0,action] = _target

        _input = self._get_input(state, Nodes)
        # Entrenamiento del modelo
        self.model.fit(_input, q_value, verbose=0)

    def evaluate(self, _input):
        """
        Evaluate realiza una inferencia sobre la red para predecir el q_valor de asociar una tarea a un nodo dado.

        _input -> ndarray bidimensional de una sola fila y 15 columnas, se tratan de los valores que describen a la tarea y al nodo concatenados, en ese orden. 
        """
        #print(_input)
        fake_stdout = StringIO()
        with contextlib.redirect_stdout(fake_stdout):
            qvalue = self.model.predict(_input)
        
        return qvalue

    def act(self, task, nodes, mask):
        """
        Utilizando el modelo este metodo determina que accion debe tomar para un estado dado. Ademas calcual la estimacion 
        del Q_value realizando una inferencia sobre la red

        task ->  Tarea que compone el estado ndarray unidimensiona de 5 elementos dtype = float

        nodes -> matriz de nodos, se realizará una inferencia en la red por cada uno de ellos. ndarray bidimensional de n * 10 elementos dtype = float

        mask -> Mascara de accion que limita las acciones elegibles por el modelo, array de booleanos de n elementos

        Este metodo devuelve la accion a realizar, un entero, y el valor de la accion en cuestion. La accion elegida es aquella que mas valor
        posee.
        """

        _input = self._get_input(task, nodes)

        Q_values = self.evaluate(_input)


        for x in range(len(nodes)):
            if not mask[x]:
                Q_values[0][x] = np.array([[-np.finfo(np.float32).max]])

        return np.argmax(Q_values), Q_values

    def _get_input(self, task, nodes):
        """
        concatena la tarea y el nodo para prepararlo para la red neuronal
        """
        
        _nodes = np.zeros(len(nodes)*10,dtype=float)
        
        for x in range(len(nodes)):
            _nodes[0:10] = self._normalize_nodes(nodes[x,:])
        _task= self._normalize_task(task)

        _input = np.concatenate((_task, _nodes))

        return np.array([_input])
    
    def _normalize_nodes(self, _input):
        """
        Normaliza el array de entrada para prepararlo para la red neuronal
        """

        # Aquí tendremos que almacenar el máximo de cada valor de entrada a la red
            
        divisores = [
        # [5] Node CPU
        1000000.0, 
        # [6] Node bwup
        3000000000.0, 
        # [7] Node pwup
        0.3,
        # [8] Node maxenergy
        750.0,
        # [9] Node RAM
        12000.0,
        # [10] Node Importance 
        1.0,
        # [11] Node pwdown
        0.7,
        # [12] Node bwdown
        150000000.0,
        # [13] Node cores
        8.0, 
        # [14] Node percnormal
        100.0
        ]

        #        print(_input)

        return np.divide(_input,divisores)

    def _normalize_task(self, _input):
        """
        Normaliza el array de entrada para prepararlo para la red neuronal
        """

        # Aquí tendremos que almacenar el máximo de cada valor de entrada a la red
            
        divisores = [
        # [0] Task_CPUt Task[0] 
        1000000.0, 
        # [1] Task_RAM Task [1]
        12000.0, 
        # [2] user Task[2]
        10.0, 
        # [3] Minimal transmission Task[3]
        1.0, 
        # [4] Disk Required Task[9]
        100.0,
        ]

        #        print(_input)

        return np.divide(_input,divisores)

