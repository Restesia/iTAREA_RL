import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np
import contextlib
from io import StringIO
from tensorflow.keras.models import load_model


class SIM_model_SQN():

    def __init__(self, ruta = ""):
        
        if ruta == "":
            #Declaracion de la arquitectura
            print("modelo creado de 0")

            #Entrada de la red
            input_layer = layers.Input(shape=(15,))

            #Capas ocultas
            hidden_layer1 = layers.Dense(512, activation='relu')(input_layer)
            hidden_layer2 = layers.Dense(units=512, activation="relu")(hidden_layer1)
            
            #separacion de las salidas
            output_layer1 = layers.Dense(1)(hidden_layer2) 
            output_layer2 = layers.Dense(1)(hidden_layer2)

            #salida final de la red
            mean_output_layer2 = tf.reduce_mean(output_layer2, axis=1, keepdims=True)   
            final_output = layers.Add()([output_layer1, layers.Subtract()([output_layer2, mean_output_layer2])])

            # Creacion el modelo
            self.model = models.Model(inputs=input_layer, outputs=final_output)

            #Compilación del modelo
            self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        else:
            #carga un modelo ya entrenado
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

        _target = reward + gamma * q_value
        _input = self._get_input(state, Nodes[action,:])

        self.model.fit(_input, np.array([[_target]]), verbose=0)

    def evaluate(self, _input):
        """
        Evaluate realiza una inferencia sobre la red para predecir el q_valor de asociar una tarea a un nodo dado.

        _input -> ndarray bidimensional de una sola fila y 15 columnas, se tratan de los valores que describen a la tarea y al nodo concatenados, en ese orden. 
        """
#        print(_input)
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

        Q_values = np.zeros(len(nodes), dtype=float)

        #print(mask)

        task_mat = np.tile(task, (len(nodes),1))
        _input = np.concatenate((task_mat, nodes), axis=1)
        _input = self._normalize(_input)
        Q_values = self.evaluate(_input)

        for x in range(len(nodes)):
            if not mask[x]:
                Q_values[x] = np.array([-np.finfo(np.float32).max])

#        print(Q_values)

        return np.argmax(Q_values), np.max(Q_values)

    def _get_input(self, task, node):

        """
        concatena la tarea y el nodo para prepararlo para la red neuronal
        """

        _input = np.concatenate((task, node))

        return np.array([self.normalize(_input)])
    
    def _normalize(self, _input):

        """
        Normaliza el array de entrada para prepararlo para la red neuronal
        """

        # Aquí tendremos que almacenar el máximo de cada valor de entrada a la red
        # [0] Task_CPUt Task[0]             
        divisores = [1000000.0, 
        # [1] Task_RAM Task [1]
        12000.0, 
        # [2] user Task[2]
        10.0, 
        # [3] Minimal transmission Task[3]
        1.0, 
        # [4] Disk Required Task[9]
        100.0,
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
        divisores = np.tile(divisores, (len(_input),1))
#        print(_input)

        return np.divide(_input,divisores)

    def normalize(self, _input):

        """
        Normaliza el array de entrada para prepararlo para la red neuronal
        """

        # Aquí tendremos que almacenar el máximo de cada valor de entrada a la red
        # [0] Task_CPUt Task[0]             
        divisores = [1000000.0, 
        # [1] Task_RAM Task [1]
        12000.0, 
        # [2] user Task[2]
        10.0, 
        # [3] Minimal transmission Task[3]
        1.0, 
        # [4] Disk Required Task[9]
        100.0,
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

        #print(_input)

        return np.divide(_input,divisores)




