import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

class SPP_model():

    def __init__(self):

        #Declaración de la arquitectura
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu', input_dim=15))
        self.model.add(layers.Dense(64, activation='relu'))  
        self.model.add(layers.Dense(1, activation='linear'))  


        #Compilación del modelo
        self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def train(self, _input, _target):
        # Entrenamiento del modelo
        self.model.fit(_input, _target, verbose=0)

    def evaluate(self, _input):
#        print(_input)
        qvalue = self.model.predict(_input)
        
        return qvalue

    def act(self, task, nodes, mask):

        Q_values = []

        for x in nodes:
            if mask[x]:
                _input = self._get_input(task, nodes[x])

                Q_values.append(self.evaluate(_input))
            else:
                Q_values.append(np.array([[-np.finfo(np.float32).max]]))

#        print(Q_values)

        return np.argmax(Q_values), np.max(Q_values)

    def _get_input(self, task, node):


        raw_node = np.array(list(node.values()))

#        print(task)
        _input = np.concatenate((task, raw_node))

        return np.array([self._normalize(_input)])
    
    def _normalize(self, _input):
        
        _input = _input.astype(float)

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

#        print(_input)

        return np.divide(_input,divisores)



