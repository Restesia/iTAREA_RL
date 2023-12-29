import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

class SPP_model():

    def __init__(self):

        #Declaración de la arquitectura
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))

        #Compilación del modelo
        self.model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def train(self):
        # Entrenamiento del modelo
        self.model.train_on_batch('Datos', 'Etiquetas')

    def evaluate(self):
        qvalue = self.model.predict('imput')
        
        return qvalue

    def act(self, task, nodes):

        Q_values = []

        for x in nodes:

            _input = self._get_input(task, nodes[x])

            Q_values.append(self.evaluate(_input))

        return np.argmax(Q_values), np.max(Q_values)

    def _get_input(self, task, node):

        raw_node = np.array(list(node.values()))

        _input = np.concatenate((task, raw_node))

        return self._normalize(_input)
    
    def _normalize(self, _input):
        
        _input = _input.astype(float)

        # Aquí tendremos que almacenar el máximo de cada valor de entrada a la red
        divisores = []

        return np.divide(_input,divisores)



