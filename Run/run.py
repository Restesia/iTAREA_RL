from Environment import environment as Env
from Model import model 
import numpy as np

class runner():

    def __init__(self, environment):
        self.environment = environment
        self.model = model.SPP_model()

    def train(self, long = 0):

        Reward_over_the_training = []
        steps = []

        # Estos parámetros deberían pasarse como parte del __Init__()
        gamma = 0.9  # Factor de descuento
        epsilon = 0.1  # Probabilidad de exploración vs. explotación

        for episode in range(long):

            # Episode using act and observe

            state, mask = self.environment.reset()

            state = list(state.values())
            terminal = False

            sum_rewards = 0.0
            num_updates = 0

            while not terminal:

                if np.random.rand() < epsilon:
#                    print("he sido aventurero")
                    action = np.random.randint(len(self.environment.get_nodes()))
                else:
#                    print("he sido avaricioso")
                    action, q_value = self.model.act(np.array(state), self.environment.get_nodes(), mask)
#                    print("la acción elegida es " + str(action))
                next_state, terminal, reward, next_mask = self.environment.execute(action=action)
#                print(terminal)

                next_state = list(next_state.values())

                action, q_value = self.model.act(np.array(next_state), self.environment.get_nodes(), next_mask)

                #Aplica la formula del Q-Learning
                target = reward + gamma * q_value

                #En nuestro caso no necesitamos los Q values de los otros estados, porque nuestra red toma solo uno
                #_, current_Q = self.model.act(np.array([state.values()]), self.environment.get_nodes())

                _input = self.model._get_input(np.array(state), self.environment.get_nodes()[action])

                # El código de ejemplo guarda aquí los Q valores en una tabla, debería buscar para qué sirve esa tabla, porque 
                # puede conincidir con la memoria que utiliza DQN. De todas formas es poco probable que esto sea util en mi caso

                self.model.train(_input, np.array([[target]]))

                state = next_state

                sum_rewards += reward

            Reward_over_the_training.append(sum_rewards)
            steps.append(episode)
            print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

        return Reward_over_the_training, steps


    def inference(self):
        """
        Este método recibirá por entrada un environment y calculará la asignación de nodos a tareas
        """