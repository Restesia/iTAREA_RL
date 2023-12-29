def train(long = 0, environment):

    Reward_over_the_training = []
    steps = []

    # Estos parámetros deberían pasarse como parte del __Init__()
    gamma = 0.9  # Factor de descuento
    epsilon = 0.1  # Probabilidad de exploración vs. explotación

    for episode in range(long):

    # Episode using act and observe
    state = self.environment.reset()
    terminal = False

    sum_rewards = 0.0
    num_updates = 0

    while not terminal:

        if np.random.rand() < epsilon:
            actions = np.random.randint(len(self.environment.get_nodes()))
        else:
            actions, q_value = self.model.act(np.array([state.values()]), self.environment.get_nodes())

        next_state, terminal, reward = self.environment.execute(actions=actions)

        actions, q_value = self.model.act(np.array([next_state.values()]), self.environment.get_nodes())

        #Aplica la formula del Q-Learning
                                        # Debemos devolver el _valor máximo para el siguiente estado
        target = reward + gamma * q_value

        _, current_Q = self.model.act(np.array([state.values()]), self.environment.get_nodes())

        # El código de ejemplo guarda aquí los Q valores en una tabla, debería buscar para qué sirve esa tabla, porque 
        # puede conincidir con la memoria que utiliza DQN. De todas formas es poco probable que esto sea util en mi caso

        self.model.train(np.array([state.values()]), current_Q)

        sum_rewards += reward

    Reward_over_the_training.append(sum_rewards)
    steps.append(episode)
    print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    return Reward_over_the_training, steps


def inference():
    """
    Este método recibirá por entrada un environment y calculará la asignación de nodos a tareas
    """