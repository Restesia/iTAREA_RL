import matplotlib.pyplot as plt
#from Run import run as run
from Environment import environment as Env
from Environment.data import Tasks, Nodes
environment = Env.SPP_Env(T=Tasks, N=Nodes)

state = environment.reset()

print(state)

"""
Reward_over_the_training, steps = run.train(1000,environment)

fig, ax = plt.subplots()
ax.plot(steps, Reward_over_the_training)

ax.set(xlabel='Epochs', ylabel='Reward')
ax.grid()

fig.savefig("test.png")
plt.show()
"""