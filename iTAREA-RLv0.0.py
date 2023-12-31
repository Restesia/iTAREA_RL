import matplotlib.pyplot as plt
#from Run import run as run
from Environment import environment as Env
from Environment.data import Tasks, Nodes
from Run import run

environment = Env.SPP_Env(T=Tasks, N=Nodes)

runner = run.runner(environment)

Reward_over_the_training, steps = runner.train(200)

fig, ax = plt.subplots()
ax.plot(steps, Reward_over_the_training)

ax.set(xlabel='Epochs', ylabel='Reward')
ax.grid()

fig.savefig("test.png")
plt.show()
