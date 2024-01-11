import matplotlib.pyplot as plt
#from Run import run as run
from Environment import environment as Env
from Environment.data import Tasks, Nodes
from Run import run
import pandas as pd
import sys

environment = Env.SPP_Env(T=Tasks, N=Nodes)

runner = run.runner(environment, modelo=sys.argv[1])

Reward_over_the_training, steps = runner.train()

"""fig, ax = plt.subplots()
ax.plot(steps, Reward_over_the_training)

ax.set(xlabel='Epochs', ylabel='Reward')
ax.grid()

fig.savefig("test.png")
plt.show()"""

df = pd.DataFrame({'Recompensas': Reward_over_the_training, 'epocas': steps})

# Guardar el DataFrame en un archivo CSV
df.to_csv("Resultados-" + sys.argv[1] + ".csv", index=False)