from Model import model, model_DDQN, model_SQN_SIM 
from Environment import environment as Env
from Run import run
import numpy as np
import pandas as pd
import copy
import sys




def _random_data(n = 4, t = 7):

    Nodes = {}

    Tasks ={}

    for x in range(n):
        Nodes[x]["cpu"] = np.random.randint(10000
                                            20000000)
        Nodes[x]["bwup"] = np.random.randint(100000000,3000000000)
        Nodes[x]["pwup"] = np.random.rand()
        Nodes[x]["maxenergy"] = np.random.randint(50,1200)
        Nodes[x]["ram"] = np.random.randint(1000,12000)
        Nodes[x]["importance"] = np.random.rand()
        Nodes[x]["pwdown"] = np.random.rand()
        Nodes[x]["bwdown"] = 150000000
        Nodes[x]["sensingunits"] = [""]
        Nodes[x]["peripherials"] = [""]
        Nodes[x]["cores"] = np.random.randint(1,10)
        Nodes[x]["percnormal"] = np.random.rand() * 100

    for y in range(t):
        Tasks[y]["Task_CPUt"] = np.random.randint(1000,300000)
        Tasks[y]["Task_RAM"] = np.random.randint(75,1000)
        Tasks[y]["user"] = 0
        Tasks[y]["MinimTrans"] = 0
        Tasks[y]["DReq"] = 100
        Tasks[y]["user"] = [""]
        Tasks[y]["user"] = [""]

    return Nodes, Tasks
