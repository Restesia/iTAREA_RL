import matplotlib.pyplot as plt
#from Run import run as run
from Environment import environment as Env
from Environment.data import Tasks, Nodes
from Run import run
import pandas as pd
import sys
import time 

environment = Env.SPP_Env(T=Tasks, N=Nodes)

runner = run.runner(environment, modelo=sys.argv[1], ruta=sys.argv[2])

time_start = time.time()

reward, mat_result = runner.inference()

runner.print_asignacion(mat_result)

time_end = time.time()

timeT = time_end - time_start

print(str(timeT) + " segundos")