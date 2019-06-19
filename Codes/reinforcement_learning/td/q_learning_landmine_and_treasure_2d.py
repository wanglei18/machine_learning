import numpy as np
import PythonCodes.reinforcement_learning.td.q_learning as alg
import PythonCodes.reinforcement_learning.landmine_and_treasure_2d as lat2d
    
m = 4
n = 4
env =lat2d.LandmineAndTreasure2d(m, n)
#pi = alg.q_learning(env, 100, 0.95, 0.2, 0.99, 0.1)
pi = alg.q_learning(env, 100, 0.95, 0.2, 0.1)

map = {0: "U", 1: "R", 2:"D", 3:"L"}
for i in range(m):
    print([map[pi[s]] for s in range(i*n, (i+1)*n)])



    
    




