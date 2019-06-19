from PythonCodes.reinforcement_learning.dynamic_programming.value_iteration import value_iteration
from PythonCodes.reinforcement_learning.landmine_and_treasure_1d import LandmineAndTreasure

env = LandmineAndTreasure(100, 30)
pi = value_iteration(env, 0.95)
print(pi)           
        




    
    




