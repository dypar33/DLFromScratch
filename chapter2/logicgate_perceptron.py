import numpy as np

def AND(x : np.ndarray):
    w = np.array([1, 1])
    b = -1
    
    result = np.sum(x * w) + b

    if result <= 0:
        return 0
    else:
        return 1
    
def OR(x : np.ndarray):
    w = np.array([1, 1])
    b = -0.9
    
    result = np.sum(x * w) + b

    if result <= 0:
        return 0
    else:
        return 1
    
def NAND(x : np.ndarray):
    w = np.array([-1, -1])
    b = 2
    
    result = np.sum(x * w) + b

    if result <= 0:
        return 0
    else:
        return 1
    
def XOR(x : np.ndarray):
    t1 = NAND(x)
    t2 = OR(x)
    
    return AND(np.array([t1, t2]))

def testing(func):
    testcases = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for case in testcases:
        print("{0} {1} = {2}".format(case[0], case[1], func(case)))
    
    print()
    
testing(AND)
testing(OR)
testing(NAND)
testing(XOR)