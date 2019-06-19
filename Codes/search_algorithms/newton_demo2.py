def F(w):
    return w ** 2 - w + 1

def dF(w):
    return 2 * w - 1

epsilon = 0.01
w = 0
while abs(dF(w)) > epsilon:
    w = w - dF(w) / 2
print(w)

'''
eta, epsilon = 0.1, 0.01
w = 0
while abs(2 * (w - 1)) > epsilon:
    w = w - eta * 2 * (w - 1)
print(w)
'''
    
    









