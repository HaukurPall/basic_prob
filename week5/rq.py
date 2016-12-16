import numpy as np

mat1 = np.array([10, 6, 2])
mat2 = np.array([[7, 4, 3],
                 [10, 3, 3],
                 [5, 4, 8]])
mat3 = np.array([[9],
                 [1],
                 [2]])
amswer =(np.dot(mat1, mat2))
print("{}".format(np.dot(amswer, mat3)))


import numpy as np

x = np.array([0, 1, 0])
k3 = np.array([[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0],
                 [0, 1, 1],
                 [1, 0, 0],
                 [1, 0, 1],
                 [1, 1, 0],
                 [1, 1, 1],
                 ])
enc = np.dot(k3, x)%2
print(enc)
x2 = np.array([1, 1, 0, 1])
k4 = np.array([[0, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 1, 1],
                 [0, 1, 0, 0],
                 [0, 1, 0, 1],
                 [0, 1, 1, 0],
                 [0, 1, 1, 1],
                 [1, 0, 0, 0],
                 [1, 0, 0, 1],
                 [1, 0, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 1],
                 [1, 1, 1, 0],
                 [1, 1, 1, 1],
                 ])
enc = np.dot(k4, x2)%2

print(enc)
print("0101101010100101")
