import numpy as np

a = [[1,2,3],[1,2,3]]
b = [[4,5,6],[4,5,6]]

c = [[7,8,9],[7,8,9]]
d = [[0,0,0],[0,0,0]]

a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)

# m = [a,b]
# n = [c,d]

# m = np.array(m)
# n = np.array(n)
# print(m.shape, n.shape)
# print(m)
# print(n)

# mn = [m, n]

# e = []
# for i in range(len(mn)):
#     data = mn[i]
#     print(data.shape)

e = []

all = [a,b,c,d]
for data in all:
    # print(data.shape)
    data = data.reshape(1,-1)
    # print(data)
    e.extend(data)
e = np.array(e)
e = e.reshape(-1, 3)
print(e)
