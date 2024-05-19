import numpy as np

N_I = 5
N = 100
K = 3

for i in range(N):
    if i >= N_I and i < N_I + K:
        print(i)
    if i < N_I:
        print("N_I: " + str(i))

for i in range(N_I, N_I+K):
    print(i)

print(len(range(N_I, N)))
print(range(N_I, N)[-1])

print("------------")

K_indexes = np.random.choice(range(N_I, N), K, replace=False)

print(K_indexes)