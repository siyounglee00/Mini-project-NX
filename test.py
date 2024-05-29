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

print("--------------------")

K_indexes = np.random.choice(range(N_I, N), K, replace=False)

print(K_indexes)

print("--------------------")

print("20 random choices of an indice between 0 and 6:")
for _ in range(20):
    print(f"> {np.random.choice(range(len([1, 4, 6, 7, 2, 4, 7])))}")

print("--------------------")

t = 5
print(np.ones_like([1, 2, 3, 5, 7, 1, 8, 2, 9, 3, 6, 1, 7, 2, 9, 3, 7]) * t)

print("--------------------")

h_inhib2 = [1, 0.5, 0, -0.5, 0.999, 5, 3, 2, 5, 10, -2, -5.5, 6.123987, float("nan")]

h = [np.random.binomial(1, h_inhib2_k) if (0 <= h_inhib2_k <= 1) and not np.isnan(h_inhib2_k) else 0 for h_inhib2_k in h_inhib2]

print(h)