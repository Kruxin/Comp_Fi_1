import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle


def buildTree(S, vol, T, N):

    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            matrix[i, j] = S * u ** j * d ** (i - j)

    return matrix


def valueOptionMatrix(tree, T, N, r, K, vol):

    dt = T / N

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    p = (np.exp(r * dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(0, S - K)

    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)


# Analytical with Black-scholes
def blackScholesExp(t, T, S_t, K, sigma, r):

    d1 = (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    return S_t * norm.cdf(d1) - np.exp(-r * (T - t)) * K * norm.cdf(d2)


# backwards = tree[0, 0]
# analytical = blackScholesExp(0, T, S, K, sigma, r)
# difference = abs(backwards - analytical)
# print(
#     f"Difference between backwards induction: {backwards} and Black-Scholes: {analytical} is {abs(backwards - analytical)}."
# )

# Convergence
def converge(max_steps, S, T, K, r, sigma):

    difference = []
    reference = blackScholesExp(0, T, S, K, sigma, r)

    for N in tqdm(np.arange(1, max_steps + 1, 1)):

        tree = buildTree(S, sigma, T, N)
        valueOptionMatrix(tree, T, N, r, K, sigma)
        difference.append(abs(tree[0, 0] - reference))

    return difference


if __name__ == "__main__":

    N = 100
    T = 1

    error = converge(N, 100, T, 99, 0.06, 0.2)
    plt.plot(error, "r-", linewidth=0.4)
    plt.plot(N - 1, error[-1], "bo", label=f"Convergence = {round(error[-1], 5)}")

    plt.xlabel("N (steps)")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

