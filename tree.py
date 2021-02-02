import numpy as np
from scipy.stats import norm


def buildTree(S, vol, T, N):

    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            matrix[i, j] = S * u ** j * d ** (i - j)

    return matrix


def valueOptionMatrix(tree, T, r, K, vol):

    dt = T / N

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    p = (np.exp(r * dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(0, S - K)

        # print(S, K)
        # print(max(0, S - K))

    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)


sigma = 0.2
S = 100
T = 1
N = 50
K = 99
r = 0.06

tree = buildTree(S, sigma, T, N)
valueOptionMatrix(tree, T, r, K, sigma)

# Analytical with Black-scholes
def blackScholesExp(t, T, S_t, sigma):

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    return S * norm.cdf(d1) - np.exp(-r * (T - t)) * K * norm.cdf(d2)


backwards = tree[0, 0]
analytical = blackScholesExp(0, T, S, sigma)
difference = abs(backwards - analytical)
print(
    f"Difference between backwards induction: {backwards} and Black-Scholes: {analytical} is {abs(backwards - analytical)}."
)
