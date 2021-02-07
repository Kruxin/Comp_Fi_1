import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle


def buildTree(S, vol, T, N) -> np.array:
    """
    Creates a binomial tree with stock prices.
    :param S: Initial stockprice at t=0
    :param vol: Volatility 
    :param T: Timescale. Lower value means higher precision
    :param N: Number of steps/nodes in the tree
    :return: Returns a lower triangular matrix containing the tree
    """

    dt = T / N
    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = np.exp(-vol * np.sqrt(dt))

    for i in np.arange(N + 1):
        for j in np.arange(i + 1):
            matrix[i, j] = S * u ** j * d ** (i - j)

    return matrix


def valueOptionMatrix(tree, T, N, r, K, vol) -> None:
    """
    Calculates the derivative value with backward induction, starting at the last node
    :param tree: Binomial tree with N nodes
    :param T: Timescale. Lower value means higher precision
    :param N: Number of steps/nodes in the tree
    :param r: Interest rate
    :param K: Strike price
    :param vol: Volatility
    :return: None
    """

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


def blackScholesExp(t, T, S_t, K, vol, r) -> tuple:
    """
    Calculates the derivative value at t=0 using Black Scholes expectation value
    :param t: time t = initial time
    :param T: Timescale. Lower value means higher precision
    :param S_t: Stockprice at time = t
    :param K: Strike price
    :param vol: Volatility
    :param r: Interest rate 
    :return: Tuple (Derivative value, d2)
    """
    d1 = (np.log(S_t / K) + (r + 0.5 * vol ** 2) * (T - t)) / (vol * np.sqrt(T - t))
    d2 = d1 - vol * np.sqrt(T - t)

    return (S_t * norm.cdf(d1) - np.exp(-r * (T - t)) * K * norm.cdf(d2), d2)


def converge(max_steps, S, T, K, r, vol) -> list:
    """
    Simulates the difference in derivative value between Black scholes and binomial tree as a function of N
    :param max_steps: Maximum number of steps to check for difference
    :param S: Stock value at t=0
    :param T: Timescale. Lower value means higher precision
    :param K: Strike price
    :param r: Interest rate 
    :param vol: Volatility
    :return: List of max_steps differences in derivative value at t=0
    """

    difference = []
    reference = blackScholesExp(0, T, S, K, vol, r)[0]

    for N in tqdm(np.arange(1, max_steps + 1, 1)):

        tree = buildTree(S, vol, T, N)
        valueOptionMatrix(tree, T, N, r, K, vol)
        difference.append(abs(tree[0, 0] - reference))

    return difference


def hedge_ratio(max_steps, N, S, T, K, r) -> list:
    """
    Simulates the difference in hedge ratios between black scholes and binomial tree as a function of volatility
    :param max_steps: Maximum number of steps to check for difference
    :param N: Constant number of nodes to check as function of volatility
    :param S: Stock price at t=0
    :param T: Timescale. Lower value means higher precision
    :param K: Strike price
    :param r: Interest rate 
    :return: List of max_steps differences in hedge ratio
    """

    difference = []

    for vol in tqdm(np.arange(0, 1, 1 / max_steps)[1:]):

        tree = buildTree(S, vol, T, N)
        u = tree[1, 1]
        d = tree[1, 0]
        valueOptionMatrix(tree, T, N, r, K, vol)
        fu = tree[1, 1]
        fd = tree[1, 0]

        d2 = blackScholesExp(0, T, S, K, vol, r)[1]

        hedge_black = norm.cdf(d2)
        hedge_tree = (fu - fd) / (u - d)

        difference.append(abs(hedge_black - hedge_tree))

    return difference


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    max_steps = 100
    T = 1
    S = 100
    K = 99
    vol = 0.2
    r = 0.06

    error = converge(max_steps, S, T, K, r, vol)

    axs[0].plot(error, "r-", linewidth=0.4)
    axs[0].plot(
        max_steps - 1, error[-1], "bo", label=f"Convergence = {round(error[-1], 5)}"
    )

    axs[0].set_title("Option value")
    axs[0].set_xlabel("N (steps)")
    axs[0].set_ylabel("Error")
    axs[0].legend()

    # Hedge parameter at t=0
    N = 100
    error_hedge = hedge_ratio(max_steps, N, S, T, K, r)

    axs[1].set_title("Hedge ratio")
    axs[1].plot(np.arange(0, 1 - 1 / 100, 1 / 100), error_hedge)
    axs[1].set_xlabel("Volatility")
    axs[1].set_ylabel("Error")

    plt.show()

