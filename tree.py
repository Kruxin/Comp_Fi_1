import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pandas as pd


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message="Option not call or put"):
        self.expression = expression
        self.message = message


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


def valueOptionMatrix(tree, T, N, r, K, vol, option="call") -> None:
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

    try:
        if option == "call":
            for c in np.arange(columns):
                S = tree[rows - 1, c]
                tree[rows - 1, c] = max(0, S - K)

        elif option == "put":
            for c in np.arange(columns):
                S = tree[rows - 1, c]
                tree[rows - 1, c] = max(0, K - S)
        else:
            raise InputError(option)

    except InputError as e:
        print(f"{e.expression} is not a valid option. Try call or put.")

    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]

            value = np.exp(-r * dt) * (p * up + (1 - p) * down)
            tree[i, j] = value


def blackScholesExp(t, T, S_t, K, vol, r, option="call") -> tuple:
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

    try:

        if option == "call":
            return (S_t * norm.cdf(d1) - np.exp(-r * (T - t)) * K * norm.cdf(d2), d1)

        elif option == "put":
            return (K * np.exp(-r * (T - t)) * norm.cdf(-d2) - S_t * norm.cdf(-d1), d1)

        else:
            raise InputError(option)

    except InputError as e:
        print(f"{e.expression} not a valid option. Try EU or USA.")


def optionValue(
    max_steps, S, T, K, r, vol, derivative="call", model="binomial"
) -> float:
    """
    Simulates the derivative value for both models and options
    :param max_steps: Maximum number of steps to check for difference
    :param S: Stock value at t=0
    :param T: Timescale. Lower value means higher precision
    :param K: Strike price
    :param r: Interest rate 
    :param vol: Volatility
    :return: tuple of values
    """
    eu_value, usa_value = 0, 0
    try:
        if model == "binomial":
            tree = buildTree(S, vol, T, max_steps)
            valueOptionMatrix(tree, T, max_steps, r, K, vol, option=derivative)
            eu_value = tree[0, 0]

            tree = buildTree(S, vol, T, max_steps)
            americanOptionMatrix(tree, T, max_steps, r, K, vol, option=derivative)
            usa_value = tree[0, 0]

        elif model == "blackscholes":
            eu_value = blackScholesExp(0, T, S, K, vol, r, option=derivative)[0]

        else:
            raise InputError(model)

    except InputError as e:
        print(f"{e.expression} not a valid model. Try binomial or blackscholes.")

    return eu_value, usa_value


def converge(max_steps, S, T, K, r, vol, option="EU", derivative="call") -> list:
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
    reference = blackScholesExp(0, T, S, K, vol, r, option=derivative)[0]

    try:
        if option == "EU":
            for N in tqdm(np.arange(1, max_steps + 1, 1)):

                tree = buildTree(S, vol, T, N)
                valueOptionMatrix(tree, T, N, r, K, vol, option=derivative)
                difference.append(abs(tree[0, 0] - reference))

        elif option == "USA":
            for N in tqdm(np.arange(1, max_steps + 1, 1)):

                tree = buildTree(S, vol, T, N)
                americanOptionMatrix(tree, T, N, r, K, vol, option=derivative)
                difference.append(abs(tree[0, 0] - reference))
        else:
            raise InputError(option)

    except InputError as e:
        print(f"{e.expression} not a valid option. Try EU or USA.")

    return difference


def hedge_ratio(max_steps, N, S, T, K, r, derivative="call") -> list:
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

    for vol in tqdm(np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:]):

        tree = buildTree(S, vol, T, N)
        u = tree[1, 1]
        d = tree[1, 0]
        valueOptionMatrix(tree, T, N, r, K, vol, option=derivative)
        fu = tree[1, 1]
        fd = tree[1, 0]

        d1 = blackScholesExp(0, T, S, K, vol, r, option=derivative)[1]

        hedge_black = norm.cdf(d1)
        hedge_tree = (fu - fd) / (u - d)

        difference.append(abs(hedge_black - hedge_tree))

    return difference


def americanOptionMatrix(tree, T, N, r, K, vol, option="call") -> None:
    """
    Calculates the derivative value with backward induction, starting at the last node for american options
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

    try:
        if option == "call":
            for c in np.arange(columns):
                S = tree[rows - 1, c]
                tree[rows - 1, c] = max(0, S - K)

        elif option == "put":
            for c in np.arange(columns):
                S = tree[rows - 1, c]
                tree[rows - 1, c] = max(0, K - S)
        else:
            raise InputError(option)

    except InputError as e:
        print(f"{e.expression} is not a valid option. Try call or put.")

    for i in np.arange(rows - 1)[::-1]:

        for j in np.arange(i + 1):

            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            S = copy.deepcopy(tree[i, j])
            value = np.exp(-r * dt) * (p * up + (1 - p) * down)

            if option == "call":
                exercise = S - K
            elif option == "put":
                exercise = K - S

            if exercise > value:
                tree[i, j] = exercise
            else:
                tree[i, j] = value


def compare(max_steps, N, S, T, K, r) -> list:
    """
    Calculates the derivative value with backward induction, starting at the last node for american options
    :param max_steps: Marginalization of volatility
    :param T: Timescale. Lower value means higher precision
    :param N: Number of steps/nodes in the tree
    :param r: Interest rate
    :param K: Strike price
    :return: List of errors
    """

    difference = []

    for vol in tqdm(np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:]):

        eu_binomial = optionValue(
            N, S, T, K, r, vol, derivative=EU_derivative, model="binomial"
        )[0]
        eu_blackscholes = optionValue(
            N, S, T, K, r, vol, derivative=EU_derivative, model="blackscholes"
        )[0]

        difference.append(abs(eu_binomial - eu_blackscholes))

    return difference


def americanSimulate(max_steps, T, N, r, K) -> list:
    """
    Simulates the American call option vs the American put option
    :param T: Timescale. Lower value means higher precision
    :param N: Number of steps/nodes in the tree
    :param r: Interest rate
    :param K: Strike price
    :return: Two lists with Call values and Put values at T=0
    """

    call_values = []
    put_values = []
    eu_call = []
    eu_put = []

    for vol in tqdm(np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:]):

        tree1 = buildTree(S, vol, T, N)
        tree2 = buildTree(S, vol, T, N)
        tree3 = buildTree(S, vol, T, N)
        tree4 = buildTree(S, vol, T, N)

        americanOptionMatrix(tree1, T, N, r, K, vol, option="call")
        americanOptionMatrix(tree2, T, N, r, K, vol, option="put")
        valueOptionMatrix(tree3, T, N, r, K, vol, option="call")
        valueOptionMatrix(tree4, T, N, r, K, vol, option="put")

        call_values.append(tree1[0, 0])
        put_values.append(tree2[0, 0])
        eu_call.append(tree3[0, 0])
        eu_put.append(tree4[0, 0])

    return call_values, put_values, eu_call, eu_put


if __name__ == "__main__":

    max_steps = 50
    T = 1
    S = 100
    K = 99
    vol = 0.2
    r = 0.06
    EU_derivative = "call"
    USA_derivative = "call"

    eu_binomial = optionValue(
        max_steps, S, T, K, r, vol, derivative=EU_derivative, model="binomial"
    )[0]
    eu_blackscholes = optionValue(
        max_steps, S, T, K, r, vol, derivative=EU_derivative, model="blackscholes"
    )[0]

    print(
        f"The option value is {eu_binomial} from the binomial tree and {eu_blackscholes} from Black Scholes for N={max_steps}."
    )

    error_eu = converge(
        max_steps, S, T, K, r, vol, option="EU", derivative=EU_derivative
    )

    plt.plot(error_eu, "r-", linewidth=1.2)
    plt.plot(
        max_steps - 1,
        error_eu[-1],
        "bo",
        label=f"Convergence = {round(error_eu[-1], 5)}",
    )

    plt.title("European option value", fontsize=28)
    plt.xlabel("N (steps)", fontsize=25)
    plt.ylabel("Error", fontsize=25)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.show()

    # Compare binomial and black scholes depending on volatility
    max_steps = 100
    N = 50
    error_volatility = compare(max_steps, N, S, T, K, r)

    plt.plot(
        np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:],
        error_volatility,
        "r-",
        linewidth=1.2,
    )

    plt.xlabel("Volatility", fontsize=25)
    plt.ylabel("Error", fontsize=25)
    plt.title("European option value N=50", fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 0.15)
    plt.vlines(0.01, 0, 0.15, linestyle="--")
    plt.vlines(0.22, 0, 0.15, linestyle="--")

    x = [0, 0.01]

    plt.fill_between(x, -0.01, 0.18, alpha=0.5)
    plt.show()

    # Hedge parameter at t=0
    N = 100
    max_steps = 200
    error_hedge_eu = hedge_ratio(max_steps, N, S, T, K, r, derivative=EU_derivative)

    plt.title("Hedge ratio", fontsize=25)
    plt.plot(np.arange(1 / max_steps, 1 + 1 / max_steps, 1 / max_steps), error_hedge_eu)
    plt.xlabel("Volatility", fontsize=25)
    plt.ylabel("Error", fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 1)
    plt.ylim(0, 0.0012)
    plt.show()

    # Simulate early exercise
    N = 100
    max_steps = 200
    call_values, put_values, eu_call, eu_put = americanSimulate(max_steps, T, N, r, K)

    plt.plot(
        np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:],
        call_values,
        "g-",
        label="American call",
    )
    plt.plot(
        np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:],
        put_values,
        "r-",
        label="American put",
    )
    plt.plot(
        np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:],
        eu_call,
        "b-",
        label="European call",
    )
    plt.plot(
        np.arange(0, 1 + 1 / max_steps, 1 / max_steps)[1:],
        eu_put,
        color="orange",
        label="European put",
    )

    plt.title("American option value at T=0", fontsize=25)
    plt.xlabel("Volatility", fontsize=25)
    plt.ylabel("Option value", fontsize=25)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.show()
