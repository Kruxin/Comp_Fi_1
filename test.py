import math


def normal_cdf(x):
    "cdf for standard normal"
    q = math.erf(x / math.sqrt(2.0))
    return (1.0 + q) / 2.0


print(normal_cdf(2))
