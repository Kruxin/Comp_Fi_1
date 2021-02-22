import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import concurrent.futures

# simulation parameters
K = 99
S_0 = 100
r = 0.06
sigma = 0.2

def call_price(S_t, t, T, sigma_BM):
    """
    Determines the call option price.
    :param S_t: stock price at time t
    :param t: current time
    :param T: time of expiry
    :param sigma_BM: implied volatility
    :return call option price
    """
    d_1 = (np.log(S_t/K) + (r+((sigma_BM)**2)/2)*(T-t))/(sigma_BM*np.sqrt(T-t))
    d_2 = d_1 - sigma_BM*np.sqrt(T-t)
    price = S_t*norm.cdf(d_1) - np.exp(-r*(T-t))*K*norm.cdf(d_2)
    return price

def Black_Scholes(sigma_BM, hedging_period):
    """
    Determines the hedging error at expiry.
    :param sigma_BM: implied volatility
    :param hedging_period: portfolio re-balacing frequency
    :return: Hedging error at option expiry
    """
    T = 1
    M = 525600
    dt = T/M

    # enter starting values of the stock price, call price, delta parameter and portfolio
    S_list = [S_0]
    C_list = [call_price(S_list[-1], 0, 1, sigma_BM)]
    Delta_list = [norm.cdf((np.log(S_list[-1]/K) + (r+((sigma_BM)**2)/2)*(T)/(sigma_BM*np.sqrt(T))))]
    portfolio_list = [C_list[-1]]
    portfolio = -Delta_list[-1]*S_0 + C_list[-1]

    for m in range(1,M):

        # Brownian motion for stock price and call price
        Z_m = np.random.normal(0,1)
        S_list.append(S_list[-1] + r * S_list[-1]*dt + sigma * S_list[-1]*np.sqrt(dt)*Z_m)
        C_list.append(call_price(S_list[-1], m/M, 1, sigma_BM))

        # Every hedging period the portfolio is rebalanced and the delta parameter is recalculated
        if m % hedging_period == 0:
            Delta_list.append(norm.cdf((np.log(S_list[-1]/K) + (r+((sigma_BM)**2)/2)*(T-m/M))/(sigma_BM*np.sqrt(T-m/M))))
            portfolio *= (1+r*dt*hedging_period)
            portfolio += Delta_list[-2]*S_list[-1]
            portfolio_list.append(portfolio)
            portfolio -= Delta_list[-1]*S_list[-1]

    # Portfolio is updated at expiry by selling all stocks
    portfolio *=  (1+r*dt*(M % hedging_period))
    portfolio += Delta_list[-1]*S_list[-1]

    # hedge error is determined
    error = portfolio - C_list[-1]
    return error

if __name__ == "__main__":
    # Simulations are ran 1000 times for the mentioed estimated volatilities and rebalacing frequencies, all errors are plotted
    # into histograms and the mean and std are saved in a csv for post processing purposes
    file_aux  = open('est_vol.csv','a')
    file_aux.write("est_vol period mean_error std_error")
    file_aux.close()
    volatilities = [0.05,0.1,0.2,0.3,0.4]
    periods = [1, 10, 60, 360, 1440, 2880, 7200, 10080, 20160, 40320, 262800]
    dict = {"Mean": [], "Std": []}
    for i in range(len(volatilities)):
        for j in range(len(periods)):
            vol_error = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                values = [executor.submit(Black_Scholes, volatilities[i], periods[j]) for _ in range(1000)]

                for f in concurrent.futures.as_completed(values):
                    vol_error.append(f.result())

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.hist(vol_error, bins=75, rwidth=0.2)
            ax.set_xlabel("Hedge error in €")
            ax.set_ylabel("Frequency")
            fig.savefig(f"hedgesigma{volatilities[i]}_{periods[j]}.jpg")

            mean_error = np.mean(vol_error)
            std_error = np.std(vol_error)
            dict["Mean"].append(mean_error)
            dict["Std"].append(std_error)
            file_aux  = open('est_vol_error.csv','a')
            file_aux.write("\n"+str(volatilities[i])+" "+str(periods[j])+" "+str(mean_error)+" "+str(std_error))
            file_aux.close()
