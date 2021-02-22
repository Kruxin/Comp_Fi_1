import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import concurrent.futures

K = 99
S_0 = 100
r = 0.06
sigma = 0.2
sigma_BM = 0.2

def call_price(S_t, t, T):
    """
    Determines the call option price, requires current stock price, current time and total period.
    """
    d_1 = (np.log(S_t/K) + (r+((sigma_BM)**2)/2)*(T-t))/(sigma_BM*np.sqrt(T-t))
    d_2 = d_1 - sigma_BM*np.sqrt(T-t)
    price = S_t*norm.cdf(d_1) - np.exp(-r*(T-t))*K*norm.cdf(d_2)
    return price

def Black_Scholes(hedging_period):
    T = 1
    M = 525600
    dt = T/M
    S_list = [S_0]
    C_list = [call_price(S_list[-1], 0, 1)]
    Delta_list = [norm.cdf((np.log(S_list[-1]/K) + (r+((sigma_BM)**2)/2)*(T)/(sigma_BM*np.sqrt(T))))]
    portfolio_list = [C_list[-1]]
    portfolio = -Delta_list[-1]*S_0 + C_list[-1]
    for m in range(1,M):
        Z_m = np.random.normal(0,1)
        S_list.append(S_list[-1] + r * S_list[-1]*dt + sigma * S_list[-1]*np.sqrt(dt)*Z_m)
        C_list.append(call_price(S_list[-1], m/M, 1))
        if m % hedging_period == 0:
            Delta_list.append(norm.cdf((np.log(S_list[-1]/K) + (r+((sigma_BM)**2)/2)*(T-m/M))/(sigma_BM*np.sqrt(T-m/M))))
            portfolio *= (1+r*dt*hedging_period)
            portfolio += Delta_list[-2]*S_list[-1]
            portfolio_list.append(portfolio)
            portfolio -= Delta_list[-1]*S_list[-1]


    portfolio *=  (1+r*dt*(M % hedging_period))
    portfolio += Delta_list[-1]*S_list[-1]
    error = portfolio - C_list[-1]
    return error

if __name__ == "__main__":
    file_aux  = open('hedge_error.csv','a')
    file_aux.write("hedge_period mean_error std_error")
    file_aux.close()
    # periods = [1, 10, 60, 360, 1440, 2880, 7200, 10080, 20160, 40320, 262800]
    periods = [131400]
    # period_word = ["minute", "10 minutes", "hour", "6 hours", "day", "two days", "5 days", "week", "2 weeks", "month", "6 months"]
    period_word = ["3 months"]
    dict = {"Mean": [], "Std": []}
    for i in range(len(periods)):
        hedge_error = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            values = [executor.submit(Black_Scholes, periods[i]) for _ in range(1000)]

            for f in concurrent.futures.as_completed(values):
                hedge_error.append(f.result())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(hedge_error, bins=75, rwidth=0.2)
        ax.set_xlabel("Hedge error in €")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Hedging error from updating portfolio every {period_word[i]}")
        fig.savefig(f"hedgesigma{sigma_BM}_{period_word[i]}_test.jpg")

        mean_error = np.mean(hedge_error)
        std_error = np.std(hedge_error)
        dict["Mean"].append(mean_error)
        dict["Std"].append(std_error)
        file_aux  = open('hedge_error.csv','a')
        file_aux.write("\n"+str(periods[i])+" "+str(mean_error)+" "+str(std_error))
        print(f"{periods[i]} is done")
        file_aux.close()



    # fig = plt.figure(figsize=(11,5))
    # ax = fig.add_subplot(111)
    # ax.set_xticks(range(1,12))
    # ax.set_xticklabels(period_word, fontsize=6.5)
    # ax.errorbar(x, dict["Mean"], yerr=dict["Std"], capsize=5)
    # ax.set_ylabel("Mean hedging error")
    # ax.set_xlabel("Hedge update period")
    # ax.set_title("Mean hedging error versus the hedge update period")
    # ax.savefig(f"meanerror.jpg")
