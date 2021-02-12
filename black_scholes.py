import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

K = 99
S_0 = 100
r = 0.06
sigma = 0.2
sigma_BM = 0.2

def call_price(S_t, t, T):
  d_1 = (np.log(S_t/K) + (r+((sigma)**2)/2)*(T-t))/(sigma*np.sqrt(T-t))
  d_2 = d_1 - sigma*np.sqrt(T-t)
  price = S_t*norm.cdf(d_1) - np.exp(-r*(T-t))*K*norm.cdf(d_2)
  return price
  
def Black_Scholes(hedging_period):
  T = 1
  M = 1000
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
    #Delta_list.append((C_list[-1]-C_list[-2])/(S_list[-1]-S_list[-2]))
    if m % hedging_period == 0:
        Delta_list.append(norm.cdf((np.log(S_list[-1]/K) + (r+((sigma_BM)**2)/2)*(T-m/M))/(sigma_BM*np.sqrt(T-m/M))))
        portfolio *= (1+r*dt)
        portfolio += Delta_list[-2]*S_list[-1]
        portfolio_list.append(portfolio)
        portfolio -= Delta_list[-1]*S_list[-1] 
  return S_list, C_list, Delta_list, portfolio_list
    
a,b,c,d = Black_Scholes(1)
print(b[:20])
print(c[:20])
print(d[:20])
plt.plot(np.linspace(0, 1, len(a)), a, label ='stock')
plt.plot(np.linspace(0, 1, len(b)), b, label='option')
plt.plot(np.linspace(0, 1, len(c)), c, label='hedge param')
plt.plot(np.linspace(0, 1, len(d)), d, label='portfolio')
plt.axhline(y=100)
plt.legend()
plt.show()