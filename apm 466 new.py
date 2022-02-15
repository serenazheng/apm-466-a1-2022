import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import datetime
import math


# get relevant data from excel
df = pd.read_excel("10 bond data cleaned.xlsx")
date_list = pd.read_excel("date list.xlsx")["Date list"]
coup = df['Coupon']
bond_list = df['Bond']
df["T"] = [round((pd.to_datetime(df["Maturity Date"][i]) - pd.to_datetime('02/14/2022')).days / 365, 2) for i in range(11)]
bond_index = [0,1,2,3,4,5,6,7,8,9,10]
maturity_date = df["Maturity Date"]

# convert to dirty price by adding accrued interest
def dirty_price(bond):
    date = [10, 11, 12, 13, 14, 17, 18, 19, 20, 21]
    for id in bond_index:
        for index in range(0, 10):
            bond_day = datetime.date.fromisoformat("2022-01-{}".format(date[index]))
            maturity_day = df["Maturity Date"][id].to_pydatetime().date()
            remaining_dd = (maturity_day - bond_day).days % 182
            coup = df["Coupon"][id] * 100
            accrued_i = remaining_dd / 365 * coup
            bond.iloc[id, 5 + index] += accrued_i

dirty_price(df)
print(df.to_string())

# calculate yield to maturity from Mastering Python for Finance By James Ma Weiming page 133
def ytm_calculator(price, par, T, coup, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * freq
    coupon = coup / 100 * par / freq
    dt = [(i + 1) / freq for i in range(int(periods))]
    ytm_func = lambda y:  sum([coupon/(1+y/freq)**(freq*t) for t in dt]) + par/(1+y/freq)**(periods) - price
    return optimize.newton(ytm_func, guess)

def ytm(bond):
    ytm_matrix = pd.DataFrame(columns = bond_list, index = date_list)
    for id in bond_index:
        for dd in range(0, 10):
            if id != 0:
                dirty_price = bond.iloc[id, 5 + dd]
                time = bond.iloc[id, 15]
                coup = bond.iloc[id, 1]*100
                ytm_matrix.iloc[dd, id] = ytm_calculator (dirty_price, 100, time, coup, freq=2, guess=0.05)*100
    return ytm_matrix

def plot_ytm(bond_list, ytm_matrix):
    plt.plot(ytm_matrix)
    plt.xlabel('Day')
    plt.ylabel('Yield to Maturity (%)')
    plt.title('Bond Yield')
    plt.legend(labels=bond_list, loc = 5, prop={"size":5})
    plt.show()

ytm_m = ytm(df)
plot_ytm(bond_list, ytm_m)

# spot rate and plot
def spot_rate(bonds, dirty_price_l, r1, curr_date):
    summation = 0
    spot_rate = []
    spot_rate.append(r1)
    coupon = bonds["Coupon"] * 100 / 2
    for id in range(1,11):
        T = bonds["T"][id]
        summation += coupon[id-1]*math.exp(-spot_rate[id-1]*(T))
        dp = dirty_price_l[id]
        r = -math.log((dp-summation)/(coupon[id]+100))/(T)
        rate = r * 100
        spot_rate.append(rate)
    return spot_rate

def spot_matrix(bonds):
    spot_matrix = pd.DataFrame(columns=date_list, index=bonds["Bond"])
    zero_coupon_bond_l = []
    for dd in range(len(date_list)):
        par_value = 100
        curr_date = date_list[dd]
        maturity_date = bonds["Maturity Date"][dd]
        T = (maturity_date - curr_date).days / 365
        dirty_price = bonds[curr_date][0]
        zcb = -math.log(dirty_price / (par_value)) / T * 100
        zero_coupon_bond_l.append(zcb)
    spot_matrix.iloc[0,:] = zero_coupon_bond_l
    for dd in range(len(date_list)):
        curr_date = date_list[dd]
        dirty_price_l = bonds[curr_date]
        r1 = spot_matrix.iloc[0, dd]
        spot_matrix.iloc[:, dd] = spot_rate(bonds, dirty_price_l, r1, curr_date)
    return spot_matrix

def plot_s(s):
    for dd in range(len(date_list)):
        plt.plot(df["T"], s.iloc[dd,:])
    plt.xlabel('Time')
    plt.ylabel('Spot Rate (%)')
    plt.title('Bond Spot Curve')
    plt.legend(labels=date_list, loc = 5, prop={"size":5})
    plt.show()

spot_m = spot_matrix(df)
plot_s(spot_m)

# forward rate and plot

def forward_matrix(spot_m):
    forward_matrix = pd.DataFrame(columns=date_list, index=["1yr-1yr","1yr-2yr", "1yr-3yr", "1yr-4yr"])
    for dd in range(len(date_list)):
        for i in range(0, 4):
            n = (1 + spot_m.iloc[(i + 1) * 2, dd] ) ** (i + 1)
            d = (1 + spot_m.iloc[i*2, dd]) ** i
            forward = n / d - 1
            forward_matrix.iloc[i, dd] = forward / 100
    return forward_matrix

def plot_forward(forward_m):
    plt.plot(forward_m)
    plt.xlabel('Time')
    plt.ylabel('Forward Rate (%)')
    plt.title('Forward Curve')
    plt.legend(labels=date_list, loc = 5, prop={"size":5})
    plt.show()


forward_m = forward_matrix(spot_m)
plot_forward(forward_m)

# PCA

def ytm_cov(ytm_matrix):
    cov_mat = np.zeros([9, 5])
    for i in range(0,5):
        for j in range(1, 10):
            year = 2 * i
            X_ij = np.log((ytm_matrix.iloc[year, j]) / (ytm_matrix.iloc[year, j - 1]))
            cov_mat[j - 1, i] = X_ij
    ytm_cov = np.cov(cov_mat.T)
    return ytm_cov

def forward_cov(forward_matrix):
    cov_mat = np.zeros([9, 4])
    for i in range(0, 4):
        for j in range(1, 10):
            X_ij = np.log((forward_matrix.iloc[i, j]) / (forward_matrix.iloc[i, j - 1]))
            cov_mat[j - 1, i] = X_ij
    forward_cov = np.cov(cov_mat.T)
    return forward_cov

ytm_cov = ytm_cov(x)
print(ytm_cov)
forward_cov = forward_cov(forward_m)
print(forward_cov)


def eigen(matrix):
    eig_val, eig_vec = np.linalg.eig(matrix)
    print(eig_val,eig_vec)

eigen(ytm_cov)
eigen(forward_cov)