import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.optimize import minimize
import scipy.cluster.hierarchy as sch
from datetime import date
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =========== hierarchical risk parity 20200103
# ref: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678

class RiskParity():

    def getIVP(self, cov):
        # Compute the inverse-variance portfolio
        ivp = 1. / np.diag(cov)
        ivp /= ivp.sum()
        return ivp

    def getcVaR(self, ret, q=0.05):
        # Compute the value-at-risk
        cvar = -ret[ret < ret.quantile(q)].mean()
#         if (cvar < 0).any():
#             raise Exception('RiskParity::getcVaR() failed: returns are all positive!')
        return cvar

    def getClusterVar(self, cov, cItems):
        # Compute variance per cluster
        cov_ = cov.loc[cItems, cItems]  # matrix slice
        w_ = self.getIVP(cov_).reshape(-1, 1)
        cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return cVar

    def getQuasiDiag(self, link):
        # Sort clustered items by distance
        link = link.astype(int)
        sortIx = pd.Series([link[-1, 0], link[-1, 1]])
        numItems = link[-1, 3]  # number of original items
        while sortIx.max() >= numItems:
            sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
            df0 = sortIx[sortIx >= numItems]  # find clusters
            i = df0.index
            j = df0.values - numItems
            sortIx[i] = link[j, 0]  # item 1
            df0 = pd.Series(link[j, 1], index=i + 1)
            sortIx = sortIx.append(df0)  # item 2
            sortIx = sortIx.sort_index()  # re-sort
            sortIx.index = range(sortIx.shape[0])  # re-index
        return sortIx.tolist()

    def getRecBipart(self, cov, sortIx):
        # Compute HRP alloc
        w = pd.Series(1, index=sortIx)
        cItems = [sortIx]  # initialize all items in one cluster
        while len(cItems) > 0:
            cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if
                      len(i) > 1]  # bi-section
            for i in range(0, len(cItems), 2):  # parse in pairs
                cItems0 = cItems[i]  # cluster 1
                cItems1 = cItems[i + 1]  # cluster 2
                cVar0 = self.getClusterVar(cov, cItems0)
                cVar1 = self.getClusterVar(cov, cItems1)
                # # !!!!!!! test 20201028
                # cVar0 = np.sqrt(cVar0)
                # cVar1 = np.sqrt(cVar1)
                alpha = 1 - cVar0 / (cVar0 + cVar1)
                w[cItems0] *= alpha  # weight 1
                w[cItems1] *= 1 - alpha  # weight 2
        return w

    def correlDist(self, corr):
        # A distance matrix based on correlation, where 0<=d[i,j]<=1
        # This is a proper distance metric
        dist = ((1 - corr) / 2.) ** .5  # distance matrix
        return dist

    def getHRP(self, cov, corr):
        # Construct a hierarchical portfolio
        dist = self.correlDist(corr)
        link = sch.linkage(dist, 'single')
        # dn = sch.dendrogram(link, labels=cov.index.values, leaf_rotation=90)
        # plt.show()
        sortIx = self.getQuasiDiag(link)
        sortIx = corr.index[sortIx].tolist()
        hrp = self.getRecBipart(cov, sortIx)
        return hrp.sort_index()

    def get_hrp_weights(self, returns, returns_TD=None, cov_mode='Classical', ret_window=5, corr_method='pearson'):

        if cov_mode == 'Classical':
            corr = returns.corr(method=corr_method)
            # sigma = returns.std() * np.sqrt(252/ret_window)
            # cov = ((sigma * corr).T * sigma).T
            cov = returns.cov()*(252/ret_window)

        elif cov_mode == 'MeanIgnore':
            ret_sign = (returns/returns.abs()).fillna(0)
            corr = (ret_sign.T.dot(ret_sign))
            diag = np.array([np.diag(corr)])
            count = diag.T.dot(diag)
            corr /= np.sqrt(count)
            sigma = returns.std() * (np.sqrt(252/ret_window))
            cov = ((sigma * corr).T * sigma).T

        elif cov_mode == 'PositiveValue & MeanIgnore':
            returns[returns < 0] = 0.
            ret_sign = (returns/returns.abs()).fillna(0)
            corr = (ret_sign.T.dot(ret_sign))
            diag = np.array([np.diag(corr)])
            count = diag.T.dot(diag)
            corr /= np.sqrt(count)
            sigma = returns.std() * (np.sqrt(252/ret_window))
            cov = ((sigma * corr).T * sigma).T

        elif cov_mode == 'Sigma2cVaR':
            if returns_TD is None:
                raise Exception('RiskParity::get_hrp_weights() failed: No returns_TD for cVaR!')
            corr = returns.corr(method=corr_method)
            cvar = self.getcVaR(returns_TD)
            cov = ((cvar * corr).T * cvar).T

        else:
            raise Exception('RiskParity::get_hrp_weights() failed: error cov_mode!')

        hrp = self.getHRP(cov, corr)
        return pd.Series(hrp)

    # # test 20200108 >> different return window for cov&corr
    # def get_hrp_weights(self, returns):
    #     ret_for_corr = returns['corr']
    #     ret_for_cov = returns['cov']
    #
    #     corr = ret_for_corr.corr()
    #     cov = ret_for_cov.cov()
    #
    #     hrp = self.getHRP(cov, corr)
    #     return pd.Series(hrp)

    def trad_wgt(self, ret, ret_window=5):
        corr = ret.corr()
        cov = ret.cov() * (252 / ret_window)
        # inv = np.linalg.inv(corr)
        inv = np.linalg.inv(cov)
        return inv.sum(axis=1)/inv.sum()

    def iv_wgt(self, ret):
        vol = ret.std()
        wgt_vol = 1/vol
        wgt_vol = wgt_vol / wgt_vol.sum()
        return wgt_vol


def main():
    # debug in this py.

    rp = RiskParity()
    path = r'D:\Users\linjuan\Desktop\index filter'
    df = pd.read_csv(os.path.join(path, 'ast_df.csv'), index_col=0) # test FXRates idx
    df.index = pd.to_datetime(df.index)
    df_ret = df.pct_change().fillna(0.)

    rp.get_hrp_weights(df_ret, cov_mode='Sigma2cVaR')

    # other risk premia tests
    # ERC equal risk contribution
    def risk_parity_weights(df_ret):

        cov = np.cov(df_ret.T) * (np.sqrt(252)**2)
        cov *= 1/(cov.mean())
        mv_pct = 0

        def fct(x):
            return np.std(x*(cov.dot(x))) \
                   # + mv_pct*(x[:-1].dot(cov).dot(x[:-1])) # sum(abs(x[:-1]*(cov.dot(x[:-1]))-x[-1]))

        n = len(cov)
        x0 = np.ones(n) * (1/n)
        cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0})
        bnds = [(0, 1)] * (n)

        return minimize(fct, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':1e4,'tol':1e-10}) # options={'maxiter':1e3,'tol':1e-10}

    w = risk_parity_weights(df_ret)
    print('Method SLSQP: ')
    print(w)

    # ==================== test ERC 'add one same underlying' problem
    rho = 0

    corr_matrix = np.array([[1, rho, rho, rho, rho],
                        [rho, 1, rho, rho, rho],
                        [rho, rho, 1, rho, rho],
                        [rho, rho, rho, 1, 1],
                        [rho, rho, rho, 1, 1]])

    # corr_matrix = np.array([[1, rho, rho, rho, rho],
    #                     [rho, 1, rho, rho, rho],
    #                     [rho, rho, 1, 1, 1],
    #                     [rho, rho, 1, 1, 1],
    #                     [rho, rho, 1, 1, 1]])

    # sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sigma = [0.1, 0.1, 0.1, 0.1, 0.2]

    cov = ((sigma * corr_matrix).T *sigma).T
    cov *= 1/(cov.mean())

    def fct(x):
        return np.sqrt(np.std(x * (cov.dot(x)))) \
            # + mv_pct*(x[:-1].dot(cov).dot(x[:-1])) # sum(abs(x[:-1]*(cov.dot(x[:-1]))-x[-1]))

    n = len(cov)
    x0 = np.ones(n) * (1 / n)
    cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0})
    bnds = [(0, 1)] * (n)

    w = minimize(fct, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-10)
    # print('Method SLSQP: ')
    print(w)


if __name__=='__main__':
    main()

# ==================== cvxpy testing
# rho = 0
# matrix = np.array([[0.25, rho, rho, rho, rho],
#                     [rho, 0.25, rho, rho, rho],
#                     [rho, rho, 0.25, rho, rho],
#                     [rho, rho, rho, 0.25, 0.25],
#                     [rho, rho, rho, 0.25, 0.25]])
#
#
# def fct(x):
#     return sum((x[:5]*(matrix.dot(x[:5]))-x[-1])**2) #x.dot(matrix).dot(x)
#
# # x0 = np.ones(5) / 5
# x0 = np.zeros(5) *0.5
# cons = ({'type': 'eq', 'fun': lambda x: x.sum() - 1.0})
# hess = lambda x, v: np.zeros((5, 5))
# bnds = [(0, 1)] * 5
#
# w = minimize(fct, x0, method='trust-constr', bounds=bnds, constraints=cons, hessp=lambda x, v: np.zeros((5)))#method='SLSQP',
# print('Method trust-constr: ')
# print(w)
#
# w = minimize(fct, x0, method='SLSQP', bounds=bnds, constraints=cons)
# print('Method SLSQP: ')
# print(w)


# rho = 0
# n = 4
# # ------------ test1 >> 4 vars
# Sigma_4 = np.array([0.1, rho, rho, rho,
#                     rho, 0.1, rho, rho,
#                     rho, rho, 0.1, rho,
#                     rho, rho, rho, 0.1])
# # Sigma_4 = np.array([1, 0.8, rho, rho,
# #                     0.8, 1, rho, rho,
# #                     rho, rho, 1, -0.5,
# #                     rho, rho, -0.5, 1]) *0.1
# Sigma_4.resize(n, n)
#
# # Define and solve the CVXPY problem
# w = cp.Variable(n)
# prob = cp.Problem(cp.Minimize(cp.sum((w*Sigma_4@w)**2-(cp.sum(w*Sigma_4@w)/n)**2)),
#                   [cp.sum(w) == 1,
#                    w >= 0])
#
# prob.solve()
# print('4 vars: \n')
# print("\nThe optimal value is", prob.value)
# print("A solution x is")
# print(w.value)
# print()
#
# # ------------ test2 >> 5 vars
# Sigma_5 = cp.Parameter((5, 5), PSD=True)
#
# val = np.array([0.25, rho, rho, rho, rho,
#                     rho, 0.25, rho, rho, rho,
#                     rho, rho, 0.25, rho, rho,
#                     rho, rho, rho, 0.25, rho,
#                     rho, rho, rho, rho, 0.25])
# val.resize(5, 5)
# Sigma_5.value = val
#
# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)
# #
#
# # Define and solve the CVXPY problem
# w = cp.Variable(6)
# # prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(w, Sigma_5)),
# #                   [cp.sum(cp.log(w)) >= -9,
# #                    cp.sum(w) == 1,
# #                    w >= 0])
#
# prob = cp.Problem(cp.Minimize(cp.sum((Sigma_5 @ w[:5] * w[:5] - w[-1])**2)),
#                   [cp.sum(w) == 1,
#                    w >= 0])
# prob.solve()
#
# print('5 vars: \n')
# # Print result.
# print("\nThe optimal value is", prob.value)
# print("A solution x is")
# print(w.value)
#
# print()
