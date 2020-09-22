# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:41:19 2020

@author: zhangliangliang
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist
from collections import OrderedDict
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import norm, t
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
default_bounds = {
    'l': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4],
    'period': [1e-3, 10]
}
class EventLogger:
    def __init__(self, gpgo):
        self.gpgo = gpgo
        self.header = 'Evaluation \t Proposed point \t  Current eval. \t Best eval.'
        self.template = '{:6} \t {}. \t  {:6} \t {:6}'
        print(self.header)

    def _printCurrent(self, gpgo):
        eval = str(len(gpgo.GP.y) - gpgo.init_evals)
        proposed = str(gpgo.best)
        curr_eval = str(gpgo.GP.y[-1])
        curr_best = str(gpgo.tau)
        if float(curr_eval) >= float(curr_best):
            curr_eval = bcolors.OKGREEN + curr_eval + bcolors.ENDC
        print(self.template.format(eval, proposed, curr_eval, curr_best))

    def _printInit(self, gpgo):
        for init_eval in range(gpgo.init_evals):
            print(self.template.format('init', gpgo.GP.X[init_eval], gpgo.GP.y[init_eval], gpgo.tau))


def l2norm_(X, Xstar):
    return cdist(X, Xstar)


def kronDelta(X, Xstar):
    return cdist(X, Xstar) < np.finfo(np.float32).eps


class matern52:
    def __init__(self, l=1, sigmaf=1, sigman=1e-6, bounds=None, parameters=['l', 'sigmaf', 'sigman']):
        self.l = l
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def K(self, X, Xstar):
        r = l2norm_(X, Xstar)/self.l
        one = (1 + np.sqrt(5 * r ** 2) + 5 * r ** 2 / 3)
        two = np.exp(-np.sqrt(5 * r ** 2))
        return self.sigmaf * one * two + self.sigman * kronDelta(X, Xstar)  #如果cdist是一个极小值，就加上sigman


        
import numpy as np
from scipy.linalg import cholesky, solve
from collections import OrderedDict
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, covfunc, optimize=False, usegrads=False, mprior=0):
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.mprior = mprior

    def getcovparams(self):

        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.nsamples = self.X.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads: grads = self._grad
            self.optHyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.K(self.X, self.X)
        # print("self.K  ",self.K  )
        # self.alpha = solve(self.K,  y - self.mprior)
        # print("self.alpha ",self.belta )
        #it is typically faster and more numerically stable to use a Cholesky decomposition
        self.L = cholesky(self.K).T
        self.alpha = solve(self.L.T, solve(self.L, y - self.mprior))
        # print("self.alpha",self.alpha)
        # print("self.y",self.y)
        # print("np.dot(self.y, self.alpha)",np.dot(self.y, self.alpha))

        # self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log( 2 * np.pi)

    def param_grad(self, k_param):

        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param)
        K = covfunc.K(self.X, self.X)
        L = cholesky(K).T
        alpha = solve(L.T, solve(L, self.y))
        inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(K)
        grads = []
        for param in k_param_key:
            gradK = covfunc.gradK(self.X, self.X, param=param)
            gradK = .5 * np.trace(np.dot(inner, gradK))
            grads.append(gradK)
        return np.array(grads)



    def predict(self, Xstar, return_std=False):

        Xstar = np.atleast_2d(Xstar)
        kstar = self.covfunc.K(self.X, Xstar).T
        fmean = self.mprior + np.dot(kstar, self.alpha)
        v = solve(self.L, kstar.T)
        fcov = self.covfunc.K(Xstar, Xstar) - np.dot(v.T, v)
        if return_std:
            fcov = np.diag(fcov)
        return fmean, fcov

    def update(self, xnew, ynew):
        y = np.concatenate((self.y, ynew), axis=0)
        X = np.concatenate((self.X, xnew), axis=0)
        self.fit(X, y)
        
        
class Acquisition:
    def __init__(self, mode, eps=1e-06, **params):
        self.params = params
        self.eps = eps
        mode_dict = {
            'ExpectedImprovement': self.ExpectedImprovement,
        }

        self.f = mode_dict[mode]

    def ExpectedImprovement(self, tau, mean, std):
        z = (mean - tau - self.eps) / (std + self.eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]
    
    def eval(self, tau, mean, std):
        return self.f(tau, mean, std, **self.params)
    
class GPGO:
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1):
        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []
        self.logger = EventLogger(self)

    def _sampleParam(self):

        d = OrderedDict()
        for index, param in enumerate(self.parameter_key):
            if self.parameter_type[index] == 'int':
                d[param] = np.random.randint(self.parameter_range[index][0], self.parameter_range[index][1])
            elif self.parameter_type[index] == 'cont':
                d[param] = np.random.uniform(self.parameter_range[index][0], self.parameter_range[index][1])
            else:
                raise ValueError('Unsupported variable type.')
        
        return d

    def _firstRun(self, n_eval=3):

        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval): #随机生成n_eval个点（xi,yi)，并使用GP.fit（）
            s_param = self._sampleParam()         
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)

    def _acqWrapper(self, xnew):

        new_mean, new_var = self.GP.predict(xnew, return_std=True)
        new_std = np.sqrt(new_var + 1e-6)
        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimizeAcq(self, method='L-BFGS-B', n_start=100):

        start_points_dict = [self._sampleParam() for i in range(n_start)]
        
        start_points_arr = np.array([list(s.values()) for s in start_points_dict])
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_arr):
                res = minimize(self._acqWrapper, x0=start_point, method=method,bounds=self.parameter_range)
                # print("res:",res.x, np.atleast_1d(res.fun)[0])
                x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acqWrapper, x0=start_point,method=method,
                                                                 bounds=self.parameter_range) for start_point in start_points_arr)           
            x_best = np.array([res.x for res in opt])
            f_best = np.array([np.atleast_1d(res.fun)[0] for res in opt])
                    

        self.best = x_best[np.argmin(f_best)]

    def updateGP(self):
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.f(**kw)
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.GP.y)
        self.history.append(self.tau)

    def getResult(self):

        argtau = np.argmax(self.GP.y)
        opt_x = self.GP.X[argtau]
        res_d = OrderedDict()
        for i, (key, param_type) in enumerate(zip(self.parameter_key, self.parameter_type)):
            if param_type == 'int':
                res_d[key] = int(opt_x[i])
            else:
                res_d[key] = opt_x[i]
        return res_d, self.tau

    def run(self, max_iter=10, init_evals=3, resume=False):
        if not resume:
            self.init_evals = init_evals
            self._firstRun(self.init_evals)
            self.logger._printInit(self)
        for iteration in range(max_iter):
            self._optimizeAcq()
            self.updateGP()
            self.logger._printCurrent(self)
            
            
def drawFun(f):
    x = np.linspace(0, 1, 1000)
    plt.plot(x, f(x))
    plt.grid()
    plt.show()

def f(x):
    return -((6*x-2)**2)
    # return (x-2)**2

if __name__ == '__main__':
    np.random.seed(20)
    drawFun(f)
    sexp = matern52()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode = 'ExpectedImprovement')

    params = {'x': ('cont', (-2, 4))}
    gpgo = GPGO(gp, acq, f, params)
    gpgo.run(max_iter = 20)
    print(gpgo.getResult())

            
            
            
        
        


        
        
        
        
        