# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:15:05 2024

@author: px2030
"""

import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.linalg import lu_factor, lu_solve
#%% FUNCTION
EPS = np.finfo(float).eps
NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

def func(t, y):
    return 3*y

def analytic_sol(t,y0):
    return y0 * np.exp(3*t)

def num_jac(fun, t, y, f, threshold, factor):
    y = np.asarray(y)
    n = y.shape[0]
    if n == 0:
        return np.empty((0, 0)), factor
    if factor is None:
        factor = np.full(n, EPS ** 0.5)
    else:
        factor = factor.copy()
    # Direct the step as ODE dictates, hoping that such a step won't lead to
    # a problematic region. For complex ODEs it makes sense to use the real
    # part of f as we use steps along real axis.
    f_sign = 2 * (np.real(f) >= 0).astype(float) - 1
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y
    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in np.nonzero(h == 0)[0]:
        while h[i] == 0:
            factor[i] *= 10
            h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

    return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
def _dense_num_jac(fun, t, y, f, h, factor, y_scale):

    NUM_JAC_DIFF_REJECT = EPS ** 0.875
    NUM_JAC_DIFF_SMALL = EPS ** 0.75
    NUM_JAC_DIFF_BIG = EPS ** 0.25
    NUM_JAC_MIN_FACTOR = 1e3 * EPS
    NUM_JAC_FACTOR_INCREASE = 10
    NUM_JAC_FACTOR_DECREASE = 0.1
    
    n = y.shape[0]
    h_vecs = np.diag(h)
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    max_ind = np.argmax(np.abs(diff), axis=0)
    r = np.arange(n)
    max_diff = np.abs(diff[max_ind, r])
    scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    if np.any(diff_too_small):
        ind, = np.nonzero(diff_too_small)
        new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
        h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
        h_vecs[ind, ind] = h_new
        f_new = fun(t, y[:, None] + h_vecs[:, ind])
        diff_new = f_new - f[:, None]
        max_ind = np.argmax(np.abs(diff_new), axis=0)
        r = np.arange(ind.shape[0])
        max_diff_new = np.abs(diff_new[max_ind, r])
        scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

        update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
        if np.any(update):
            update, = np.nonzero(update)
            update_ind = ind[update]
            factor[update_ind] = new_factor[update]
            h[update_ind] = h_new[update]
            diff[:, update_ind] = diff_new[:, update]
            scale[update_ind] = scale_new[update]
            max_diff[update_ind] = max_diff_new[update]

    diff /= h

    factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
    factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
    factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

    return diff, factor

def predict_factor(dt, dt_old, error_norm, error_norm_old):
    # E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
    # Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    if error_norm_old is None or dt_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = dt / dt_old * (error_norm_old / error_norm) ** 0.25

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25

    return factor
def norm(x):
    # Compute RMS norm.
    return np.linalg.norm(x) / x.size ** 0.5
#%% RADAU
def solve_collocation_system(fun, t, y, h, Z0, scale, tol,
                             LU_real, LU_complex):
    n = y.shape[0]
    M_real = MU_REAL / h
    M_complex = MU_COMPLEX / h

    W = TI.dot(Z0)
    Z = Z0

    F = np.empty((3, n))
    ch = h * C

    dW_norm_old = None
    dW = np.empty_like(W)
    converged = False
    rate = None
    for k in range(NEWTON_MAXITER):
        for i in range(3):
            F[i] = fun(t + ch[i], y + Z[i])

        if not np.all(np.isfinite(F)):
            break

        f_real = F.T.dot(TI_REAL) - M_real * W[0]
        f_complex = F.T.dot(TI_COMPLEX) - M_complex * (W[1] + 1j * W[2])

        dW_real = lu_solve(LU_real, f_real, overwrite_b=True)
        dW_complex = lu_solve(LU_complex, f_complex, overwrite_b=True)

        dW[0] = dW_real
        dW[1] = dW_complex.real
        dW[2] = dW_complex.imag

        dW_norm = norm(dW / scale)
        if dW_norm_old is not None:
            rate = dW_norm / dW_norm_old

        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dW_norm > tol)):
            break

        W += dW
        Z = T.dot(W)

        if (dW_norm == 0 or
                rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        dW_norm_old = dW_norm

    return converged, k + 1, Z, rate

def radau_ii_a_step(y_current,t_current,dt,dt_old,error_norm_old,func,jac_factor,LU_real,LU_complex,
                    re_tol=1e-1, abs_tol=1e-6):
    dt_current = dt
    f_current = func(t_current,y_current)
    J,jac_factor = num_jac(func, t_current, y_current, f_current, abs_tol, jac_factor)
    I = np.identity(y_current.size)
    newton_tol = max(10 * EPS / re_tol, min(0.03, re_tol ** 0.5))
    
    rejected = False
    step_accepted = False
    while not step_accepted:
        t_new = t_current + dt_current 
        # 这里Z0的第一维度是3，因为这个Radau是5阶的，有3组参数，也就是分3阶段求解的，这里Z第一维度的每个占位表示1个阶段
        Z0 = np.zeros((3, y_current.shape[0]))
        
        scale = abs_tol + np.abs(y_current) * re_tol
        converged = False
        while not converged:
            if LU_real is None or LU_complex is None:
                LU_real = lu_factor(MU_REAL / dt_current *I - J, overwrite_a=True)
                LU_complex = lu_factor(MU_COMPLEX / dt_current * I - J, overwrite_a=True)
    
            converged, n_iter, Z, rate = solve_collocation_system(
                func, t_current, y_current, dt_current, Z0, scale, newton_tol,
                LU_real, LU_complex)
    
            # if not converged:
            #     if current_jac:
            #         break
            #   这一段是某些情况下后续程序会判断没有必要更新J，所以上边的迭代使用上一步的J计算的，可能会发散
            #   这种情况下就重新计算一下J，如果还发散，就缩小dt（或者一开始就是用的新的J计算的，也一样）
            #     J = self.jac(t, y, f)
            #     current_jac = True
            #     LU_real = None
            #     LU_complex = None
        # 求解方程组时没有收敛，尝试缩小dt，这里应当设置一个允许的最小时间步，否则可能会无限循环
        if not converged:
            dt_current *= 0.5
            LU_real = None
            LU_complex = None
            continue
        
        # 根据误差估计再调整一次时间步，但只调整一次（为啥？）
#%% MAIN   
if __name__ == "__main__":
    S6 = 6 ** 0.5
    # Butcher tableau. A is not used directly, see below.
    C = np.array([(4 - S6) / 10, (4 + S6) / 10, 1])
    E = np.array([-13 - 7 * S6, -13 + 7 * S6, -1]) / 3
    
    # Eigendecomposition of A is done: A = T L T**-1. There is 1 real eigenvalue
    # and a complex conjugate pair. They are written below.
    MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)
    MU_COMPLEX = (3 + 0.5 * (3 ** (1 / 3) - 3 ** (2 / 3))
                  - 0.5j * (3 ** (5 / 6) + 3 ** (7 / 6)))
    
    # These are transformation matrices.
    T = np.array([
        [0.09443876248897524, -0.14125529502095421, 0.03002919410514742],
        [0.25021312296533332, 0.20412935229379994, -0.38294211275726192],
        [1, 1, 0]])
    TI = np.array([
        [4.17871859155190428, 0.32768282076106237, 0.52337644549944951],
        [-4.17871859155190428, -0.32768282076106237, 0.47662355450055044],
        [0.50287263494578682, -2.57192694985560522, 0.59603920482822492]])
    # These linear combinations are used in the algorithm.
    TI_REAL = TI[0]
    TI_COMPLEX = TI[1] + 1j * TI[2]
    
    y0 = np.array([0.0,0.5,0.4,0.6,1.0])
    t_eval=np.arange(0, 10, 1, dtype=float)
    
    y = np.zeros((len(y0),len(t_eval)))
    y[:,0] = y0
    
    y_res_tem_list = []
    t_res_tem_list = []
    erro_list = []
    
    t_current = 0
    y_current = y0
    dt = 0.01 
    dt_old = None
    error_norm_old = None
    jac_factor = None
    LU_real = None
    LU_complex = None
    
    while t_current < max(t_eval):
        y_current,dt_new,erro_norm_old = radau_ii_a_step(y_current,t_current,dt,dt_old,error_norm_old,
                                                     func,jac_factor)