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
from numba import jit, njit
#%% FUNCTION
EPS = np.finfo(float).eps
NEWTON_MAXITER = 6  # Maximum number of Newton iterations.
MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1
@jit(nopython=True)
def func(t, y):
    return 3*y

def analytic_sol(t,y0):
    return y0 * np.exp(3*t)

@jit(nopython=True)
def num_jac(fun, t, y, f, threshold, factor):
    # y = np.asarray(y)
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
    f_sign = 2 * (np.real(f) >= 0).astype(np.float64) - 1
    y_scale = f_sign * np.maximum(threshold, np.abs(y))
    h = (y + factor * y_scale) - y
    # Make sure that the step is not 0 to start with. Not likely it will be
    # executed often.
    for i in range(n):
        if h[i] == 0:
            while h[i] == 0:
                factor[i] *= 10
                h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

    return _dense_num_jac(fun, t, y, f, h, factor, y_scale)
@jit(nopython=True)
def _dense_num_jac(fun, t, y, f, h, factor, y_scale):
    n = y.shape[0]
    h_vecs = np.diag(h)
    # h_vecs = np.zeros((n, n))
    # for i in range(n):
    #     h_vecs[i, i] = h[i]
    f_new = fun(t, y[:, None] + h_vecs)
    diff = f_new - f[:, None]
    # f_new = np.empty((n, n))
    # for i in range(n):
    #     y_temp = y.copy()
    #     y_temp[i] += h[i]
    #     f_new[:, i] = fun(t, y_temp)
    # diff = f_new - f
    
    max_ind = np.argmax(np.abs(diff), axis=0)
    # r = np.arange(n)
    # max_diff = np.abs(diff[max_ind, r])
    # scale = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))
    max_diff = np.empty(n)
    scale = np.empty(n)
    for i in range(n):
        max_diff[i] = np.abs(diff[max_ind[i], i])
        scale[i] = np.maximum(np.abs(f[max_ind[i]]), np.abs(f_new[max_ind[i], i]))

    diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
    # if np.any(diff_too_small):
    # ind, = np.nonzero(diff_too_small)
    # new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
    # h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
    # h_vecs[ind, ind] = h_new
    # f_new = fun(t, y[:, None] + h_vecs[:, ind])
    # diff_new = f_new - f[:, None]
    # max_ind = np.argmax(np.abs(diff_new), axis=0)
    # r = np.arange(ind.shape[0])
    # max_diff_new = np.abs(diff_new[max_ind, r])
    # scale_new = np.maximum(np.abs(f[max_ind]), np.abs(f_new[max_ind, r]))

    # update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
    # if np.any(update):
    #     update, = np.nonzero(update)
    #     update_ind = ind[update]
    #     factor[update_ind] = new_factor[update]
    #     h[update_ind] = h_new[update]
    #     diff[:, update_ind] = diff_new[:, update]
    #     scale[update_ind] = scale_new[update]
    #     max_diff[update_ind] = max_diff_new[update]
    if np.any(diff_too_small):
        inds = np.nonzero(diff_too_small)[0]  # 获取满足条件的索引数组
        m = len(inds)
        f_new_update = np.zeros((n,m))
        max_diff_new = np.empty(m)
        scale_new = np.empty(m)
        for idx,ind in enumerate(inds):
            new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
            h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
            h_vecs[ind, ind] = h_new  # 只有一个标量索引，Numba支持
            # 此处需要调整fun的调用方式以接受向量输入
            f_new_update[:,idx] = fun(t, y + h_vecs[:, ind])  
            diff_new = f_new_update - f[:, None]
            max_ind_update = np.argmax(np.abs(diff_new), axis=0)
            
            for i in (len(max_ind_update)):
                max_diff_new[i] = np.abs(diff_new[max_ind_update[i], i])
                scale_new[i] = np.maximum(np.abs(f[max_ind_update[i]]), np.abs(f_new_update[max_ind_update[i], i]))
            update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
            if np.any(update):
                update_inds = np.nonzero(update)
                factor[ind] = new_factor
                h[ind] = h_new
                diff[:, ind] = diff_new[:, ind]
                scale[ind] = scale_new
                max_diff[ind] = max_diff_new

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

def interpolation_radau(t_old, t_current, t_eval, y_old, Q):
    x = (t_eval - t_old) / (t_current - t_old)
    order = Q.shape[1] - 1
    if t_eval.ndim == 0:
        p = np.tile(x, order + 1)
        p = np.cumprod(p)
    else:
        p = np.tile(x, (order + 1, 1))
        p = np.cumprod(p, axis=0)
    y = np.dot(Q, p)
    if y.ndim == 2:
        y += y_old[:, None]
    else:
        y += y_old
    return y
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
        # 注意后续需要求的是y，这里直接求解的是降维度后的W，需要从W求到Z，
        # Z的第一维度表示每一阶段的结果，到外部之后再从Z求到y，只需要使用最终阶段的值，也就是Z[-1]
        if (dW_norm == 0 or
                rate is not None and rate / (1 - rate) * dW_norm < tol):
            converged = True
            break

        dW_norm_old = dW_norm

    return converged, k + 1, Z, rate

def radau_ii_a_step(y_current,y_old, t_current,t_old,dt,dt_old,error_norm_old,func,LU_real,LU_complex,
                    J,jac_factor,is_new_jac,Q,re_tol=1e-1, abs_tol=1e-6):
    dt_current = dt
    f_current = func(t_current,y_current)
    
    rejected = False
    step_accepted = False
    while not step_accepted:
        t_new = t_current + dt_current
        if t_new - t_eval[-1] > 0:
            t_new = t_eval[-1]
            dt_current = t_new - t_current
        # 这里Z0的第一维度是3，因为这个Radau是5阶的，有3组参数，也就是分3阶段求解的，这里Z第一维度的每个占位表示1个阶段
        if Q is None:
            Z0 = np.zeros((3, y_current.shape[0]))
        else:
            t_tem = t_current + dt_current*C
            Z0 = interpolation_radau(t_old, t_current, t_tem, y_old, Q).T - y_current
        
        scale = abs_tol + np.abs(y_current) * re_tol
        converged = False
        while not converged:
            if LU_real is None or LU_complex is None:
                LU_real = lu_factor(MU_REAL / dt_current *I - J, overwrite_a=True)
                LU_complex = lu_factor(MU_COMPLEX / dt_current * I - J, overwrite_a=True)
    
            converged, n_iter, Z, rate = solve_collocation_system(
                func, t_current, y_current, dt_current, Z0, scale, newton_tol,
                LU_real, LU_complex)
    
            if not converged:
                if is_new_jac:
                    break
              # 这一段是某些情况下后续程序会判断没有必要更新J，所以上边的迭代使用上一步的J计算的，可能会发散
              # 这种情况下就重新计算一下J，如果还发散，就缩小dt（或者一开始就是用的新的J计算的，也一样）
                J,jac_factor = num_jac(func, t_current, y_current, f_current, abs_tol, jac_factor)
                is_new_jac = True
                LU_real = None
                LU_complex = None
        # 求解方程组时没有收敛，尝试缩小dt，这里应当设置一个允许的最小时间步，否则可能会无限循环
        if not converged:
            dt_current *= 0.5
            LU_real = None
            LU_complex = None
            continue
        
        # 根据误差估计再调整一次时间步，但只调整一次（为啥？）
        y_new = y_current + Z[-1]
        ZE = Z.T.dot(E) / dt
        error = lu_solve(LU_real, f_current + ZE, overwrite_b=True)
        scale = abs_tol + np.maximum(np.abs(y_current), np.abs(y_new)) * re_tol
        error_norm = norm(error / scale)
        safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER
                                                   + n_iter)
    
        if rejected and error_norm > 1:
            error = lu_solve(LU_real, func(t_current, y_current + error) + ZE, overwrite_b=True)
            error_norm = norm(error / scale)
    
        if error_norm > 1:
            factor = predict_factor(dt_current, dt_old,
                                    error_norm, error_norm_old)
            dt_current *= max(MIN_FACTOR, safety * factor)
    
            LU_real = None
            LU_complex = None
            rejected = True
        else:
            step_accepted = True
    # 根据变化率和迭代数判断是否需要在下一时间步中重新计算雅可比矩阵
    recompute_jac = n_iter > 2 and rate > 1e-3
    # 只是对下一步要用的dt的初始值进行了一定的放大，不影响后续本步中的t_new
    factor = predict_factor(dt_current, dt_old, error_norm, error_norm_old)
    factor = min(MAX_FACTOR, safety * factor)
    # 为factor设置一个阈值，低于此阈值就不让dt发生变化，防止一点小波动就对dt的频繁改变
    if not recompute_jac and factor < 1.2:
        factor = 1
    else:
        LU_real = None
        LU_complex = None

    f_new = func(t_new, y_new)
    # 如果需要，重新计算下一步骤中要用的雅可比矩阵
    if recompute_jac:
        J,jac_factor = num_jac(func, t_new, y_new, f_new, abs_tol, jac_factor)
        is_new_jac = True
    y_old = y_current
    t_old = t_current
    dt_old = dt
    error_norm_old = error_norm
    dt_new = dt_current * factor
    Q = np.dot(Z.T, P)
    
    return y_new, t_new, dt_new, y_old,t_old, dt_old, error_norm_old, LU_real,LU_complex,J, jac_factor, is_new_jac,Q

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
    
    # Interpolator coefficients.
    P = np.array([
    [13/3 + 7*S6/3, -23/3 - 22*S6/3, 10/3 + 5 * S6],
    [13/3 - 7*S6/3, -23/3 + 22*S6/3, 10/3 - 5 * S6],
    [1/3, -8/3, 10/3]])
    
    y0 = np.array([0.0,0.5,0.4,0.6,1.0])
    t_eval=np.arange(0, 10, 1, dtype=float)
       
    y_res_tem_list = []
    t_res_tem_list = []
    erro_list = []
    Q_list = []
    
    t_current = 0
    y_current = y0
    f_current = func(t_current,y_current)
    dt = 0.1 
    dt_old = None
    error_norm_old = None
    jac_factor = np.full(y0.shape[0], EPS ** 0.5)
    LU_real = None
    LU_complex = None
    I = np.identity(y_current.size)
    # re_tol基本可以认为就是数值解和解析解之间的误差值，比如某个时间点上y的正确值为1e14级别的，1e-6的相对误差仍然会导致最大1e8级别的误差！
    re_tol=1e-1
    abs_tol=1e-6
    newton_tol = max(10 * EPS / re_tol, min(0.03, re_tol ** 0.5))
    is_new_jac=True
    # 计算初始状态的雅可比矩阵
    J,jac_factor = num_jac(func, t_current, y_current, f_current, abs_tol, jac_factor)
    y_res_tem_list.append(y_current)
    t_res_tem_list.append(t_current)
    t_eval_i = 1  # t_eval的当前索引
    t_old = t_eval[0]
    y_old = np.copy(y0)
    Q=None
    y_old=None
    while t_current < t_eval[-1] and t_eval_i < len(t_eval):
        
        y_new, t_new, dt_new, y_old, t_old,dt_old, error_norm_old, LU_real,LU_complex,J, jac_factor, is_new_jac, Q = radau_ii_a_step(
            y_current,y_old,t_current,t_old,dt,dt_old,error_norm_old,func,LU_real,LU_complex,J,jac_factor,is_new_jac,Q,
            re_tol=re_tol, abs_tol=abs_tol)
        # new是指用于下一个时间步的，current是当前时间步内的，old指刚进入当前时间步时使用的，
        # 所以每次进入下一个时间步，需要把new的值赋给current，这只是重命名，方便理解，无实际意义
        # 这里dt之所以不加current，是因为dt还牵扯到需要在step内保留其原始值，方便在最后获取dt_old，
        y_current = y_new
        t_current = t_new
        dt = dt_new

        y_res_tem_list.append(y_current)
        t_res_tem_list.append(t_current)
        erro_list.append(error_norm_old)
        Q_list.append(Q)
    
        
    y_res_tem = np.array(y_res_tem_list).T
    t_res_tem = np.array(t_res_tem_list)   
    Q_res_tem = np.array(Q_list)  
 
    # 使用searchsorted找到t_eval中每个时间点对应的区间索引
    indexes = np.searchsorted(t_res_tem, t_eval) - 1
    
    # 预分配y_evaluated数组
    y_evaluated = np.zeros((len(y0), len(t_eval)))
    y_evaluated[:,0] = y0
    for i, index in enumerate(indexes):
        if i == 0:
            continue
        t_old = t_res_tem[index] 
        t_current = t_res_tem[index + 1]
        y_old = y_res_tem[:,index] 
        # 注意Q，或者说Z是个过程量，其本身比t和y短1，但是是短在结束时间点上，中间的对应关系不变
        Q = Q_res_tem[index]
        # 计算插值
        y_evaluated[:,i] = interpolation_radau(t_old, t_current, t_eval[i], y_old, Q)

    # y_analytic = np.zeros((len(t_eval),len(y0)))
    # for idt, t_val in enumerate(t_eval):
    #     y_analytic[idt,:] = analytic_sol(t_val, y0)
    
    # y_e = abs(y_results-y_analytic)   
    
    y_analytic = np.zeros((len(y0),len(t_eval)))
    for idt, t_val in enumerate(t_eval):
        y_analytic[:,idt] = analytic_sol(t_val, y0)
    
    RES = integrate.solve_ivp(func,
                              [0,max(t_eval)], 
                              y0,t_eval=t_eval,
                              method='Radau',first_step=0.1,rtol=1e-1)
    # Reshape and save result to N and t_vec
    y_ivp = RES.y
    
    y_e_ivp=abs(y_analytic-y_ivp)  
    y_e_eval=abs(y_analytic-y_evaluated)
    y_e = abs(y_evaluated-y_ivp)  
    y_e.mean(axis=0)
    print(y_e.mean(axis=0))
    