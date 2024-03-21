import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate
from scipy.interpolate import interp1d

def func(t, y):
    return 3*y

def analytic_sol(t,y0):
    return y0 * np.exp(3*t)
    
def k_equations(k_flat,a,c,n,dim_y,f,t,y,dt):
    k = k_flat.reshape((n, dim_y))
    t_k = t + c * dt
    y_k = np.array([f(t_k[j], y + dt * np.dot(a[j], k)) for j in range(n)])
    return (k - y_k).flatten()

def predict_factor(h_abs, h_abs_old, error_norm, error_norm_old):
    # E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
    # Equations II: Stiff and Differential-Algebraic Problems", Sec. IV.8.
    if error_norm_old is None or h_abs_old is None or error_norm == 0:
        multiplier = 1
    else:
        multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25

    with np.errstate(divide='ignore'):
        factor = min(1, multiplier) * error_norm ** -0.25

    return factor

def solve_k_system(k_equations,k_guess,a,c,n,dim_y,f,t,y,dt):
    args=(a,c,n,dim_y,f,t,y,dt)
    k = fsolve(k_equations, k_guess, args=args)
    k = k.reshape((n, dim_y))
    dy = dt * np.dot(b, k)
    return dy
# def radau_ii_a_step(y, t, dt, dt_old, error_norm_old, f, a,b,c,a_e,b_e,c_e,
#                     try_t_max=100,re_tol=1e-1, abs_tol=0.0, dt_min=1e-3,
#                     min_factor=0.2):
#     n = len(b)
#     n_e = len(b_e)
#     dim_y = len(y)
#     dt_accepted = False
#     dt_new=dt
    
#     for i in range(try_t_max): 
#         k_guess = np.zeros(n * dim_y)
#         dy = solve_k_system(k_equations,k_guess,a,c,n,dim_y,f,t,y,dt_new)
#         dy_e = solve_k_system(k_equations,k_guess,a_e, c_e,n_e,dim_y,f,t,y,dt_new)
#         y_new = y + dy
        
#         erro = np.abs(dy -dy_e)
#         scale = abs_tol + np.maximum(np.abs(y),np.abs(y_new)) * re_tol
#         error_norm = np.linalg.norm(erro/scale)

#         # If the error is too large, reduce dt
#         # print(f'current dy_diff is {dy_diff}, current tol_up is {tol_up}')
#         if error_norm > 1 and not dt_accepted:
#             factor = predict_factor(dt_new, dt_old, error_norm, error_norm_old)
#             dt_accepted = True
#         elif error_norm > 1:
            
#         else:
#             dt_accepted = True
#             break
    
#     if not dt_accepted:
#         raise RuntimeError(f"Could not find an accepted dt within {try_t_max} iterations")
    
        
#     return y+dy, dt, error_norm

if __name__ == "__main__":
    # Coefficients a_ij in the Butcher tableau for a 5th-order Radau IIA method
    a = np.array([[ 0.19681548, -0.06553543,  0.02377097],
                 [ 0.39442431,  0.29207341, -0.04154875],
                 [ 0.37640306,  0.51248583,  0.11111111]]
                )
    b = np.array([0.37640306, 0.51248583, 0.11111111])
    c = np.array([0.15505103, 0.64494897, 1.0])
     
    # Coefficients in the Butcher tableau for a 3th-order Radau IIA method, 
    # used for erro estimation and time step prediction
    a_e = np.array([[ 0.41666667, -0.08333333],
                   [ 0.75      ,  0.25      ]]    
                  )
    b_e = np.array([0.75, 0.25])
    c_e = np.array([0.33333333, 1.0])
    
    
    
    
    y0 = np.array([0.0,0.5,0.4,0.6,1.0])
    t_eval=np.arange(0, 10, 1, dtype=float)
    
    y = np.zeros((len(y0),len(t_eval)))
    y[:,0] = y0
    
    y_res_tem_list = []
    t_res_tem_list = []
    erro_list = []
    
    current_t = 0
    current_y = y0
    dt = 0.01
    dt_old = None
    error_norm_old = None
    # while current_t < max(t_eval):
    #     new_y, new_dt, erro_norm = radau_ii_a_step(current_y, current_t, dt, dt_old, error_norm_old, 
    #                                           func, a, b, c, a_e, b_e, c_e)
        
    #     current_y = new_y
    #     current_t += dt
    #     # Let the step size of the next time step be tried at a larger value
    #     dt = new_dt
        
    #     y_res_tem_list.append(current_y)
    #     t_res_tem_list.append(current_t)
    #     erro_list.append(erro)
        
    # y_res_tem = np.array(y_res_tem_list).T  # Transpose to match the expected shape
    # t_res_tem = np.array(t_res_tem_list)   
    
    # interp_func = interp1d(t_res_tem, y_res_tem, kind='cubic', axis=1, fill_value="extrapolate")
    # y_results = interp_func(t_eval)
    
    RES = integrate.solve_ivp(func,
                              [0,max(t_eval)], 
                              y0,t_eval=t_eval,
                              method='Radau',first_step=0.1,rtol=1e-9)
    # Reshape and save result to N and t_vec
    y_ivp = RES.y
    y_analytic = np.zeros((len(y0),len(t_eval)))
    for idt, t_val in enumerate(t_eval):
        y_analytic[:,idt] = analytic_sol(t_val, y0)
    
    y_e = abs(y_analytic-y_ivp)