import numpy as np
from scipy.optimize import fsolve
import scipy.integrate as integrate

def func(t, y):
    return -t*y

def equations(k_flat,a,c,n,dim_y,f,t,y,dt):
    k = k_flat.reshape((n, dim_y))
    t_k = t + c * dt
    y_k = np.array([f(t_k[j], y + dt * np.dot(a[j], k)) for j in range(n)])
    return (k - y_k).flatten()

def radau_ii_a_step(y, t, dt, f, a,b,c,a_,b_,c_,try_t_max=100,tol=1e-3,t_step_min=1e-3):
    n = len(b)
    n_ = len(b_)
    dim_y = len(y)
    for i in range(try_t_max): 
        args=(a,c,n,dim_y,f,t,y,dt)
        k_guess = np.zeros(n * dim_y)
        k = fsolve(equations, k_guess, args=args)
        k = k.reshape((n, dim_y))
        dy = dt * np.dot(b, k)
        
        args = (a_, c_,n_,dim_y,f,t,y,dt)
        k_guess = np.zeros(n_ * dim_y)
        k = fsolve(equations, k_guess, args=args)
        k = k.reshape((n_, dim_y))
        dy_ = dt * np.dot(b_, k)
        
        dy_diff= np.max(abs(dy - dy_))
        # Check if the error is within the tolerance, if so, break the loop
        if dy_diff <= tol:
            accepted = True
            break
        else:
            # If the error is too large, reduce dt and try again
            dt_new = dt * (tol/dy_diff)**(1/4)
            # Limit how much dt can change in one iteration
            # to prevent instability or very slow convergence
            factor = 0.5  # Example factor, this can be adjusted as needed
            dt_new = max(dt_new, dt * factor)
            dt_new = min(dt_new, dt / factor)
            dt = dt_new
    
    if not accepted:
        # If we exit the loop without an accepted solution, there is an issue
        raise RuntimeError(f"Could not find an accepted solution within {try_t_max} iterations")
    
        
    return y+dy, dt

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
    a_ = np.array([[ 0.41666667, -0.08333333],
                   [ 0.75      ,  0.25      ]]    
                  )
    b_ = np.array([0.75, 0.25])
    c_ = np.array([0.33333333, 1.0])
    
    y0 = np.array([0.0,0.5,0.4,0.6,1.0])
    t=np.arange(0, 1, 0.1, dtype=float)
    dt = 0.1 
    
    y = np.zeros((len(y0),len(t)))
    y[:,0] = y0
    
    t_tem = np.min(t)
    i_tem = 1
    y_tem = y0
    while True:
        if t_tem <= np.max(t):
            y_tem_old = y_tem
            y_tem, dt = radau_ii_a_step(y_tem_old, t_tem, dt, func, a,b,c,a_,b_,c_)
            t_tem += dt
            if t_tem >= t[i_tem]:
                y[:,i_tem] = y_tem
                dt = t[i_tem] - t_tem
                i_tem += 1
        else:
            break
        
    RES = integrate.solve_ivp(func,
                              [0,max(t)], 
                              y0,t_eval=t,
                              method='Radau',first_step=0.1,rtol=1e-1)
    # Reshape and save result to N and t_vec
    y_ivp = RES.y