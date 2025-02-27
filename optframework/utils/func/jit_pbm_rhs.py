import numpy as np
import math
from numba import jit
import optframework.utils.func.jit_pbm_qmom as qmom
import optframework.utils.func.jit_kernel_agg as kernel_agg
import optframework.utils.func.jit_kernel_break as kernel_break
import optframework.utils.func.jit_pbm_chyqmom as chyqmom
from scipy.optimize import root
from scipy.optimize import least_squares
from numba import jit

def filter_negative_nodes(xi, wi, threshold=1e-7):
    """
    Filter the nodes (xi) and weights (wi) based on the condition:
    - If xi[i] < 0 and wi[i] < threshold, remove xi[i] and wi[i].
    - If xi[i] < 0 but wi[i] >= threshold, raise an error.

    Parameters:
        xi (array-like): Array of nodes.
        wi (array-like): Array of weights corresponding to the nodes.
        threshold (float): Threshold for weights below which the corresponding nodes are removed.

    Returns:
        tuple: Filtered (xi, wi) arrays.
    """
    xi = np.array(xi)
    wi = np.array(wi)

    if len(xi) != len(wi):
        raise ValueError("xi and wi must have the same length.")

    # Indices where xi < 0
    negative_indices = np.where(xi < 0)[0]

    # Check each negative xi
    for idx in negative_indices:
        if wi[idx] >= threshold:
            raise ValueError(
                f"Node xi[{idx}] = {xi[idx]} is negative, but its weight wi[{idx}] = {wi[idx]} exceeds the threshold {threshold}."
            )

    # Remove elements where xi < 0 and wi < threshold
    valid_indices = [i for i in range(len(xi)) if xi[i] >= 0 or wi[i] >= threshold]

    # Filter xi and wi
    xi_filtered = xi[valid_indices]
    wi_filtered = wi[valid_indices]

    return xi_filtered, wi_filtered

def hyqmom_newton_correction(xi, wi, moments, method="lm"):
    """
    通过 Newton 迭代修正 QMOM 的 xi, wi，使其满足矩方程。
    
    参数：
    - xi: 节点 (可能包含负值)
    - wi: 权重
    - moments: 目标矩 [M0, M1, M2, ...]
    - max_iter: Newton 最大迭代次数
    - tol: 终止条件 ‖R_k‖ ≈ 0
    
    返回：
    - 修正后的 xi, wi
    """
    SMALL = 1e-8
    # 处理负数 xi
    negative_idx = xi < 0
    if np.any(negative_idx):
        # print(f"修正 {np.sum(negative_idx)} 个负数节点")
        xi[negative_idx] = SMALL  # 设为极小值
    else:
        return xi, wi
    
    xi_free_idx = ~negative_idx  # 需要优化的 xi
    xi_free = xi[xi_free_idx]
    wi_free = wi
    # Newton 迭代过程
    def residuals(variables):
        """
        计算 R_k(x) 及 Jacobian 矩阵
        """
        xi_new = xi.copy()
        xi_new[xi_free_idx] = variables[:len(xi_free)]
        wi_new = variables[len(xi_free):]
    
        # return np.array([np.sum(wi_new * (xi_new ** k)) - moments[k] for k in range(len(moments))])
        return np.array([np.sum(wi_new * (xi_new ** k)) - moments[k] for k in range(3)])
    # 初始变量 [xi_free, wi_free]
    x0 = np.concatenate([xi_free, wi_free])
    
    # Newton 迭代
    # sol = root(residuals, x0, method=method)
    sol = least_squares(residuals, x0, method="trf", ftol=1e-3)
    if not sol.success:
        raise ValueError(f"优化未收敛: {sol.message}")

    xi[xi_free_idx] = sol.x[:len(xi_free)]
    wi = sol.x[len(xi_free):]

    return xi, wi
    
@jit(nopython=True)
def get_dMdt_1d(t, moments, x_max, GQMOM, GQMOM_method, 
                moments_norm_factor, n_add, nu, 
                COLEVAL, CORR_BETA, G, alpha_prim, EFFEVAL, 
                SIZEEVAL, V_unit, X_SEL, Y_SEL, 
                V1_mean, pl_P1, pl_P2, BREAKRVAL, 
                v, q, BREAKFVAL, type_flag):
    dMdt = np.zeros(moments.shape)
    dMdt_norm = np.zeros(moments.shape)
    # Check if moments are unrealizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")
    
    m = len(moments)
    n = m // 2  # Number of xi based on available moments
    adaptive = False
    use_central=False
    if not GQMOM:
        xi, wi, n = qmom.calc_qmom_nodes_weights(moments, n, adaptive, use_central)
    else:
        xi, wi, n = qmom.calc_gqmom_nodes_weights(moments, n, n_add, GQMOM_method, nu,adaptive, use_central)
        # xi, wi = filter_negative_nodes(xi, wi)
        n += n_add

    # if n == 2:
    #     xi, wi = chyqmom.hyqmom2(moments)
    # elif n == 3:
    #     xi, wi = chyqmom.hyqmom3(moments)
    # else:
    #     raise ValueError("Unsupport moment order")
    # # xi, wi = filter_negative_nodes(xi, wi, threshold=1e7)
    # # n = len(xi)
    # xi, wi = hyqmom_newton_correction(xi,wi,moments)
    
    if np.any(xi<0):
        print(f"t = {t}")
        print(f"xi = {xi}, wi = {wi}")  
    # print(moments)
    ## In order to calculate the contribution of each kernel,
    ## the coordinates need to be restored to their original values.
    xi *= x_max
    
    ## 因为PBE中的计算考虑了零点，所以这里对于PBM使用的时候坐标会发生一定偏移
    V = np.zeros(n+1)
    R = np.zeros(n+1)
    V[1:] = xi
    R[1:] = (V[1:]*3/(4*math.pi))**(1/3)
    # R[1:] = xi
    # V[1:] = 4*math.pi*R[1:]**3/4
    
    if type_flag == "agglomeration" or type_flag == "mix":
        F_M_tem = kernel_agg.calc_F_M_1D(n+2, COLEVAL, CORR_BETA, G, R, 
                                     alpha_prim, EFFEVAL, SIZEEVAL, X_SEL, Y_SEL)
        F_M = F_M_tem[1:,1:] / V_unit
        
    if type_flag == "breakage" or type_flag == "mix":
        B_F_intxk = np.zeros((m, n))
        B_R = kernel_break.breakage_rate_1d(V[1:], V1_mean, pl_P1, pl_P2, G, BREAKRVAL)

        xs1 = np.array([-9.681602395076260859e-01,
                    -8.360311073266357695e-01,
                    -6.133714327005903577e-01,
                    -3.242534234038089158e-01,
                    0.000000000000000000e+00,
                    3.242534234038089158e-01,
                    6.133714327005903577e-01,
                    8.360311073266357695e-01,
                    9.681602395076260859e-01,])
        ws1 = np.array([8.127438836157471758e-02,
                    1.806481606948571184e-01,
                    2.606106964029356599e-01,
                    3.123470770400028074e-01,
                    3.302393550012596712e-01,
                    3.123470770400028074e-01,
                    2.606106964029356599e-01,
                    1.806481606948571184e-01,
                    8.127438836157471758e-02])
        for k in range(m):
            for i in range(n):
                argsk = (V[i+1],v,q,BREAKFVAL,k)
                func = kernel_break.breakage_func_1d_xk
                B_F_intxk[k, i] = kernel_break.gauss_legendre(func, 0.0, V[i+1], xs1, ws1, args=argsk)
    
    for k in range(m):
        dMdt_agg_ij = 0.0
        dMdt_break_i = 0.0
        for i in range(n):
            if type_flag == "agglomeration" or type_flag == "mix":
                for j in range(n):
                    dMdt_agg_ij += 0.5 * wi[i]*wi[j]*F_M[i, j]*((xi[i]+xi[j])**k-xi[i]**k-xi[j]**k)
            if type_flag == "breakage" or type_flag == "mix":
                dMdt_break_i += wi[i] * B_R[i] * B_F_intxk[k, i] - wi[i] * xi[i]**k * B_R[i]
        dMdt[k] = dMdt_agg_ij + dMdt_break_i
    dMdt_norm = dMdt / moments_norm_factor
    
    return dMdt_norm

@jit(nopython=True)
def get_dMdt_2d(t, moments, n, indices, COLEVAL, CORR_BETA, G, alpha_prim, EFFEVAL, 
                SIZEEVAL, V_unit, X_SEL, Y_SEL, 
                V1_mean, V3_mean, pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, 
                v, q, BREAKFVAL, type_flag):
    # eta = 1e-2
    n0 = n
    dMdt = np.zeros(moments.shape)
    # Check if moments are unrealizable
    if moments[0] <= 0:
        raise ValueError("Wheeler: Moments are NOT realizable (moment[0] <= 0.0).")

    xi, wi, n = qmom.calc_cqmom_2d(moments, n, indices, use_central=True)
    
    # for Debug
    # moments_cqmom = np.zeros_like(moments)
    # for idx, _ in enumerate(moments_cqmom):
    #     indice = indices[idx]
    #     moments_cqmom[idx] = chyqmom.quadrature_2d(wi, xi, indice)
    # print(np.mean(abs(moments_cqmom-moments)/moments))
    
    if n0 > n:
        print(f"Warning: At t = {t}, the moments are NOT realizable, abscissas reduced to {n}.")
    V = np.ones((n, n))
    V1 = xi[0,:]
    V3 = xi[1,:]
    V_flat = np.ones(n*n)
    
    ## 因为PBE中的计算考虑了零点，所以这里对于PBM使用的时候坐标会发生一定偏移
    R = np.ones((n+1, n+1))
    X1 = np.ones((n+1, n+1))
    X3 = np.ones((n+1, n+1))
    
    for i in range(n):
        for j in range(n):
            V[i,j] = V1[i*n+j] + V3[i*n+j]
            V_flat[i*n+j] = V[i,j]
            X1[i+1,j+1] = V1[i*n+j] / V[i,j]
            X3[i+1,j+1] = V3[i*n+j] / V[i,j]
    
    if np.any(V<0):
        print(f"t = {t}")
        print(f"xi = {xi}, wi = {wi}")   
    # print(moments)
    # print(t)
    # print(xi[:,0])
    
    R[1:, 1:] = (V*3/(4*math.pi))**(1/3)
    
    if type_flag == "agglomeration" or type_flag == "mix":
        F_M_tem = kernel_agg.calc_F_M_2D(n+2, COLEVAL, CORR_BETA, G, R, X1, X3,EFFEVAL, 
                                     alpha_prim, SIZEEVAL, X_SEL, Y_SEL)
        F_M = F_M_tem[1:,1:,1:,1:] / V_unit
        
    if type_flag == "breakage" or type_flag == "mix":
        ## kB_F_intxkxl is not actually fully populated. 
        ## Only the positions corresponding to the indices need to be computed, 
        ## which correspond to the second-order moments used by the system.
        B_F_intxk = np.zeros((2*n, n, 2*n, n))
        B_R_flat = kernel_break.breakage_rate_2d_flat(V_flat, V1, V3, V1_mean, V3_mean, G,
                                                     pl_P1, pl_P2, pl_P3, pl_P4, BREAKRVAL, BREAKFVAL)
        B_R = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                # if V1[i] < eta or V3[j] < eta:
                #     continue
                B_R[i,j] = B_R_flat[i*n+j]

        xs1 = np.array([-9.681602395076260859e-01,
                    -8.360311073266357695e-01,
                    -6.133714327005903577e-01,
                    -3.242534234038089158e-01,
                    0.000000000000000000e+00,
                    3.242534234038089158e-01,
                    6.133714327005903577e-01,
                    8.360311073266357695e-01,
                    9.681602395076260859e-01,])
        ws1 = np.array([8.127438836157471758e-02,
                    1.806481606948571184e-01,
                    2.606106964029356599e-01,
                    3.123470770400028074e-01,
                    3.302393550012596712e-01,
                    3.123470770400028074e-01,
                    2.606106964029356599e-01,
                    1.806481606948571184e-01,
                    8.127438836157471758e-02])
        xs3 = xs1
        ws3 = ws1
        for idx, _ in enumerate(dMdt):
            k = indices[idx,0]
            l = indices[idx,1]
            for i in range(n):
                for j in range(n):
                    # if V1[i*n+j] < eta or V3[i*n+j] < eta:
                    #     continue
                    ## If a calculation error occurs, it may be necessary to check whether 
                    ## V1 or V3 is equal to zero and apply a different integration strategy accordingly.
                    argsk = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,k,l)
                    func = kernel_break.breakage_func_2d_x1kx3l
                    B_F_intxk[k, i, l, j] = kernel_break.dblgauss_legendre(func, 0.0, V1[i*n+j], 0.0, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk)
                    
                    # argsk = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,1,eta)
                    # func1 = kernel_break.breakage_func_2d_x1k_trunc
                    # func2 = kernel_break.breakage_func_2d_x3k_trunc
                    # norm_fac1 = kernel_break.dblgauss_legendre(func1, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk)
                    # norm_fac2 = kernel_break.dblgauss_legendre(func2, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk)
                    # argsk_trunc = (V1[i*n+j],V3[i*n+j],v,q,BREAKFVAL,k,l,eta)
                    # func_norm = kernel_break.breakage_func_2d_trunc
                    # B_F_intxk_trunk = kernel_break.dblgauss_legendre(func_norm, eta, V1[i*n+j], eta, V3[i*n+j], xs1, ws1, xs3,ws3,args=argsk_trunc)
                    # B_F_intxk[k, i, l, j] = B_F_intxk_trunk / ((norm_fac1 + norm_fac2) / V[i,j])

    for idx, _ in enumerate(dMdt):
        k = indices[idx,0]
        l = indices[idx,1]
        dMdt_agg_ijab = 0.0
        dMdt_break_ij = 0.0
        for i in range(n):
            for j in range(n):
                if type_flag == "agglomeration" or type_flag == "mix":
                    for a in range(n):
                        for b in range(n):
                            dMdt_agg_ijab += 0.5 * wi[i*n+j]*wi[a*n+b]*F_M[i,j,a,b]*(
                                (xi[0,i*n+j]+xi[0,a*n+b])**k*(xi[1,i*n+j]+xi[1,a*n+b])**l
                                -xi[0,i*n+j]**k*xi[1,i*n+j]**l
                                -xi[0,a*n+b]**k*xi[1,a*n+b]**l)
                            
                            # print((F_M[i,j,a,b]
                            # -(xi[0,i*n+j]+xi[0,a*n+b]+xi[1,i*n+j]+xi[1,a*n+b])*CORR_BETA)/F_M[i,j,a,b])
                if type_flag == "breakage" or type_flag == "mix":
                    dMdt_break_ij += (wi[i*n+j] * B_R[i,j] * B_F_intxk[k,i,l,j] - wi[i*n+j]
                                      * xi[0,i*n+j]**k * xi[1,i*n+j]**l * B_R[i,j])
                    
        dMdt[idx] = dMdt_agg_ijab + dMdt_break_ij
    
    return dMdt

        