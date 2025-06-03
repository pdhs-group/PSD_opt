# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:06:08 2025

@author: px2030
"""
import os
import originpro as op
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize   import minimize_scalar
from optframework.utils.general_scripts.convert_exp_data_Batch import generate_nonuniform_grid, interpolate_Qx
import matplotlib.pyplot as plt

def process_origin_sheets_and_save():
    # op.set_show(True)
    # op.new()

    # op.open(r'C:\Users\px2030\Code\Ergebnisse\Simon_Messung\Auswertung CPS.opju')
    
    targets = {
        'Stufe1': '[Stufe1]Sheet1',
        'Stufe4': '[Stufe4]Sheet1',
        'Stufe3X': ['[Stufe3X1]Sheet1', '[Stufe3X2]Sheet1', '[Stufe3X3]Sheet1']
    }
    
    time_labels = ["00:00:00", "00:02:00", "00:04:00", "00:09:00", "00:14:00", "00:29:00"]
    def read_and_format(sheet_path):
        df = op.find_sheet('w', sheet_path).to_df()
        df = df.iloc[:, :7] 
        df.columns = ["Circular Equivalent Diameter"] + time_labels
        df.set_index("Circular Equivalent Diameter", inplace=True)
        return df
    
    df1 = read_and_format(targets['Stufe1'])
    save_to_excel(df1, "Q3", "Stufe1.xlsx")
    
    df4 = read_and_format(targets['Stufe4'])
    save_to_excel(df4, "Q3", "Stufe4.xlsx")
    
    dfs_3x = [read_and_format(path) for path in targets['Stufe3X']]
    df3x_avg = pd.concat(dfs_3x).groupby(level=0).mean()
    save_to_excel(df3x_avg, "Q3", "Stufe3.xlsx")
    
    return df1, df4, df3x_avg

def save_to_excel(df, sheet_name, file_path):
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
        
def recalc_qx_and_Qx(file_path):
    Q_df_ori = pd.read_excel(file_path, sheet_name="Q3", index_col=0)
    labels = Q_df_ori.columns
    index_name = Q_df_ori.index.name
    x_ori = Q_df_ori.index.to_numpy()
    Q_ori = Q_df_ori.to_numpy() / 100

    x = generate_nonuniform_grid(x_min=0.073, x_max=2, num_points=30, gamma=1.42)
    Q = np.zeros((len(x), len(labels)))
    q = np.zeros_like(Q)
    x_volume_mean = np.zeros(len(labels))

    for i, label in enumerate(labels):
        Q[1:, i], _ = interpolate_Qx(
            x_vals=x_ori,
            Q_vals=Q_ori[:, i],
            x_target=x[1:],
            method='int1d',
            fraction=1.0
        )

        dQ = np.diff(Q[:, i])
        dx = np.diff(x)
        q_tmp = np.concatenate([[0], dQ / dx])
        q_tmp[q_tmp < 0] = 0.0
        area = np.sum(q_tmp[1:] * dx)
        q[:, i] = q_tmp / area if area != 0 else q_tmp

        dQ_ori = np.diff(Q_ori[:, i])
        x_ori_bar = x_ori[1:]
        x_volume_mean[i] = (np.sum(dQ_ori / x_ori_bar**3))**(-1/3)

    Q_df = pd.DataFrame(Q[1:, :], index=x[1:], columns=labels)
    q_df = pd.DataFrame(q[1:, :], index=x[1:], columns=labels)
    for df, name in zip([Q_df, q_df], ["Q3_int", "qq3_int"]):
        df.index.name = index_name
        save_to_excel(df, name, file_path)

    return Q_df, q_df, x_volume_mean, x_ori, Q_ori, x[1:]
    
def plot_Q_comparison(x_ori, Q_ori, x_new, Q_interp, labels, title="Q Comparison"):
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.plot(x_ori, Q_ori[:, i], label=f"{label} ori", linestyle='-')
        plt.plot(x_new, Q_interp[:, i], label=f"{label} interp", linestyle='None', marker='x')

    plt.xlabel("Circular Equivalent Diameter")
    plt.ylabel("Q3")
    plt.title(title)
    plt.grid(True)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.show()
    
def calc_n_for_E(x_volume_mean, E_flag):
    if E_flag == "Handbuch":
        E_datas = [7, 16.5, 27.5]
        # n = 0.646
    elif E_flag == "Experimentell":
        E_datas = [12.45, 31.48, 35.77]
        # n = 0.964
    else:
        raise ValueError(f"Unknown E_flag: {E_flag}")
        
    E_values = np.array(E_datas, dtype=float)
    times = np.array([1, 3, 5, 10, 15, 30], dtype=float)     
    
    # To prevent extrapolation, the group that is "compressed" the most should be selected as the reference curve, 
    # i.e., the one with the largest E_value.
    ref_idx = -1
    t_ref = times
    D_ref = x_volume_mean[ref_idx]
    E_ref = E_values[ref_idx]
    interp_ref = interp1d(t_ref, D_ref, kind='linear', fill_value='extrapolate')
    # The only difference between the curves is that the time axis is horizontally stretched/compressed.
    def sse(n):
        tot = 0.0
        # Construct the residual sum of squares function for all points (after stretching/compression) across all groups.
        for i in range(len(E_values)):
            if i == ref_idx: continue
            factor = (E_values[i]/E_ref)**n
            t_scaled = times * factor
            D_est = interp_ref(t_scaled)
            tot += np.sum((x_volume_mean[i] - D_est)**2)
        return tot

    res = minimize_scalar(sse, bounds=(-5,5), method='bounded')  # 根据经验给 n 一个区间
    return res.x, res.fun

if __name__ == '__main__':
    # df1, df4, df3x_avg = process_origin_sheets_and_save()
    
    base_path = r"C:\Users\px2030\Code\Ergebnisse\Simon_Messung\Daten"

    df_paths = [
        os.path.join(base_path, "Stufe1.xlsx"),
        os.path.join(base_path, "Stufe3.xlsx"),
        os.path.join(base_path, "Stufe4.xlsx")
    ]
    
    x_volume_mean_list = []
    for path in df_paths:
        Q_df, q_df, x_volume_mean, x_ori, Q_ori, x_new = recalc_qx_and_Qx(path)
        plot_Q_comparison(x_ori, Q_ori, x_new, Q_df.to_numpy(), Q_df.columns, title=f"Q vs Q_interp for {os.path.basename(path)}")
        x_volume_mean_list.append(x_volume_mean)
        
        # x_ori *= 1e-6 # convert the unit from um to m
        # base_name = os.path.splitext(os.path.basename(path))[0] 
        # npy_name = f"{base_name}_int.npy"
        # dist_path = os.path.join(base_path, npy_name)
        # dict_Qx = {'Q_PSD':Q_ori[:, 0],'x_PSD':x_ori, 'r0_001':x_ori[0], 'r0_005':x_ori[1], 'r0_01':x_ori[2]}
        # np.save(dist_path, dict_Qx)

    # n_opt, sse_min = calc_n_for_E(np.array(x_volume_mean_list), 'Handbuch')
    # print(f"Best n = {n_opt:.3f}, SSE = {sse_min:.3e}")
    
    