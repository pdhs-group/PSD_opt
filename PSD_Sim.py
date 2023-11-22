# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:46:06 2023

@author: px2030
"""

from PSD_Exp import read_exp

class write_sim():
    
    def __init__(self, x_uni, q3, t_vec, save_path):
        self.x_uni = x_uni
        self.q3 = q3
        self.t_vec = t_vec
        
    def save_to_excel(self, save_path):

        df = pd.DataFrame(index=self.x_uni)
        df.index.name = 'Circular Equivalent Diameter'
    
        # 为每个时间点填充数据
        for idt in range(len(self.t_vec)):
            # 调用 return_distribution 函数
            x_uni, q3, _, _, _, _ = self.return_distribution(t=t)
    
            # 将 q3 数据添加到 DataFrame
            df[p.t_vec[idt]] = q3
    
        # 将 DataFrame 保存为 Excel 文件
        df.to_excel(excel_path)