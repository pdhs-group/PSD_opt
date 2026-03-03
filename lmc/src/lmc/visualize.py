# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:17:36 2025

@author: px2030
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Sequence
from .func_jit import to_old_G_layout_jit

def _broken_loc_to_big_coords(l: Tuple[int,int,int]) -> Tuple[int,int]:
    axis, i, j = l
    if axis == 0: return (2*i+1, 2*(j+1))
    return (2*(i+1), 2*j+1)

class Plotter:
    def plot_compact(self, M: np.ndarray, Hbond: np.ndarray, Vbond: np.ndarray,
                     labels: Optional[np.ndarray] = None,
                     crack_paths: Optional[Sequence] = None,
                     title: Optional[str] = None, mode: str = 'materials'):
        G = to_old_G_layout_jit(M, Hbond, Vbond)
        Hbig, Wbig = G.shape
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        if mode == 'materials':
            base = np.full_like(G, np.nan, dtype=float)
            mask_mat = (G == 1) | (G == 2)
            base[mask_mat] = G[mask_mat]
            ax.imshow(base, origin='lower', interpolation='nearest')
        else:
            ax.imshow(np.full_like(G, np.nan, dtype=float), origin='lower', interpolation='nearest')
        by, bx = np.where((G == 11) | (G == 12) | (G == 22))
        ax.scatter(bx, by, s=8)
        if crack_paths:
            # 判定是否是“分组裂缝”
            is_grouped = isinstance(crack_paths[0], (list, tuple)) and \
                         len(crack_paths[0]) > 0 and isinstance(crack_paths[0][0], (list, tuple)) and \
                         len(crack_paths[0][0]) > 0 and isinstance(crack_paths[0][0][0], (tuple, list))
            if not is_grouped:
                # 兼容旧格式：把每条裂缝当作一个“组”
                crack_groups = [[path] for path in crack_paths]
            else:
                crack_groups = crack_paths
    
            cmap = plt.colormaps.get_cmap('tab10')
            for gidx, group in enumerate(crack_groups):
                color = cmap(gidx % 10)
                for path in group:
                    if not path:
                        continue
                    ys, xs = [], []
                    for (axis, i, j) in path:
                        yy, xx = _broken_loc_to_big_coords((axis, i, j))
                        ys.append(yy); xs.append(xx)
                    ax.scatter(xs, ys, s=14, marker='x', color=color, alpha=0.9)
    
        if labels is not None and mode == 'fragments':
            lab_big = np.full_like(G, -1)
            lab_big[1::2, 1::2] = labels
            show = np.ma.masked_where(lab_big < 0, lab_big)
            K = int(np.max(labels) + 1) if labels.size > 0 else 0
            if K > 0:
                colors = plt.colormaps.get_cmap('nipy_spectral').resampled(K)
                ax.imshow(show, origin='lower', interpolation='nearest', alpha=0.9, cmap=colors)
    
        ax.set_xlim(-0.5, Wbig - 0.5)
        ax.set_ylim(-0.5, Hbig - 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title or ('Materials view' if mode == 'materials' else 'Fragments view'))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        return ax, fig

    def plot_F(self, F: np.ndarray):
        from matplotlib.colors import LogNorm
        import matplotlib.pyplot as plt
        fig1, ax1 = plt.subplots(figsize=(6,5))
        x1, y1 = F[:,1], F[:,2]
        H1 = ax1.hist2d(x1, y1, bins=20, norm=LogNorm(vmin=1, vmax=max(1, np.sum(np.isfinite(x1))))) 
        ax1.set_xlabel(r"Partial Volume $V_A$")
        ax1.set_ylabel(r"Partial Volume $V_B$")
        plt.colorbar(H1[3], ax=ax1, label='counts')
        ax1.set_title("Fragments: V_A vs V_B")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.hist(F[:,3], bins=100)
        ax2.set_xlabel("Fracture Energy (a.u.)")
        ax2.set_ylabel("Counts")
        ax2.set_title("Fracture energy distribution")
        fig3, ax3 = plt.subplots(figsize=(6,5))
        x3, y3 = F[:,0], F[:,3]
        H3 = ax3.hist2d(x3, y3, bins=20, norm=LogNorm(vmin=1, vmax=max(1, np.sum(np.isfinite(x3))))) 
        ax3.set_xlabel(r"Fragment Size $V$")
        ax3.set_ylabel("Fracture Energy (a.u.)")
        plt.colorbar(H3[3], ax=ax3, label='counts')
        ax3.set_title("Size vs Energy")
        fig4, ax4 = plt.subplots(figsize=(6,4))
        ax4.hist(F[:,0], bins=100)
        ax4.set_xlabel(r"Fragment Size $V$")
        ax4.set_ylabel("Counts")
        ax4.set_title("Fragment size distribution")
        plt.tight_layout()
        return ax1, ax2, ax3, ax4

    def plot_compact_paper(
        self,
        M: np.ndarray,
        Hbond: np.ndarray,
        Vbond: np.ndarray,
        labels: Optional[np.ndarray] = None,
        crack_paths: Optional[Sequence] = None,
        title: Optional[str] = None,
        show_materials: bool = True,
        show_fragments: bool = True,
        show_cracks: bool = True,
        figsize: Tuple[float, float] = (7.0, 7.0),
    ):
        """
        Paper-style visualization of a fractured aggregate on the cell grid.

        - 坐标轴风格尽量贴近 plot_materials_grid（以单元格为坐标）。
        - 材料背景：柔和浅色 (A/B)。
        - 碎片：用 contourf 将每个碎片区域填为一种半透明颜色。
        - 裂缝：细线表示（仍使用 crack_paths 和 _broken_loc_to_big_coords）。

        Parameters
        ----------
        M, Hbond, Vbond : np.ndarray
            Lattice material / bond representation from LMC.
            M 形状 (H, W)，0=empty, 1/2=materials.
        labels : np.ndarray, optional
            Fragment labels on the original cell grid (H, W)，-1 表示未占用。
        crack_paths : sequence, optional
            Crack path description，格式同 plot_compact。
        title : str, optional
            Figure title.
        show_materials : bool
            是否显示材料背景。
        show_fragments : bool
            是否叠加碎片填色。
        show_cracks : bool
            是否绘制裂缝路径。
        figsize : (float, float)
            Figure size，默认 (7,7)。

        Returns
        -------
        ax, fig : Matplotlib axis and figure.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        H, W = M.shape  # 单元格坐标尺寸

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # ---------- 1. 材料背景：柔和浅色 ----------
        if show_materials:
            mat = np.full((H, W), np.nan, dtype=float)
            mask_A = (M == 1)
            mask_B = (M == 2)
            mat[mask_A] = 0.0
            mat[mask_B] = 1.0

            cmap_mat = ListedColormap([
                (0.85, 0.85, 0.95),  # 材料 A：浅蓝灰
                (0.95, 0.85, 0.85),  # 材料 B：浅红灰
            ])

            ax.imshow(
                mat,
                origin='lower',
                interpolation='nearest',
                cmap=cmap_mat,
                alpha=0.9,
                extent=(-0.5, W - 0.5, -0.5, H - 0.5),
            )
        else:
            ax.imshow(
                np.full((H, W), np.nan),
                origin='lower',
                interpolation='nearest',
                extent=(-0.5, W - 0.5, -0.5, H - 0.5),
            )

        # ---------- 2. 碎片：类似 contour plots 的区域填色 ----------
        if labels is not None and show_fragments:
            # labels: (H, W), -1 表示无碎片
            lab = np.array(labels, copy=True)
            mask = (lab < 0)
            if not np.all(mask):
                lab = np.ma.masked_where(mask, lab)
                K = int(lab.max() + 1)
                if K > 0:
                    levels = np.arange(-0.5, K + 0.5, 1.0)
                    cmap_frag = plt.colormaps.get_cmap('Pastel1').resampled(K)
                    x = np.arange(W)
                    y = np.arange(H)
                    X, Y = np.meshgrid(x, y)  # 注意：X 对应列，Y 对应行
                    # contourf 会将每个整数 label 区域填为一个颜色
                    ax.contourf(
                        X,
                        Y,
                        lab,
                        levels=levels,
                        cmap=cmap_frag,
                        alpha=0.7,
                        antialiased=True,
                        extend='neither',
                    )

        # ---------- 3. 裂缝：细线，坐标从“大格子”缩放回单元格 ----------
        if crack_paths and show_cracks:
            # 判定是否是“分组裂缝”
            try:
                is_grouped = (
                    isinstance(crack_paths[0], (list, tuple)) and
                    len(crack_paths[0]) > 0 and
                    isinstance(crack_paths[0][0], (list, tuple)) and
                    len(crack_paths[0][0]) > 0 and
                    isinstance(crack_paths[0][0][0], (tuple, list))
                )
            except Exception:
                is_grouped = False

            if not is_grouped:
                # 兼容旧格式：把每条裂缝当作一个“组”
                crack_groups = [[path] for path in crack_paths]
            else:
                crack_groups = crack_paths

            cmap_crack = plt.colormaps.get_cmap('tab10')

            for gidx, group in enumerate(crack_groups):
                color = cmap_crack(gidx % 10)
                for path in group:
                    if not path:
                        continue
                    ys_big, xs_big = [], []
                    for (axis, i, j) in path:
                        yy_big, xx_big = _broken_loc_to_big_coords((axis, i, j))
                        ys_big.append(yy_big)
                        xs_big.append(xx_big)
                    # 大格子坐标 -> 单元格坐标（近似除以2即可）
                    xs = [x / 2.0 for x in xs_big]
                    ys = [y / 2.0 for y in ys_big]
                    ax.plot(
                        xs,
                        ys,
                        '-',
                        color=color,
                        linewidth=1.0,
                        alpha=0.9,
                    )

        # ---------- 4. 坐标轴、网格线、标题 ----------
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("2D Lattice aggregate fragments (paper view)")

        # 与 plot_materials_grid 一致的细网格设置
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', linewidth=0.3, alpha=0.3)

        # 统一字号（可根据需要调整）
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        return ax, fig
    
    def plot_fragments_simple(
        self,
        M: np.ndarray,
        labels: np.ndarray,
        title: str = "Fragments",
        figsize: Tuple[float, float] = (7, 7),
    ):
        """
        Simple, paper-friendly fragment visualization:
        - No cracks
        - No bonds
        - Just plot each fragment as a uniformly colored region
        - Axes, ticks, grids follow the style of plot_materials_grid
        """
        H, W = labels.shape
    
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
        # -------- 1. 构造 fragment 区域 mask --------
        lab = np.array(labels, copy=True)
        K = int(lab.max() + 1) if lab.size > 0 else 0
    
        # -------- 2. 构造柔和颜色 Colormap（不刺眼） --------
        if K > 0:
            cmap_frag = plt.colormaps.get_cmap("tab20").resampled(K)
            show = np.ma.masked_where(lab < 0, lab)
            ax.imshow(
                show,
                origin="lower",
                interpolation="nearest",
                # alpha=0.85,
                cmap=cmap_frag,
            )
        else:
            # 没有碎片（应该不会），但保持格式完整
            ax.imshow(
                np.zeros_like(labels),
                origin="lower",
                interpolation="nearest",
                alpha=0.2,
                cmap="gray",
            )
    
        # -------- 3. 坐标轴设置（与 plot_materials_grid 对齐） --------
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
    
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
        # minor ticks 网格（每 1 单元格）
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which="minor", linewidth=0.3, alpha=0.3)
    
        # 字号控制（与 materials_grid 类似）
        ax.tick_params(labelsize=10)
    
        plt.tight_layout()
        return ax, fig

