"""Modern plotting helper for publication-style scientific figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


@dataclass
class PlotStyle:
    color: Any
    marker: str
    linestyle: str


class PaperPlotter:
    """Unified plot facade for validation and paper-oriented figures."""

    DEFAULT_PALETTE = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
    DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
    DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]

    def __init__(
        self,
        figure_mode: str = "half",
        width_cm: Optional[float] = None,
        aspect: float = 0.72,
        dpi: float = 300.0,
        font_family: str = "serif",
        font_size: float = 9.0,
        label_size: float = 9.0,
        tick_size: float = 8.0,
        legend_size: float = 8.0,
        line_width: float = 1.4,
        marker_size: float = 5.0,
    ) -> None:
        self.figure_mode = figure_mode
        self.width_cm = width_cm
        self.aspect = aspect
        self.dpi = dpi
        self.font_family = font_family
        self.font_size = font_size
        self.label_size = label_size
        self.tick_size = tick_size
        self.legend_size = legend_size
        self.line_width = line_width
        self.marker_size = marker_size
        self._style_cache: Dict[str, PlotStyle] = {}
        self.apply_style()

    def apply_style(self) -> None:
        plt.rcdefaults()
        plt.rc("mathtext", fontset="cm")
        plt.rc("font", family=self.font_family, size=self.font_size)
        plt.rc("axes", labelsize=self.label_size, titlesize=self.label_size, linewidth=0.6)
        plt.rc("xtick", labelsize=self.tick_size)
        plt.rc("ytick", labelsize=self.tick_size)
        plt.rc("legend", fontsize=self.legend_size, fancybox=True, framealpha=0.9, edgecolor="0.2")
        plt.rcParams["lines.linewidth"] = self.line_width
        plt.rcParams["lines.markersize"] = self.marker_size
        plt.rcParams["axes.axisbelow"] = True
        plt.rcParams["figure.dpi"] = self.dpi
        plt.rcParams["savefig.dpi"] = self.dpi

    def figure(
        self,
        projection: Optional[str] = None,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
    ) -> Tuple[plt.Figure, plt.Axes]:
        figsize = self._figsize(width_scale=width_scale, height_scale=height_scale)
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        if projection is None:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection=projection)
        return fig, ax

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
        sharex: bool = False,
        sharey: bool = False,
        subplot_kw: Optional[dict] = None,
    ) -> Tuple[plt.Figure, Any]:
        figsize = self._figsize(width_scale=width_scale, height_scale=height_scale)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            dpi=self.dpi,
            sharex=sharex,
            sharey=sharey,
            subplot_kw=subplot_kw,
        )
        return fig, axes

    def style_for_key(self, key: str, family: Optional[str] = None) -> PlotStyle:
        cache_key = f"{family or 'default'}::{key}"
        if cache_key not in self._style_cache:
            idx = len(self._style_cache)
            color = self.DEFAULT_PALETTE[idx % len(self.DEFAULT_PALETTE)]
            marker = self.DEFAULT_MARKERS[idx % len(self.DEFAULT_MARKERS)]
            linestyle = self.DEFAULT_LINESTYLES[(idx // len(self.DEFAULT_MARKERS)) % len(self.DEFAULT_LINESTYLES)]
            if family == "analytical":
                color = "black"
                linestyle = "-"
            self._style_cache[cache_key] = PlotStyle(color=color, marker=marker, linestyle=linestyle)
        return self._style_cache[cache_key]

    def plot_line(
        self,
        ax: plt.Axes,
        x,
        y,
        *,
        key: str,
        family: Optional[str] = None,
        label: Optional[str] = None,
        error: Optional[Any] = None,
        markevery: Optional[int] = None,
        alpha: float = 1.0,
    ) -> PlotStyle:
        style = self.style_for_key(key, family=family)
        ax.plot(
            x,
            y,
            label=label,
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            markevery=markevery,
            alpha=alpha,
        )
        if error is not None:
            ax.fill_between(x, y - error, y + error, color=style.color, alpha=0.15)
        return style

    def plot_scatter(
        self,
        ax: plt.Axes,
        x,
        y,
        *,
        key: str,
        family: Optional[str] = None,
        label: Optional[str] = None,
        size: float = 70.0,
    ) -> PlotStyle:
        style = self.style_for_key(key, family=family)
        ax.scatter(x, y, label=label, color=style.color, marker=style.marker, s=size)
        return style

    def plot_wireframe(
        self,
        ax,
        X,
        Y,
        Z,
        *,
        key: str,
        family: Optional[str] = None,
        linewidth: float = 1.0,
        alpha: float = 0.65,
    ) -> PlotStyle:
        style = self.style_for_key(key, family=family)
        ax.plot_wireframe(X, Y, Z, color=style.color, linewidth=linewidth, alpha=alpha)
        return style

    def proxy_handle(self, key: str, family: Optional[str] = None, linewidth: float = 2.0) -> Line2D:
        style = self.style_for_key(key, family=family)
        return Line2D([0], [0], color=style.color, marker=style.marker, linestyle=style.linestyle, lw=linewidth)

    def finalize_axes(
        self,
        ax: plt.Axes,
        *,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        legend: bool = False,
        legend_kwargs: Optional[dict] = None,
        grid: bool = True,
    ) -> None:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if grid:
            ax.grid(True, which="both", alpha=0.25)
        if legend:
            ax.legend(**(legend_kwargs or {}))

    def tighten(self, fig: plt.Figure) -> None:
        fig.tight_layout()

    def show(self) -> None:
        plt.show()

    def _figsize(self, width_scale: float = 1.0, height_scale: float = 1.0) -> Tuple[float, float]:
        if self.width_cm is not None:
            width_cm = self.width_cm
        elif self.figure_mode == "full":
            width_cm = 16.5
        else:
            width_cm = 8.2
        width_in = (width_cm / 2.54) * width_scale
        height_in = width_in * self.aspect * height_scale
        return width_in, height_in
