# -*- coding: utf-8 -*-
"""
Adapter: using MLPEnergyModel to provide breakage rates for MCPBE.

æ ¸å¿ƒæŽ¥å£ï¼š
    - MLPBreakageRateAdapter.compute_rates_full(pbe)  -> np.ndarray, shape (a,)
    - MLPBreakageRateAdapter.compute_rate_single(pbe, i) -> float

å…¶ä¸­ pbe åº”è¯¥æ˜¯ MCPBEBreak æˆ–å…¼å®¹çš„å¯¹è±¡ï¼Œè‡³å°‘è¦æš´éœ²ï¼š
    - pbe.a_tot              : å½“å‰æ´»è·ƒé¢—ç²’æ•° a
    - pbe.dim                : 1 æˆ– 2ï¼ˆç›®å‰æ”¯æŒ 1D/2Dï¼‰
    - pbe.V_flat             : shape (dim, max_a)ï¼Œæœ€åŽä¸€è¡Œä¸ºæ€»ä½“ç§¯ V
      Â· è‹¥ dim == 1: V_flat[-1, :a] ä¸ºé¢—ç²’ä½“ç§¯
      Â· è‹¥ dim == 2: V_flat[0,:a] ä¸º v1, V_flat[1,:a] ä¸º v3, V = v1+v3
    - ï¼ˆå¯é€‰ï¼‰pbe.lmc_gamma, pbe.lmc_NO_FRAG, pbe.lmc_int_bre,
              pbe.lmc_Df, pbe.lmc_MAS
      è‹¥æ²¡æœ‰ï¼Œåˆ™ä½¿ç”¨ adapter åˆå§‹åŒ–æ—¶ä¼ å…¥çš„å‚æ•°æˆ–é»˜è®¤å€¼ã€‚

MLP æ¨¡åž‹å¿…é¡»æ˜¯ MLPEnergyModelï¼ˆæˆ–å…¼å®¹æŽ¥å£ï¼‰ï¼Œå³ï¼š
    - model.predict(X: np.ndarray) -> np.ndarray (logE_need),
      å†…éƒ¨å·²ç»åšå¥½æ ‡å‡†åŒ–ç­‰å¤„ç†ã€‚
"""

from __future__ import annotations

from typing import Optional, Callable, Sequence

import numpy as np
import torch

from breakage_rate_model.base import BaseEnergyModel
from breakage_rate_model.mlp_model import MLPEnergyModel


class MLPBreakageRateAdapter:
    """
    ç”¨è®­ç»ƒå¥½çš„ MLPEnergyModel ç»™ MCPBE æä¾›ç ´ç¢ŽçŽ‡ï¼ˆbreakage rateï¼‰ã€‚

    ä½¿ç”¨æ–¹å¼ï¼ˆç¤ºä¾‹ï¼‰ï¼š

        adapter = MLPBreakageRateAdapter(
            model_path="mlp_energy_model.pkl",
            lambda_E=1.0,          # E_in(V) = lambda_E * V^energy_exp
            energy_exp=1.0,
            # å¯é€‰ï¼šæŒ‡å®š LMC å‚æ•°é»˜è®¤å€¼ï¼ˆè‹¥ pbe ä¸Šæ²¡æœ‰å¯¹åº”å±žæ€§ï¼‰
            gamma=1.0,
            NO_FRAG=4,
            int_bre=0.0,
            Df=2.5,
            MAS=0.0,
        )

        mcpbe.break_rate_adapter = adapter
        mcpbe.use_mlp_break_rate = True

    ç„¶åŽåœ¨ MCPBEBreak._calc_break_rates_full / _break_rate_single ä¸­
    è°ƒç”¨ adapter.compute_rates_full / compute_rate_single å³å¯ã€‚
    """

    def __init__(
        self,
        *,
        model: Optional[MLPEnergyModel] = None,
        model_path: Optional[str] = None,
        # å…¥å°„èƒ½é‡ E_in(V) = lambda_E * V^energy_exp  æˆ– è‡ªå®šä¹‰ energy_in_fn
        lambda_E: float = 1.0,
        energy_exp: float = 1.0,
        energy_in_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        # LMC å‚æ•°çš„é»˜è®¤å€¼ï¼ˆå¦‚æžœ pbe æ²¡æŒ‚å¯¹åº”å±žæ€§ï¼Œå°±ç”¨è¿™äº›ï¼‰
        gamma: float = 1.0,
        NO_FRAG: float = 4.0,
        int_bre: float = 0.0,
        Df: float = 2.5,
        MAS: float = 0.0,
        # ç ´ç¢ŽçŽ‡è£å‰ªï¼ˆé¿å…æžç«¯å€¼/æ•°å€¼ä¸ç¨³å®šï¼‰
        rate_min: float = 0.0,
        rate_max: Optional[float] = None,
        eps_E: float = 1e-30,
        A0_run: float = 1.0,
    ):
        """
        å‚æ•°
        ----
        model:
            å·²ç»åœ¨å¤–éƒ¨æž„å»ºå¹¶è®­ç»ƒå¥½çš„ MLPEnergyModel å®žä¾‹ï¼ˆå¯é€‰ï¼‰ã€‚
        model_path:
            è‹¥ model ä¸º Noneï¼Œåˆ™é€šè¿‡ BaseEnergyModel.load(model_path) è½½å…¥ã€‚
        lambda_E, energy_exp:
            é»˜è®¤å…¥å°„èƒ½é‡æ¨¡åž‹ï¼šE_in(V) = lambda_E * V^energy_expã€‚
            è‹¥æä¾›äº† energy_in_fnï¼Œåˆ™å¿½ç•¥è¿™ä¸¤ä¸ªã€‚
        energy_in_fn:
            è‡ªå®šä¹‰èƒ½é‡è¾“å…¥å‡½æ•°ï¼Œç­¾åï¼šenergy_in_fn(V: np.ndarray) -> np.ndarrayã€‚
        gamma, NO_FRAG, int_bre, Df, MAS:
            LMC å‚æ•°çš„é»˜è®¤å€¼ï¼Œç”¨äºŽä»Ž pbe ä¸­å–ä¸åˆ°å¯¹åº”å±žæ€§æ—¶çš„ fallbackã€‚
        rate_min, rate_max:
            å¯¹æœ€ç»ˆé€ŸçŽ‡åš clipï¼›rate_max ä¸º None æ—¶åªåšä¸‹æˆªæ–­ã€‚
        eps_E:
            é¿å… E_need ä¸º 0 æ—¶çš„é™¤é›¶ã€‚
        """
        if model is None:
            if model_path is None:
                raise ValueError("Either 'model' or 'model_path' must be provided.")
            loaded = BaseEnergyModel.load(model_path, device="cpu")
            if not isinstance(loaded, MLPEnergyModel):
                raise TypeError(
                    f"Loaded model from {model_path} is not MLPEnergyModel (got {type(loaded)})"
                )
            # åŒä¿é™©ï¼šç¡®ä¿ device å’Œ _net éƒ½åœ¨ CPU
            loaded.device = torch.device("cpu")
            if getattr(loaded, "_net", None) is not None:
                loaded._net.to("cpu")
            self.model: MLPEnergyModel = loaded
        else:
            # å¤–éƒ¨ä¼ è¿›æ¥çš„æ¨¡åž‹ä¹Ÿç»Ÿä¸€è¿ç§»åˆ° CPU
            model.device = torch.device("cpu")
            if getattr(model, "_net", None) is not None:
                model._net.to("cpu")
            self.model = model

        self.lambda_E = float(lambda_E)
        self.energy_exp = float(energy_exp)
        self.energy_in_fn = energy_in_fn

        # LMC é»˜è®¤å‚æ•°
        self.gamma_default = float(gamma)
        self.NO_FRAG_default = float(NO_FRAG)
        self.int_bre_default = float(int_bre)
        self.Df_default = float(Df)
        self.MAS_default = float(MAS)

        self.rate_min = float(rate_min)
        self.rate_max = None if rate_max is None else float(rate_max)
        self.eps_E = float(eps_E)
        self.A0_run = A0_run

    # ------------------------------------------------------------------
    # å†…éƒ¨å·¥å…·ï¼šä»Ž pbe èŽ·å– LMC å‚æ•°ï¼ˆè‹¥æ²¡æœ‰åˆ™ç”¨é»˜è®¤ï¼‰
    # ------------------------------------------------------------------
    def _get_lmc_params_from_pbe(self, pbe) -> tuple[float, float, float, float, float]:
        gamma = float(getattr(pbe, "lmc_gamma", self.gamma_default))
        NO_FRAG = float(getattr(pbe, "lmc_NO_FRAG", self.NO_FRAG_default))
        int_bre = float(getattr(pbe, "lmc_int_bre", self.int_bre_default))
        Df = float(getattr(pbe, "lmc_Df", self.Df_default))
        MAS = float(getattr(pbe, "lmc_MAS", self.MAS_default))
        return gamma, NO_FRAG, int_bre, Df, MAS

    # ------------------------------------------------------------------
    # å†…éƒ¨å·¥å…·ï¼šå…¥å°„èƒ½é‡ E_in(V)
    # ------------------------------------------------------------------
    def _energy_in(self, V: np.ndarray) -> np.ndarray:
        """
        å…¥å°„èƒ½é‡æ¨¡åž‹ï¼š

            è‹¥ energy_in_fn ä¸ä¸º Noneï¼š
                E_in(V) = energy_in_fn(V)
            å¦åˆ™ï¼š
                E_in(V) = lambda_E * V^energy_exp
        """
        V = np.asarray(V, dtype=float)
        if self.energy_in_fn is not None:
            E_in = self.energy_in_fn(V)
            return np.asarray(E_in, dtype=float)
        # é»˜è®¤ï¼špower law in V
        return self.lambda_E * (V**self.energy_exp)

    # ------------------------------------------------------------------
    # å†…éƒ¨å·¥å…·ï¼šä¸ºä¸€æ‰¹ç²’å­æž„é€ ç‰¹å¾ X
    # ------------------------------------------------------------------
    def _build_features_batch(
        self,
        pbe,
        indices: Optional[Sequence[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ä¸ºç»™å®š pbe ä¸­çš„è‹¥å¹²ç²’å­ï¼ˆindicesï¼‰æž„é€ ç‰¹å¾çŸ©é˜µ Xã€‚

        è¿”å›ž:
            X   : shape (n, 7) ç‰¹å¾
            V   : shape (n,)  å¯¹åº”é¢—ç²’ä½“ç§¯ï¼ˆä¾¿äºŽåŽç»­ç®— E_inï¼‰
        """
        dim = int(getattr(pbe, "dim", 1))
        a = int(getattr(pbe, "a_tot", 0))
        if a <= 0:
            return np.zeros((0, 7), dtype=float), np.zeros((0,), dtype=float)

        V_flat = np.asarray(pbe.V_flat, dtype=float)

        if indices is None:
            idx = np.arange(a, dtype=int)
        else:
            idx = np.asarray(indices, dtype=int)
            # ç®€å• sanity check
            idx = idx[(idx >= 0) & (idx < a)]
        if idx.size == 0:
            return np.zeros((0, 7), dtype=float), np.zeros((0,), dtype=float)

        # V & ç»„åˆ†ä½“ç§¯åˆ†æ•° X1
        if dim == 1:
            V = V_flat[-1, idx]
            X1 = np.ones_like(V)
        elif dim == 2:
            v1 = V_flat[0, idx]
            v3 = V_flat[1, idx]
            A = v1 + v3
            V = A
            X1 = np.where(A > 0.0, v1 / A, 0.5)
        else:
            raise NotImplementedError(f"MLPBreakageRateAdapter currently only supports dim=1 or 2 (got dim={dim})")
        
        V /= self.A0_run
        gamma, NO_FRAG, int_bre, Df, MAS = self._get_lmc_params_from_pbe(pbe)

        # æž„é€ ç‰¹å¾
        logV = np.log(np.maximum(V, 1e-30))
        log_gamma = np.log(gamma)
        log_NOFRAG = np.log(NO_FRAG)

        # æ ‡é‡ â†’ å‘é‡å¹¿æ’­
        log_gamma_v = np.full_like(logV, log_gamma, dtype=float)
        log_nf_v = np.full_like(logV, log_NOFRAG, dtype=float)
        int_bre_v = np.full_like(logV, int_bre, dtype=float)
        Df_v = np.full_like(logV, Df, dtype=float)
        MAS_v = np.full_like(logV, MAS, dtype=float)

        X = np.stack(
            [logV, log_gamma_v, log_nf_v, int_bre_v, Df_v, MAS_v, X1],
            axis=1,
        )  # shape (n, 7)

        return X, V

    # ------------------------------------------------------------------
    # å…¬å…±æŽ¥å£ï¼šå…¨è¡¨ç ´ç¢ŽçŽ‡
    # ------------------------------------------------------------------
    def compute_rates_full(self, pbe) -> np.ndarray:
        """
        è®¡ç®—å½“å‰ pbe ä¸­æ‰€æœ‰æ´»è·ƒé¢—ç²’çš„ç ´ç¢ŽçŽ‡æ•°ç»„ï¼Œshape = (a_tot,)ã€‚

        æ­¥éª¤ï¼š
          1. æ ¹æ® pbe.V_flat, pbe.dim ç­‰æž„é€ ç‰¹å¾ Xï¼›
          2. è°ƒç”¨ MLP æ¨¡åž‹é¢„æµ‹ logE_needï¼›
          3. è®¡ç®— E_in(V)ï¼›
          4. è®¡ç®— rate = E_in / E_needï¼Œå¹¶åšè£å‰ªã€‚
        """
        X, V = self._build_features_batch(pbe, indices=None)
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=float)

        # é¢„æµ‹ logE_need
        logE_need = self.model.predict(X)           # shape (n,)
        E_need = np.exp(np.asarray(logE_need, dtype=float))
        E_need = np.maximum(E_need, self.eps_E)

        # å…¥å°„èƒ½é‡
        E_in = self._energy_in(V)

        # ç ´ç¢ŽçŽ‡
        rates = E_in / E_need
        rates = np.maximum(rates, self.rate_min)
        if self.rate_max is not None:
            rates = np.minimum(rates, self.rate_max)

        # --------------------------------------------------
        # debug print if very small volumes appear
        # --------------------------------------------------
        # mask_small = V < 1.0
        # if np.any(mask_small):
        #     idx = np.where(mask_small)[0]
        #     print(
        #         "[MLPBreakageRateAdapter] Detected V < 1.0 in compute_rates_full:\n"
        #         f"  indices : {idx.tolist()}\n"
        #         f"  V       : {V[mask_small]}\n"
        #         f"  rates   : {rates[mask_small]}"
        #     )
        
        return rates

    # ------------------------------------------------------------------
    # å…¬å…±æŽ¥å£ï¼šå•é¢—ç²’ç ´ç¢ŽçŽ‡
    # ------------------------------------------------------------------
    def compute_rate_single(self, pbe, i: int) -> float:
        """
        è®¡ç®— pbe ä¸­ç¬¬ i ä¸ªé¢—ç²’çš„ç ´ç¢ŽçŽ‡ã€‚

        é€šå¸¸ç”¨äºŽï¼š
          - ç ´ç¢Žäº‹ä»¶ä¹‹åŽï¼Œæ–°ç”Ÿæˆé¢—ç²’çš„å±€éƒ¨æ›´æ–°
          - æŸä¸ªé¢—ç²’ä½“ç§¯/ç»„åˆ†å‘ç”Ÿå˜åŒ–åŽçš„å±€éƒ¨æ›´æ–°
        """
        a = int(getattr(pbe, "a_tot", 0))
        if i < 0 or i >= a:
            return 0.0

        X, V = self._build_features_batch(pbe, indices=[i])
        if X.shape[0] == 0:
            return 0.0

        logE_need = self.model.predict(X[0:1, :])   # shape (1,)
        E_need = float(np.exp(logE_need[0]))
        if E_need < self.eps_E:
            E_need = self.eps_E

        E_in = float(self._energy_in(np.array([V[0]], dtype=float))[0])
        rate = E_in / E_need
        if rate < self.rate_min:
            rate = self.rate_min
        if self.rate_max is not None and rate > self.rate_max:
            rate = self.rate_max

        return float(rate)

