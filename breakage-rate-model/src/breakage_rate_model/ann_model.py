# -*- coding: utf-8 -*-
"""
更“黑盒”的 ANN 能量 surrogate：

    y = log E_need(V, θ) ≈ f_ANN([log V, log γ, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2])

相比 MLPEnergyModel，这里：
- 默认网络更深/更宽
- 使用 BatchNorm + Dropout，训练更稳定、表达能力更强
- 仍然直接回归 logE，与其它模型保持一致
"""

from __future__ import annotations

from typing import Optional, Sequence, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseEnergyModel


# =============================================================================
# 深层 ANN 网络定义
# =============================================================================

class _ANNNet(nn.Module):
    """
    稍微“黑盒”一点的前馈网络：

        input_dim
          -> (Linear -> BatchNorm -> 激活 -> Dropout) * N
          -> Linear(最后一层到 1)

    - hidden_sizes 控制层数和宽度，比如 (256, 256, 128)
    - activation: ReLU / SiLU / Tanh
    - dropout: 每层后面加一个小概率 Dropout 以防过拟合
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        activation: str = "silu",
        dropout: float = 0.1,
    ):
        super().__init__()

        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "silu":
            act_layer = nn.SiLU
        elif activation.lower() == "tanh":
            act_layer = nn.Tanh
        else:
            raise ValueError(
                f"Unknown activation '{activation}', choose from 'relu', 'silu', 'tanh'."
            )

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        # 输出层：1 维，线性
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输出形状: (batch,)
        return self.net(x).squeeze(-1)


# =============================================================================
# ANNEnergyModel
# =============================================================================

class ANNEnergyModel(BaseEnergyModel):
    """
    更“黑盒”的 ANN 模型：

        输入: X[i] = [logV, log γ, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
        输出: y_pred[i] = log(E_pred)

    特点：
    - 相比 MLPEnergyModel 更深/更宽
    - 使用 BatchNorm + Dropout
    - Adam + weight decay
    - 支持验证集 + early stopping
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        activation: str = "silu",
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        batch_size: int = 256,
        patience: int = 30,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """
        参数
        ----
        input_dim:
            input feature dimension, default 10 ([logV, log_gamma, log_NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2])
        hidden_sizes:
            隐藏层大小序列，例如 (256, 256, 128)
        activation:
            激活函数: 'relu' / 'silu' / 'tanh'
        dropout:
            每层的 Dropout 概率（0~1），如 0.1
        lr:
            Adam 学习率（稍微小一点，因为网络更深）
        weight_decay:
            Adam 的 weight decay（L2 正则）
        max_epochs:
            最大训练轮数
        batch_size:
            批大小
        patience:
            early stopping 容忍的连续无提升 epoch 数
        device:
            'cpu' / 'cuda' / None（自动判断）
        seed:
            随机种子（可选）
        """
        super().__init__(name=name or "ANNEnergyModel")

        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.seed = seed
        if self.seed is not None:
            self._set_seed(self.seed)

        self._net: Optional[_ANNNet] = None

        # 训练记录（可选）
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        
        # 输入特征标准化参数（在 fit() 中初始化）
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 工具：设置随机种子
    # ------------------------------------------------------------------
    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # 拟合接口
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ANNEnergyModel":
        """
        训练 ANN 模型。

        参数
        ----
        X_train, y_train:
            训练集特征和标签（numpy 数组），y_train = log(E)
        X_val, y_val:
            验证集特征和标签（可选，若提供则启用 early stopping）
        """
        if X_train is None or y_train is None:
            raise ValueError("ANNEnergyModel.fit requires X_train and y_train (numpy arrays).")

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        n_train, d = X_train.shape
        if d != self.input_dim:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim}, got {d}")

        # -------------------------------------------------------
        # 🔥 输入标准化：记录 mean/std，并应用到 X_train / X_val
        # -------------------------------------------------------
        self._x_mean = X_train.mean(axis=0)
        self._x_std = X_train.std(axis=0)
        self._x_std[self._x_std < 1e-12] = 1.0  # 避免除以 0

        X_train = (X_train - self._x_mean) / self._x_std

        has_val = (X_val is not None) and (y_val is not None)
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            X_val = (X_val - self._x_mean) / self._x_std
            y_val = np.asarray(y_val, dtype=np.float32)
        else:
            X_val = None
            y_val = None
        # -------------------------------------------------------

        # 构建网络
        self._net = _ANNNet(
            input_dim=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            dropout=self.dropout,
        )
        self._net.to(self.device)

        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # 训练集 DataLoader
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 验证集 DataLoader（若有）
        if has_val:
            val_dataset = TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            val_loader = None

        best_val_loss = float("inf")
        best_state_dict = None
        no_improve_epochs = 0

        self._train_losses = []
        self._val_losses = []

        for epoch in range(self.max_epochs):
            # ---- 训练阶段 ----
            self._net.train()
            train_loss_sum = 0.0
            n_train_batches = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                y_pred = self._net(xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                n_train_batches += 1

            avg_train_loss = train_loss_sum / max(n_train_batches, 1)
            self._train_losses.append(avg_train_loss)

            # ---- 验证阶段（如果有） ----
            if has_val and val_loader is not None:
                self._net.eval()
                val_loss_sum = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        y_pred = self._net(xb)
                        loss = criterion(y_pred, yb)
                        val_loss_sum += loss.item()
                        n_val_batches += 1

                avg_val_loss = val_loss_sum / max(n_val_batches, 1)
                self._val_losses.append(avg_val_loss)

                print(
                    f"[Epoch {epoch+1:03d}] train_loss={avg_train_loss:.6g}, "
                    f"val_loss={avg_val_loss:.6g}, no_improve={no_improve_epochs}"
                )

                if avg_val_loss < best_val_loss - 1e-6:
                    best_val_loss = avg_val_loss
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in self._net.state_dict().items()
                    }
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if self.patience is not None and no_improve_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}, best val_loss={best_val_loss:.6g}")
                    break
            else:
                # 无验证集时仅打印训练损失
                print(f"[Epoch {epoch+1:03d}] train_loss={avg_train_loss:.6g}")

        # 若有验证集且记录了 best_state_dict，则恢复最佳参数
        if has_val and best_state_dict is not None:
            self._net.load_state_dict(best_state_dict)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        输入特征矩阵 X（numpy 数组），输出预测的 logE（numpy 数组）。

            X[i] = [logV, log γ, log NO_FRAG, int_bre, Df, MAS, X1, STR0, STR1, STR2]
        """
        if not self._is_fitted or self._net is None:
            raise RuntimeError("ANNEnergyModel is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(
                f"X shape not compatible: got {X.shape}, expect (n_samples, {self.input_dim})"
            )

        # -----------------------------------------
        # 🔥 应用训练时记录的 mean/std
        # -----------------------------------------
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("Scaler not initialized. Did you call fit()?")
        X = (X - self._x_mean) / self._x_std
        # -----------------------------------------

        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            y_pred = self._net(xb).cpu().numpy()  # shape (n_samples,)

        return y_pred


    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        """
        与 predict 相同，但返回的是 E_pred 而非 log(E_pred)。
        """
        logE = self.predict(X)
        return np.exp(logE)


