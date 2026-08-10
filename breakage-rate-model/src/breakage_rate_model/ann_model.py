# -*- coding: utf-8 -*-
"""Deeper ANN surrogate with the canonical full-vector feature interface."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseEnergyModel
from .features import (
    FULL_ENERGY_FEATURE_NAMES,
    active_feature_indices,
    normalize_active_feature_names,
    require_full_feature_matrix,
)


class _ANNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int], activation: str, dropout: float):
        super().__init__()
        activation_name = activation.lower()
        if activation_name == "relu":
            activation_layer = nn.ReLU
        elif activation_name == "silu":
            activation_layer = nn.SiLU
        elif activation_name == "tanh":
            activation_layer = nn.Tanh
        else:
            raise ValueError("activation must be one of 'relu', 'silu', or 'tanh'.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: List[nn.Module] = []
        layer_input_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend((nn.Linear(layer_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), activation_layer()))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            layer_input_dim = hidden_dim
        layers.append(nn.Linear(layer_input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ANNEnergyModel(BaseEnergyModel):
    """BatchNorm/Dropout ANN for ``log(E_need)`` with a stable 10-column API."""

    def __init__(
        self,
        input_dim: int = len(FULL_ENERGY_FEATURE_NAMES),
        hidden_sizes: Sequence[int] = (256, 256, 128),
        activation: str = "silu",
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        batch_size: int = 256,
        patience: Optional[int] = 30,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        active_feature_names: Optional[Sequence[str]] = None,
    ):
        super().__init__(name=name or "ANNEnergyModel")
        if input_dim != len(FULL_ENERGY_FEATURE_NAMES):
            raise ValueError(
                f"input_dim must remain {len(FULL_ENERGY_FEATURE_NAMES)} for the canonical "
                "external feature interface."
            )
        self.input_dim = input_dim
        self.active_feature_names = normalize_active_feature_names(active_feature_names)
        self._active_feature_indices = active_feature_indices(self.active_feature_names)
        self._model_input_dim = int(self._active_feature_indices.size)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 because ANN uses BatchNorm.")
        self.patience = patience
        self.device = torch.device("cuda" if device is None and torch.cuda.is_available() else device or "cpu")
        self.seed = seed
        if seed is not None:
            self._set_seed(seed)
        self._net: Optional[_ANNNet] = None
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_active_features(self, X: np.ndarray) -> np.ndarray:
        X_full = require_full_feature_matrix(X)
        return X_full[:, self._active_feature_indices].astype(np.float32, copy=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ANNEnergyModel":
        if X_train is None or y_train is None:
            raise ValueError("ANNEnergyModel.fit requires X_train and y_train.")
        X_train_active = self._select_active_features(X_train)
        y_train = np.asarray(y_train, dtype=np.float32)
        if y_train.ndim != 1 or y_train.shape[0] != X_train_active.shape[0] or not np.all(np.isfinite(y_train)):
            raise ValueError("y_train must be a finite one-dimensional array aligned with X_train.")
        if X_train_active.shape[0] < 2:
            raise ValueError("ANNEnergyModel requires at least two training samples for BatchNorm.")

        self._x_mean = X_train_active.mean(axis=0)
        self._x_std = X_train_active.std(axis=0)
        constant_columns = np.where(self._x_std < 1e-12)[0]
        if constant_columns.size:
            names = tuple(self.active_feature_names[index] for index in constant_columns)
            raise ValueError(f"Selected training features are constant: {names}. Remove them from active_feature_names.")
        X_train_active = (X_train_active - self._x_mean) / self._x_std

        has_val = (X_val is not None) and (y_val is not None)
        if has_val:
            X_val_active = self._select_active_features(X_val)
            y_val = np.asarray(y_val, dtype=np.float32)
            if y_val.ndim != 1 or y_val.shape[0] != X_val_active.shape[0] or not np.all(np.isfinite(y_val)):
                raise ValueError("y_val must be a finite one-dimensional array aligned with X_val.")
            X_val_active = (X_val_active - self._x_mean) / self._x_std
        else:
            X_val_active = None

        self._net = _ANNNet(self._model_input_dim, self.hidden_sizes, self.activation, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train_active), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        if X_train_active.shape[0] % self.batch_size == 1:
            raise ValueError(
                "ANNEnergyModel would create a final one-sample BatchNorm training batch; "
                "choose a different batch_size."
            )
        val_loader = None
        if has_val:
            assert X_val_active is not None and y_val is not None
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_val_active), torch.from_numpy(y_val)),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )

        best_val_loss = float("inf")
        best_state_dict = None
        no_improve_epochs = 0
        self._train_losses = []
        self._val_losses = []
        for epoch in range(self.max_epochs):
            self._net.train()
            train_loss_sum = 0.0
            n_train_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                n_train_batches += 1
            train_loss = train_loss_sum / n_train_batches
            self._train_losses.append(train_loss)

            if val_loader is None:
                print(f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.6g}")
                continue
            self._net.eval()
            val_loss_sum = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    val_loss_sum += float(criterion(self._net(xb), yb).item())
                    n_val_batches += 1
            val_loss = val_loss_sum / n_val_batches
            self._val_losses.append(val_loss)
            print(
                f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.6g}, "
                f"val_loss={val_loss:.6g}, no_improve={no_improve_epochs}"
            )
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state_dict = {key: value.detach().cpu().clone() for key, value in self._net.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if self.patience is not None and no_improve_epochs >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.6g}")
                break

        if best_state_dict is not None:
            self._net.load_state_dict(best_state_dict)
        self._net.to(torch.device("cpu"))
        self.device = torch.device("cpu")
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("ANNEnergyModel is not fitted yet.")
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("ANNEnergyModel feature scaler is not initialized.")
        X_active = self._select_active_features(X)
        X_active = (X_active - self._x_mean) / self._x_std
        self._net.eval()
        with torch.no_grad():
            return self._net(torch.from_numpy(X_active).to(self.device)).cpu().numpy()

    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict(X))
