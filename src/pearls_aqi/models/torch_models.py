"""Small PyTorch feed-forward AQI regressors, one per forecast horizon."""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn


class MultiHorizonTorchModel:
    def __init__(self, feature_cols: List[str], epochs: int = 300, hidden_size: int = 96):
        self.feature_cols, self.epochs, self.hidden_size = feature_cols, epochs, hidden_size
        self.imputers, self.scalers, self.target_scalers, self.models = {}, {}, {}, {}

    def _columns_for(self, target: str) -> List[str]:
        return self.feature_cols[target] if isinstance(self.feature_cols, dict) else self.feature_cols

    def fit(self, train_df: pd.DataFrame, target_cols: List[str]):
        torch.manual_seed(42)
        for target in target_cols:
            valid = train_df[target].notna()
            feature_cols = self._columns_for(target)
            raw_x = train_df.loc[valid, feature_cols]
            raw_y = train_df.loc[valid, [target]]
            split = max(1, int(len(raw_x) * 0.85))
            imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
            x_train = scaler.fit_transform(imputer.fit_transform(raw_x.iloc[:split]))
            x_valid = scaler.transform(imputer.transform(raw_x.iloc[split:]))
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(raw_y.iloc[:split]).astype(np.float32)
            y_valid = y_scaler.transform(raw_y.iloc[split:]).astype(np.float32)
            model = nn.Sequential(
                nn.Linear(x_train.shape[1], self.hidden_size), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Linear(self.hidden_size // 2, 1),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            x_tensor, y_tensor = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train)
            x_validation, y_validation = torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid)
            best_loss, best_state, patience = float("inf"), None, 0
            for _ in range(self.epochs):
                model.train()
                optimizer.zero_grad()
                nn.MSELoss()(model(x_tensor), y_tensor).backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_loss = nn.MSELoss()(model(x_validation), y_validation).item()
                scheduler.step(validation_loss)
                if validation_loss < best_loss:
                    best_loss, best_state, patience = validation_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 30:
                        break
            model.load_state_dict(best_state)
            self.imputers[target], self.scalers[target], self.target_scalers[target], self.models[target] = imputer, scaler, y_scaler, model.eval()

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        output = {}
        for target, model in self.models.items():
            x = self.scalers[target].transform(self.imputers[target].transform(df[self._columns_for(target)]))
            with torch.no_grad():
                scaled = model(torch.tensor(x, dtype=torch.float32)).numpy()
            output[target] = self.target_scalers[target].inverse_transform(scaled).ravel()
        return output
