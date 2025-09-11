# # nn_wrapper.py
# import numpy as np
# import torch
# from torch import nn
#
# class ProbWrapper:
#     """
#     统一对外接口：predict_proba(X) / predict(X)
#     内部包含:
#       - model: torch.nn.Module
#       - scaler: StandardScaler 或 None
#       - features: 训练用特征顺序(list[str])
#     """
#     def __init__(self, model: nn.Module, scaler=None, features=None):
#         self.model = model
#         self.scaler = scaler
#         self.features = list(features) if features is not None else None
#         self.model.eval()
#
#     def _prepare(self, X):
#         X = np.asarray(X, dtype=np.float32)
#         if self.scaler is not None:
#             # 注意：scaler 期望 float64，但我们只用于中心化/缩放，cast 回 float32 喂模型
#             X = self.scaler.transform(X).astype(np.float32)
#         return torch.from_numpy(X)
#
#     def predict_proba(self, X):
#         t = self._prepare(X)
#         with torch.no_grad():
#             out = self.model(t).squeeze()
#             p = torch.sigmoid(out).cpu().numpy()
#         # 统一成二列概率输出 [p0, p1]
#         p1 = p.astype(np.float64)
#         p0 = 1.0 - p1
#         return np.vstack([p0, p1]).T
#
#     def predict(self, X, threshold=0.5):
#         proba = self.predict_proba(X)
#         return (proba[:, 1] > threshold).astype(int)
# # nn_wrapper.py
# import numpy as np
# import torch
# from torch import nn
#
# class ProbWrapper:
#     """
#     统一接口：
#       - predict_proba(X) / predict(X)
#       - predict_risk(X)  （越大越危险）
#       - baseline_survival(times)  （返回随时间的基线生存函数）
#       - predict_survival_curve(X) （返回个体生存曲线）
#     """
#     def __init__(self, model: nn.Module, scaler=None, features=None,
#                  t_base=None, S_base=None):
#         self.model = model
#         self.scaler = scaler
#         self.features = list(features) if features is not None else None
#         self.model.eval()
#         # 基线生存函数
#         self.time_index = t_base
#         self.baseline_survival_array = S_base
#
#     def _prepare(self, X):
#         X = np.asarray(X, dtype=np.float32)
#         if self.scaler is not None:
#             X = self.scaler.transform(X).astype(np.float32)
#         return torch.from_numpy(X)
#
#     # ------------------ 生存相关 ------------------
#     def baseline_survival(self, times):
#         if self.baseline_survival_array is None or self.time_index is None:
#             raise ValueError("baseline_survival not set!")
#         # 简单线性插值
#         return np.interp(times, self.time_index, self.baseline_survival_array)
#
#     def predict_survival_curve(self, X):
#         """返回每个样本的个体生存曲线"""
#         if self.baseline_survival_array is None:
#             raise ValueError("baseline_survival not set!")
#         X_t = self._prepare(X)
#         with torch.no_grad():
#             risk = self.model(X_t).squeeze().cpu().numpy()
#         surv_curves = np.array([self.baseline_survival_array ** np.exp(r) for r in risk])
#         return surv_curves
#
#     # ------------------ 二分类相关 ------------------
#     def predict_proba(self, X):
#         t = self._prepare(X)
#         with torch.no_grad():
#             out = self.model(t).squeeze()
#             p = torch.sigmoid(out).cpu().numpy()
#         # 统一成二列概率输出 [p0, p1]
#         p1 = p.astype(np.float64)
#         p0 = 1.0 - p1
#         return np.vstack([p0, p1]).T
#
#     def predict(self, X, threshold=0.5):
#         proba = self.predict_proba(X)
#         return (proba[:, 1] > threshold).astype(int)
#
#     def predict_risk(self, X):
#         prob = self.predict_proba(X)[:,1]
#         return -np.log(prob + 1e-8)  # 越大越危险
# nn_wrapper_surv.py
import numpy as np
import torch
from torch import nn

class ProbWrapper:
    """
    DeepSurv wrapper，仅支持生存曲线
    """
    def __init__(self, model: nn.Module, scaler=None, features=None,
                 t_base=None, S_base=None):
        self.model = model
        self.scaler = scaler
        self.features = list(features) if features else None
        self.model.eval()
        self.time_index = t_base
        self.baseline_survival_array = S_base

    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        return torch.from_numpy(X)

    def baseline_survival(self, times):
        if self.baseline_survival_array is None or self.time_index is None:
            raise ValueError("baseline_survival not set!")
        return np.interp(times, self.time_index, self.baseline_survival_array)

    def predict_survival_curve(self, X):
        """返回每个样本的生存曲线 S(t|X)"""
        if self.baseline_survival_array is None:
            raise ValueError("baseline_survival not set!")

        X_t = self._prepare(X)
        with torch.no_grad():
            risk = self.model(X_t).squeeze().cpu().numpy()

        # ✅ 确保 risk 可迭代
        risk = np.atleast_1d(risk)

        return np.array([self.baseline_survival_array ** np.exp(r) for r in risk])

    def predict_risk(self, X):
        """返回风险分数（越大越危险）"""
        X_t = self._prepare(X)
        with torch.no_grad():
            risk = self.model(X_t).squeeze().cpu().numpy()
        return risk
    # ------------------ 二分类相关 ------------------
    def predict_proba(self, X):
        t = self._prepare(X)
        with torch.no_grad():
            out = self.model(t).squeeze()
            p = torch.sigmoid(out).cpu().numpy()
        # 统一成二列概率输出 [p0, p1]
        p1 = p.astype(np.float64)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)

    def predict_risk(self, X):
        prob = self.predict_proba(X)[:,1]
        return -np.log(prob + 1e-8)  # 越大越危险