import numpy as np
import pandas as pd
import xgboost as xgb

def cox_event_prob(model, baseline_survival: pd.DataFrame, X_sample: pd.DataFrame, time_horizon: float) -> float:
    """
    Cox-PH 事件概率
    :param model: lifelines.CoxPHFitter 模型
    :param baseline_survival: Cox 模型自带 baseline_survival_ (DataFrame)
    :param X_sample: 单一样本 DataFrame
    :param time_horizon: 时间点 t
    :return: 事件发生概率 (float)
    """
    # Cox 部分风险分数 (partial hazard)
    risk_score = model.predict_partial_hazard(X_sample)[0]

    # 找到 <= t 的最后一个基线生存值 S0(t)
    S0_t = float(baseline_survival.loc[:time_horizon].iloc[-1, 0])

    # 个体生存概率 S(t|x)
    S_xt = S0_t ** np.exp(risk_score)

    # 事件发生概率 = 1 - 生存概率
    return float(1.0 - S_xt)


def xgb_event_prob(xgb_model: xgb.Booster, baseline_survival: pd.DataFrame, X_sample: pd.DataFrame, time_horizon: float) -> float:
    """
    XGBoost Cox 事件概率
    :param xgb_model: 训练好的 XGBoost Cox 模型 (Booster)
    :param baseline_survival: 来自 Cox 模型的基线生存函数
    :param X_sample: 单一样本 DataFrame
    :param time_horizon: 时间点 t
    :return: 事件发生概率 (float)
    """
    dmatrix = xgb.DMatrix(X_sample)
    risk_score = xgb_model.predict(dmatrix)[0]

    # 找到 <= t 的最后一个基线生存值 S0(t)
    S0_t = float(baseline_survival.loc[:time_horizon].iloc[-1, 0])

    # 个体生存概率 S(t|x)
    S_xt = S0_t ** np.exp(risk_score)

    return float(1.0 - S_xt)
