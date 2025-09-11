import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

import xgboost as xgb
from scipy.stats import norm, logistic

def xgb_aft_survival_function(model, X, time_points, distribution="normal"):
    """
    返回 shape: (len(X), len(time_points))
    """
    dmatrix = xgb.DMatrix(X)
    mu = model.predict(dmatrix)
    sigma = float(model.attr('aft_sigma'))  # 获取 sigma
    surv = []
    cdf = norm.cdf


    for m in mu:
        s = 1 - cdf((np.log(time_points) - m) / sigma)
        surv.append(s)
    return np.array(surv)


from sklearn.linear_model import LogisticRegression

def plot_survival_curves(X, cox_model=None, rsf_model=None, xgb_aft_model=None,
                         deepsurv_model=None):
    """
    绘制 Cox / RSF / XGB-AFT / DeepSurv / Logistic 回归 生存曲线
    """
    fig = go.Figure()

    # ----------------- Cox -----------------
    if cox_model is not None:
        X_cox = X[cox_model.feature_names_in_] if hasattr(cox_model, "feature_names_in_") else X.copy()
        surv_func = cox_model.predict_survival_function(X_cox)
        mean_surv = surv_func.mean(axis=1).values
        fig.add_trace(go.Scatter(x=surv_func.index, y=mean_surv, mode='lines', name='Cox'))

    # ----------------- RSF -----------------
    if rsf_model is not None:
        X_rsf = X[rsf_model.feature_names_in_] if hasattr(rsf_model, "feature_names_in_") else X.copy()
        surv_curves = rsf_model.predict_survival_function(X_rsf)
        first_curve = surv_curves[0]
        if hasattr(first_curve, "x"):
            rsf_time = first_curve.x
        else:
            rsf_time = np.linspace(0, 1, 100)
        indiv_surv_list = [fn(rsf_time) for fn in surv_curves]
        mean_surv = np.array(indiv_surv_list).mean(axis=0)
        fig.add_trace(go.Scatter(x=rsf_time, y=mean_surv, mode='lines', name='RSF'))

    fig.update_layout(
        title="Kaplan-Meier Survival Curves From Different Models",
        xaxis_title="Time (days)",
        yaxis_title="Survival Probability",
        yaxis_range=[0, 1],
        height=400
    )
    return fig


# ---- Streamlit 中展示 ----
# fig = plot_survival_curves(X, cox_model=cph, rsf_model=rsf, xgb_aft_model=xgb_model, deepsurv_model=ds_model)
# st.plotly_chart(fig, use_container_width=True)
