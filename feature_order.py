# feature_order.py
import json
import os
import numpy as np
import pandas as pd

def _unwrap_estimator(model):
    """尽量从Pipeline/Calibrated/Wrapper中取到最终estimator"""
    m = model
    # sklearn Pipeline
    if hasattr(m, "steps"):
        m = m.steps[-1][1]
    # CalibratedClassifierCV
    if hasattr(m, "base_estimator"):
        m = m.base_estimator
    return m

def extract_feature_order_from_model(model, fallback_cols):
    """
    从模型对象尽可能提取训练时的列顺序。
    优先：feature_names_in_ / get_feature_names_out()
    其次：xgboost Booster 的 feature_names（若存在）
    最后：使用fallback_cols（如当前input_data.columns）
    """
    m = _unwrap_estimator(model)

    # 1) 标准sklearn接口
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)

    # 2) 支持 get_feature_names_out 的前处理/管道
    if hasattr(m, "get_feature_names_out"):
        try:
            return list(m.get_feature_names_out())
        except Exception:
            pass

    # 3) xgboost Booster（若通过DMatrix训练可能没有保留）
    if hasattr(m, "feature_names") and m.feature_names is not None:
        try:
            return list(m.feature_names)
        except Exception:
            pass

    # 4) lifelines / 自定义对象通常没有列信息，回退
    return list(fallback_cols)

def build_and_save_feature_orders(models_dict, output_json_path, sample_df):
    """
    针对每个模型生成列顺序，并保存到JSON。
    models_dict: {'cox': model, 'rsf': model, ...}
    sample_df:   当前页面组装的input_data（用于回退列顺序）
    """
    orders = {}
    for key, model in models_dict.items():
        if model is None:
            continue
        try:
            cols = extract_feature_order_from_model(model, sample_df.columns)
            orders[key] = cols
        except Exception:
            # 出错时也回退
            orders[key] = list(sample_df.columns)

    # 额外给一个'common'备用
    if 'common' not in orders:
        orders['common'] = list(sample_df.columns)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=2)

    return orders

def load_feature_orders(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)