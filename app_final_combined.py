from feature_order import build_and_save_feature_orders, load_feature_orders
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import joblib
import pickle
import warnings
import os
import random
import sys, types
from torch import nn
import tempfile

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# --- Language Configuration ---
LANGUAGES = {
    "中文": {
        "page_title": "肝细胞癌无复发生存分析预测系统",
        "main_title": "<h1 style='text-align: center;'>肝细胞癌无复发生存分析预测系统</h1>",
        "disclaimer": "<h3 style='text-align: center; color: red;'>声明：仅作为研究目的</h3>",
        "sidebar_header_info": "📊 患者信息输入",
        "sidebar_subheader_basic": "基本信息",
        "age_label": "年龄",
        "gender_label": "性别（男=1，女=0）",
        "sidebar_subheader_lab": "实验室检查指标",
        "alb_label": "白蛋白 ALB (g/L)",
        "ast_label": "天门冬氨酸氨基转移酶 AST (U/L)",
        "tbil_label": "总胆红素 TBIL (umol/L)",
        "albi_label": "ALBI评分",
        "alt_label": "丙氨酸氨基转移酶 ALT (U/L)",
        "afp_label": "甲胎蛋白 AFP (μg/L)",
        "pt_label": "凝血酶原时间 (s)",
        "inr_label": "国际标准化比值 INR",
        "wbc_label": "白细胞计数 WBC (10^9/L)",
        "sidebar_subheader_surgery": "术中指标",
        "blood_loss_label": "术中失血量 (mL)",
        "wide_resection_label": "扩大切除 （是=1，否=0）",
        "sidebar_subheader_pathology": "病理指标",
        "tumor_mvi_label": "肿瘤微血管侵犯（M0=1，M1或M2=0）",
        "tumor_diameter_label": "肿瘤直径（cm）",
        "hepatitis_label": "肝炎（是=1, 否=0）",
        "cirrhosis_label": "肝硬化（是=1，否=0）",
        "capsule_invasion_label": "包膜侵犯（是=1, 否=0）",
        "tumor_differentiation_label": "肿瘤分化 (巨块型=1, 卫星结节型或其他=0）",
        "child_stage_label": "Child-Pugh分期（A=1, B或C=0）",
        "predict_button_label": "🚀 开始预测",
        "reset_button_label": "🔄 重置输入",
        "generating_results": "✅ 正在产出结果...",
        "input_confirmation": "输入确认",
        "basic_info_title": "基本信息:",
        "disease_surgery_title": "病理与手术指标:",
        "age_display": "- 年龄: {age} ",
        "gender_display": "- 性别（男=1，女=0）: {gender}",
        "alb_display": "- ALB: {alb} g/L",
        "ast_display": "- AST: {ast} U/L",
        "alt_display": "- ALT: {alt} U/L",
        "tbil_display": "- TBIL: {tbil} umol/L",
        "albi_display": "- ALBI分期: {albi}",
        "afp_display": "- AFP: {afp} μg/L",
        "afp_greater_400_display": "- AFP > 400 μg/L: {afp_greater_400}",
        "afp_less_400_display": "- AFP ≤ 400 μg/L: {afp_less_400}",
        "pt_display": "- PT: {pt} s",
        "inr_display": "- INR: {inr}",
        "wbc_display": "- WBC: {wbc} ×10^9/L",
        "blood_loss_display": "- 术中失血量: {blood_loss} mL",
        "tumor_mvi_display": "- 肿瘤微血管侵犯（M0=1，M1或M2=0）: {tumor_mvi}",
        "tumor_diameter_display": "- 肿瘤直径: {tumor_diameter} cm",
        "cirrhosis_display": "- 肝硬化（是=1，否=0）: {cirrhosis}",
        "capsule_invasion_display": "- 包膜侵犯（是=1, 否=0）: {capsule_invasion}",
        "tumor_differentiation_display": "- 肿瘤分化 (巨块型=1, 卫星结节型或其他=0）: {tumor_differentiation}",
        "hepatitis_display": "- 肝炎（是=1, 否=0）: {hepatitis}",
        "wide_resection_display": "- 扩大切除 （是=1，否=0）: {wide_resection}",
        "child_stage_display": "- Child-Pugh分期（A=1, B或C=0）: {child_stage}",
        "results_subheader": "结果",
        "risk_probability_title": "**各模型风险概率:**",
        "model_col": "模型名称",
        "risk_col": "风险概率",
        "risk_level_col": "风险等级",
        "high_risk": "高风险",
        "medium_risk": "中风险",
        "low_risk": "低风险",
        "comparison_chart_title": "各模型风险概率对比",
        "chart_x_label": "模型",
        "chart_y_label": "风险概率",
        "comprehensive_risk_assessment": "**综合风险评估:**",
        "overall_risk": "综合风险: {overall_risk}",
        "avg_risk_prob": "平均风险概率: {avg_risk:.3f}",
        "recommendation_high": "建议: 密切监测，积极治疗",
        "recommendation_medium": "建议: 定期随访，注意观察",
        "recommendation_low": "建议: 继续监测，保持现状",
        "model_consistency": "**模型一致性:**",
        "consistency_high": "模型预测高度一致",
        "consistency_medium": "模型预测基本一致",
        "consistency_low": "模型预测存在分歧",
        "risk_std_dev": "标准差: {risk_std:.3f}",
        "survival_analysis_subheader": "生存分析",
        "xgboost_rfs_title": "无复发生存期（XGBoost预测）",
        "predicted_rfs": "预测生存: {display_value} 天",
        "shap_explainability_subheader": "SHAP解释",
        "feature_contribution_title": "特征贡献 (预生成)",
        "shap_waterfall_title": "SHAP瀑布图",
        "shap_x_label": "SHAP value",
        "individual_feature_contribution": "个体特征贡献",
        "result_export_subheader": "结果导出",
        "download_report_button": "📥 下载预测报告 (CSV)",
        "download_detailed_button": "📥 下载详细结果 (CSV)",
        "report_prediction_time": "预测时间",
        "report_patient_age": "患者年龄",
        "report_patient_gender": "患者性别",
        "report_albi_score": "ALBI评分",
        "report_afp_value": "AFP值",
        "report_afp_greater_400": "AFP_greater_400",
        "report_afp_less_400": "AFP_less_400",
        "report_tumor_diameter": "肿瘤直径",
        "report_cirrhosis": "Cirrhosis",
        "report_avg_risk": "综合风险概率",
        "report_overall_risk_level": "综合风险等级",
        "report_model_consistency": "模型一致性",
        "report_consistency_consistent": "一致",
        "report_consistency_divergent": "存在分歧",
        "initial_info_message": " 请在左侧输入患者信息，然后点击'开始预测'按钮",
        "afp_conversion_instructions_subheader": " AFP指标转换说明",
        "afp_conversion_rules_title": "**AFP指标自动转换规则:**",
        "afp_conversion_rule_1": "- 当AFP > 400 μg/L时：AFP_greater_400 = 1, AFP_less_400 = 0",
        "afp_conversion_rule_2": "- 当AFP ≤ 400 μg/L时：AFP_greater_400 = 0, AFP_less_400 = 1",
        "afp_conversion_example_title": "**示例:**",
        "afp_conversion_example_1": "- AFP = 500 → AFP_greater_400 = 1, AFP_less_400 = 0",
        "afp_conversion_example_2": "- AFP = 400 → AFP_greater_400 = 0, AFP_less_400 = 1",
        "afp_conversion_example_3": "- AFP = 200 → AFP_greater_400 = 0, AFP_less_400 = 1",
        "rsf_model_load_fail": "⚠️ RSF 模型加载失败: {error}",
        "xgboost_model_load_fail": "⚠️ XGBoost 模型加载失败: {error}",
        "xgboost_not_installed": "⚠️ XGBoost 未安装",
        "cox_model_load_fail": "⚠️ Cox 模型加载失败: {error}",
        "lr_model_load_fail": "⚠️ LR 模型加载失败: {error}",
        "nn_model_load_fail": "⚠️ NN 模型加载失败: {error}",
        "general_error": "❌ 错误: {error}",
        "rsf_predict_fail": "RSF模型没有可用的预测接口",
        "cox_predict_fail": "Cox模型没有可用的预测接口",
        "missing_feature_order": "缺少 {model_key} 的训练列顺序，请先生成 models/feature_order.json",
        "missing_features": "{model_key} 缺失特征: {missing}",
        "mock_model_fail": "{model_name} 未加载成功，且禁止使用随机兜底。请检查模型文件。",
        "waterfall_generation_failed": "瀑布图生成失败: {error}",
        "feature_mapping": {
            '年龄__y': '年龄',
            '性别_1': '性别（男=1，女=0）',
            'ALB': 'ALB (g/L)',
            'AST': 'AST (U/L)',
            'ALT': 'ALT (U/L)',
            'TBIL': 'TBIL (umol/L)',
            'AFP': 'AFP (μg/L)',
            'AFP_greater_400': 'AFP > 400',
            'AFP_less_400': 'AFP ≤ 400',
            'PT': 'PT (秒)',
            'INR': 'INR',
            'WBC': 'WBC (10^9/L)',
            '失血量': '术中失血量（mL）',
            '肿瘤MVI_M0': '肿瘤微血管侵犯（M0=1，M1或M2=0）',
            '肿瘤直径': '肿瘤直径（cm）',
            '是否合并肝硬化_1': '肝硬化（是=1，否=0）',
            '包膜是否受侵犯_未浸及': '包膜侵犯（是=1, 否=0）',
            '肿瘤是否巨块型分化_1': '肿瘤分化 (巨块型=1, 结节型或其他=0）',
            '是否合并肝炎_1': '肝炎（是=1, 否=0）',
            '是否大范围切除_1': '扩大切除 （是=1，否=0）',
            'child分期_A': 'Child-Pugh分期（A=1, B或C=0）',
            'ALBI': 'ALBI评分'
        }
    },
    "English": {
        "page_title": "HCC Recurrence-Free Survival Analysis Prediction System",
        "main_title": "<h1 style='text-align: center;'>HCC Recurrence-Free Survival Analysis Prediction System</h1>",
        "disclaimer": "<h3 style='text-align: center; color: red;'>Disclaimer: Only used for research purposes</h3>",
        "sidebar_header_info": "Info Input",
        "sidebar_subheader_basic": "Basic Info",
        "age_label": "Age (Years)",
        "gender_label": "Gender (Male=1, Female=0)",
        "sidebar_subheader_lab": "Laboratory indicators",
        "alb_label": "ALB (g/L)",
        "ast_label": "AST (U/L)",
        "tbil_label": "TBIL (umol/L)",
        "albi_label": "ALBI Score",
        "alt_label": "ALT (U/L)",
        "afp_label": "AFP (μg/L)",
        "pt_label": "PT (s)",
        "inr_label": "INR",
        "wbc_label": "WBC (10^9/L)",
        "sidebar_subheader_surgery": "Surgery",
        "blood_loss_label": "Blood Loss (mL)",
        "wide_resection_label": "Extensive Resection (yes=1, no=0)",
        "sidebar_subheader_pathology": "Pathology",
        "tumor_mvi_label": "Tumor MVI (M0=1, M1 or M2=0)",
        "tumor_diameter_label": "Tumor Diameter(cm)",
        "hepatitis_label": "Hepatitis (present=1, absent=0)",
        "cirrhosis_label": "Cirrhosis (present=1, absent=0)",
        "capsule_invasion_label": "Capsule Invasion (present=1, absent=0)",
        "tumor_differentiation_label": "Tumor Differentiation (giant mass type=1, Others=0)",
        "child_stage_label": "Child-Pugh Stage (A=1, B or C=0)",
        "predict_button_label": "🚀 Start Prediction",
        "reset_button_label": "🔄 Reset Input",
        "generating_results": "✅ Generating Results...",
        "input_confirmation": "Input Confirmation",
        "basic_info_title": "**Basic Info:**",
        "disease_surgery_title": "**Disease and Surgery:**",
        "age_display": "- Age: {age} ",
        "gender_display": "- Gender (Male=1, Female=0): {gender}",
        "alb_display": "- ALB: {alb} g/L",
        "ast_display": "- AST: {ast} U/L",
        "alt_display": "- ALT: {alt} U/L",
        "tbil_display": "- TBIL: {tbil} umol/L",
        "albi_display": "- ALBI Score: {albi}",
        "afp_display": "- AFP: {afp} μg/L",
        "afp_greater_400_display": "- AFP > 400 μg/L: {afp_greater_400}",
        "afp_less_400_display": "- AFP ≤ 400 μg/L: {afp_less_400}",
        "pt_display": "- PT: {pt} s",
        "inr_display": "- INR: {inr}",
        "wbc_display": "- WBC: {wbc} ×10^9/L",
        "blood_loss_display": "- Blood Loss: {blood_loss} mL",
        "tumor_mvi_display": "- Tumor MVI (M0=1, M1 or M2=0): {tumor_mvi}",
        "tumor_diameter_display": "- Tumor Diameter: {tumor_diameter} cm",
        "cirrhosis_display": "- Cirrhosis (present=1, absent=0): {cirrhosis}",
        "capsule_invasion_display": "- Capsule Invasion (present=1, absent=0): {capsule_invasion}",
        "tumor_differentiation_display": "- Tumor Differentiation (Massive=1, Others=0): {tumor_differentiation}",
        "hepatitis_display": "- Hepatitis (present=1, absent=0): {hepatitis}",
        "wide_resection_display": "- Wide Resection (yes=1, no=0): {wide_resection}",
        "child_stage_display": "- Child-Pugh Stage (A=1, B or C=0): {child_stage}",
        "results_subheader": "Results",
        "risk_probability_title": "**Risk Probability of Each Model:**",
        "model_col": "Model",
        "risk_col": "Risk",
        "risk_level_col": "Risk Level",
        "high_risk": "High risk",
        "medium_risk": "Medium risk",
        "low_risk": "Low risk",
        "comparison_chart_title": "Comparison of Risk Probability",
        "chart_x_label": "Model",
        "chart_y_label": "Risk",
        "comprehensive_risk_assessment": "**Comprehensive Risk Assessment:**",
        "overall_risk": "⚠️ Overall Risk: {overall_risk}",
        "avg_risk_prob": "Average Risk Probability: {avg_risk:.3f}",
        "recommendation_high": "Recommendation: Close monitoring and active treatment are advised.",
        "recommendation_medium": "Recommendation: Regular follow-up and careful observation are recommended.",
        "recommendation_low": "Recommendation: Continue monitoring and maintain current status.",
        "model_consistency": "**Model Consistency:**",
        "consistency_high": "Model predictions are highly consistent",
        "consistency_medium": "The model predictions are basically consistent",
        "consistency_low": "Model predictions differ",
        "risk_std_dev": "Risk Standard Deviation: {risk_std:.3f}",
        "survival_analysis_subheader": "Survival analysis",
        "xgboost_rfs_title": "XGBoost Predicted Recurrence-Free Survival",
        "predicted_rfs": "Predicted RFS: {display_value} days",
        "shap_explainability_subheader": "SHAP Explainability",
        "feature_contribution_title": "Feature Contribution (Pre-generated)",
        "shap_waterfall_title": "SHAP Waterfall Plot (Single Patient)",
        "shap_x_label": "SHAP value (impact on predicted recurrence risk)",
        "individual_feature_contribution": "Individual Feature Contribution",
        "result_export_subheader": "Result Export",
        "download_report_button": "📥 Download Prediction Report(CSV)",
        "download_detailed_button": "📥 Download all model predictions (CSV)",
        "report_prediction_time": "Prediction time",
        "report_patient_age": "Age",
        "report_patient_gender": "Gender (1=Male,0=Female)",
        "report_albi_score": "ALBI",
        "report_afp_value": "AFP",
        "report_afp_greater_400": "AFP_greater_400",
        "report_afp_less_400": "AFP_less_400",
        "report_tumor_diameter": "Tumor Diameter (cm)",
        "report_cirrhosis": "Cirrhosis",
        "report_avg_risk": "Average risk",
        "report_overall_risk_level": "Overall risk level",
        "report_model_consistency": "Model Consistency",
        "report_consistency_consistent": "Consistent",
        "report_consistency_divergent": "Divergent",
        "initial_info_message": "Please enter the patient information on the left and click the 'Start Prediction' button",
        "afp_conversion_instructions_subheader": "AFP Conversion Instructions",
        "afp_conversion_rules_title": "**AFP's Automatic Conversion Rules:**",
        "afp_conversion_rule_1": "- When AFP > 400 μg/L：AFP_greater_400 = 1, AFP_less_400 = 0",
        "afp_conversion_rule_2": "- When AFP ≤ 400 μg/L：AFP_greater_400 = 0, AFP_less_400 = 1",
        "afp_conversion_example_title": "**Example:**",
        "afp_conversion_example_1": "- AFP = 500 → AFP_greater_400 = 1, AFP_less_400 = 0",
        "afp_conversion_example_2": "- AFP = 400 → AFP_greater_400 = 0, AFP_less_400 = 1",
        "afp_conversion_example_3": "- AFP = 200 → AFP_greater_400 = 0, AFP_less_400 = 1",
        "rsf_model_load_fail": "⚠️ RSF Model loading failed: {error}",
        "xgboost_model_load_fail": "⚠️ XGBoost Model loading failed: {error}",
        "xgboost_not_installed": "⚠️ XGBoost not installed",
        "cox_model_load_fail": "⚠️ Cox Model loading failed: {error}",
        "lr_model_load_fail": "⚠️ LR Model loading failed: {error}",
        "nn_model_load_fail": "⚠️ NN Model loading failed: {error}",
        "general_error": "❌ Error: {error}",
        "rsf_predict_fail": "RSF model has no available prediction interface",
        "cox_predict_fail": "Cox model has no available prediction interface",
        "missing_feature_order": "Missing training column order for {model_key}, please generate models/feature_order.json first",
        "missing_features": "{model_key} missing features: {missing}",
        "mock_model_fail": "{model_name} failed to load, and random fallback is prohibited. Please check the model file.",
        "waterfall_generation_failed": "Waterfall generation failed: {error}",
        "feature_mapping": {
            '年龄__y': 'Age (Years)',
            '性别_1': 'Gender (Male=1, Female=0)',
            'ALB': 'ALB (g/L)',
            'AST': 'AST (U/L)',
            'ALT': 'ALT (U/L)',
            'TBIL': 'TBIL (umol/L)',
            'AFP': 'AFP (μg/L)',
            'AFP_greater_400': 'AFP > 400',
            'AFP_less_400': 'AFP ≤ 400',
            'PT': 'PT (s)',
            'INR': 'INR',
            'WBC': 'WBC (10^9/L)',
            '失血量': 'Blood Loss (mL)',
            '肿瘤MVI_M0': 'Tumor MVI (M0=1, M1 or M2=0)',
            '肿瘤直径': 'Tumor Diameter (cm)',
            '是否合并肝硬化_1': 'Cirrhosis (present=1, absent=0)',
            '包膜是否受侵犯_未浸及': 'Capsule Invasion (present=1, absent=0)',
            '肿瘤是否巨块型分化_1': 'Tumor Differentiation (Massive=1, Others=0)',
            '是否合并肝炎_1': 'Hepatitis (present=1, absent=0)',
            '是否大范围切除_1': 'Wide Resection (yes=1, no=0)',
            'child分期_A': 'Child-Pugh Stage (A=1, B or C=0)',
            'ALBI': 'ALBI Score'
        }
    }
}

# --- Streamlit Session State Initialization ---
if "language" not in st.session_state:
    st.session_state.language = "中文"  # Default language

if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

current_lang = LANGUAGES[st.session_state.language]

# --- Language Selection UI ---
with st.sidebar:
    language_selection = st.radio("选择语言 / Select Language", ["中文", "English"])
    if language_selection != st.session_state.language:
        st.session_state.language = language_selection
        st.rerun()

# --- Default Inputs ---
DEFAULT_INPUTS = {
    "age": 47, "gender": 1, "alb": 42.3, "ast": 30.0, "tbil": 11.0, "albi": -2.91,
    "alt": 42.0, "afp": 600.0, "pt": 11.5, "inr": 0.97, "wbc": 8.88, "blood_loss": 500,
    "wide_resection": 1, "tumor_mvi": 0, "tumor_diameter": 4.0, "hepatitis": 1,
    "cirrhosis": 0, "capsule_invasion": 1, "tumor_differentiation": 0, "child_stage": 1,
}

# --- Random Seed Setting ---
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    if 'torch' in globals() and TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed(42)

# --- DeepMLP Class ---
class DeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),      nn.BatchNorm1d(32),  nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

if 'main' in sys.modules:
    setattr(sys.modules['main'], 'DeepMLP', DeepMLP)
else:
    _m = types.ModuleType('main')
    _m.DeepMLP = DeepMLP
    sys.modules['main'] = _m

# --- Helper Function for Error Handling ---
def _fail(msg: str):
    st.error(msg)
    raise RuntimeError(msg)

warnings.filterwarnings('ignore')

# --- XGBoost and PyTorch Availability Check ---
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning(current_lang["xgboost_not_installed"])

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch is not installed.")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=current_lang["page_title"],
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    .stButton > button {
        width: 100%;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HCCSurvivalPredictor Class ---
class HCCSurvivalPredictor:
    def __init__(self):
        self.models = {}
        self.time_horizon = 730.0
        self.load_models()

    def _coerce_estimator(self, obj):
        m = obj
        if hasattr(m, "best_estimator_"):
            m = m.best_estimator_
        if hasattr(m, "steps"):
            m = m.steps[-1][1]
        if hasattr(m, "base_estimator"):
            m = m.base_estimator
        if isinstance(m, dict):
            for k in ("model", "estimator", "best_estimator_", "final_estimator"):
                if k in m:
                    return self._coerce_estimator(m[k])
            for v in m.values():
                e = self._coerce_estimator(v)
                if hasattr(e, "predict") or hasattr(e, "predict_proba") or hasattr(e, "forward"):
                    return e
            return m
        if isinstance(m, (list, tuple)):
            for v in m:
                e = self._coerce_estimator(v)
                if hasattr(e, "predict") or hasattr(e, "predict_proba") or hasattr(e, "forward"):
                    return e
            return m
        return m

    def _survival_at(self, surv_obj, t):
        import numpy as np
        import pandas as pd
        if callable(surv_obj):
            return float(surv_obj(t))
        if isinstance(surv_obj, pd.DataFrame):
            if surv_obj.shape[1] == 0:
                return float('nan')
            surv_obj = surv_obj.iloc[:, 0]
        if isinstance(surv_obj, pd.Series):
            times = np.asarray(surv_obj.index, dtype=float)
            vals = np.asarray(surv_obj.values, dtype=float)
            if times.size == 0:
                return float('nan')
            idx = np.searchsorted(times, t, side='right') - 1
            idx = max(0, min(idx, len(vals) - 1))
            return float(vals[idx])
        if isinstance(surv_obj, (list, np.ndarray)):
            arr = np.asarray(surv_obj, dtype=float)
            if arr.size == 0:
                return float('nan')
            return float(arr[-1])
        try:
            return float(surv_obj)
        except Exception:
            return float('nan')

    def _pos_proba(self, model, proba: np.ndarray) -> float:
        if proba.ndim == 1:
            return float(proba[0])
        idx = 1
        if hasattr(model, "classes_"):
            cls = list(model.classes_)
            if 1 in cls:
                idx = cls.index(1)
            elif True in cls:
                idx = cls.index(True)
        return float(proba[0, idx])

    def _rsf_prob(self, model, X: pd.DataFrame, time_horizon: float) -> float:
        if hasattr(model, "predict_survival_function"):
            surv_funcs = model.predict_survival_function(X)
            s = self._survival_at(surv_funcs[0], time_horizon)
            return float(1.0 - s)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return self._pos_proba(model, proba)
        if hasattr(model, "predict"):
            yhat = model.predict(X)
            val = float(yhat[0]) if hasattr(yhat, "__getitem__") else float(yhat)
            return float(1.0 / (1.0 + np.exp(-val)))
        _fail(current_lang["rsf_predict_fail"])

    def _cox_prob(self, model, X: pd.DataFrame, time_horizon: float) -> float:
        if hasattr(model, "predict_survival_function"):
            surv_funcs = model.predict_survival_function(X, times=[time_horizon])
            probs = 1.0 - surv_funcs.iloc[0].values
            return probs
        _fail(current_lang["cox_predict_fail"])

    def ensure_feature_orders(self, sample_df: pd.DataFrame):
        temp_dir = os.path.join(tempfile.gettempdir(), 'hcc_predictor')
        json_path = os.path.join(temp_dir, 'feature_order.json')
        need_build = True
        orders = {}
        
        bundled_json_path = get_resource_path('models/feature_order.json')
        if os.path.exists(bundled_json_path):
            try:
                orders = load_feature_orders(bundled_json_path)
                keys_needed = [k for k in ['cox', 'rsf', 'logistic', 'deepsurv', 'xgboost'] if
                               self.models.get(k) is not None]
                if all(k in orders for k in keys_needed):
                    need_build = False
                    json_path = bundled_json_path
            except Exception:
                need_build = True
        
        if need_build:
            if os.path.exists(json_path):
                try:
                    orders = load_feature_orders(json_path)
                    keys_needed = [k for k in ['cox', 'rsf', 'logistic', 'deepsurv', 'xgboost'] if
                                   self.models.get(k) is not None]
                    if all(k in orders for k in keys_needed):
                        need_build = False
                except Exception:
                    need_build = True

            if need_build:
                orders = build_and_save_feature_orders(self.models, json_path, sample_df)

        self.feature_orders = orders

    def _align_X(self, df: pd.DataFrame, model, model_key: str) -> pd.DataFrame:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            if not hasattr(self, "feature_orders") or model_key not in self.feature_orders:
                raise RuntimeError(current_lang["missing_feature_order"].format(model_key=model_key))
            cols = self.feature_orders[model_key]

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(current_lang["missing_features"].format(model_key=model_key, missing=missing))
        X = df[cols].astype("float32")
        return X

    def _create_feature_mapping(self):
        return current_lang["feature_mapping"]

    def _create_reverse_mapping(self):
        return {v: k for k, v in self._create_feature_mapping().items()}

    def _convert_afp_features(self, afp_value):
        if afp_value > 400:
            return {'AFP_greater_400': 1, 'AFP_less_400': 0}
        else:
            return {'AFP_greater_400': 0, 'AFP_less_400': 1}

    def load_models(self):
        try:
            try:
                try:
                    self.models['rsf'] = joblib.load(get_resource_path('models/newbest_rsf_model.pkl'))
                except:
                    with open(get_resource_path('models/newbest_rsf_model.pkl'), 'rb') as f:
                        self.models['rsf'] = pickle.load(f)
            except Exception as e:
                st.sidebar.warning(current_lang["rsf_model_load_fail"].format(error=str(e)))
                self.models['rsf'] = self.create_mock_model('RSF')
            if XGBOOST_AVAILABLE:
                try:
                    self.models['xgboost'] = xgb.Booster()
                    self.models['xgboost'].load_model(get_resource_path('models/aft_model.ubj'))
                except Exception as e:
                    st.sidebar.warning(current_lang["xgboost_model_load_fail"].format(error=str(e)))
                    self.models['xgboost'] = self.create_mock_model('XGBoost')
            else:
                st.sidebar.warning(current_lang["xgboost_not_installed"])
                self.models['xgboost'] = self.create_mock_model('XGBoost')

            try:
                self.models['cox'] = joblib.load(get_resource_path('models/cox_model.pkl'))
            except Exception as e:
                st.sidebar.warning(current_lang["cox_model_load_fail"].format(error=str(e)))
                self.models['cox'] = self.create_mock_model('Cox')

            try:
                self.models['logistic'] = joblib.load(get_resource_path('models/logistic_model.pkl'))
            except Exception as e:
                st.sidebar.warning(current_lang["lr_model_load_fail"].format(error=str(e)))
                self.models['logistic'] = self.create_mock_model('logistic')

            try:
                self.models['deepsurv'] = joblib.load(get_resource_path('models/deepsurv_model.joblib'))
            except Exception as e:
                st.sidebar.warning(current_lang["nn_model_load_fail"].format(error=str(e)))
                self.models['deepsurv'] = self.create_mock_model('deepsurv')

        except Exception as e:
            st.error(current_lang["general_error"].format(error=str(e)))

    def create_mock_model(self, model_name):
        class MockModel:
            def predict_proba(self, X):
                _fail(current_lang["mock_model_fail"].format(model_name=model_name))

            def predict(self, X):
                _fail(current_lang["mock_model_fail"].format(model_name=model_name))

        return MockModel(model_name)

    def preprocess_input(self, input_data):
        return input_data.copy()

    def predict_survival(self, input_data):
        processed_data = self.preprocess_input(input_data)
        self.ensure_feature_orders(processed_data)

        predictions = {}

        if self.models.get('rsf') is not None:
            mr = self.models['rsf']
            X = self._align_X(processed_data, mr, "rsf")
            prob = self._rsf_prob(mr, X, self.time_horizon)
            predictions['Random Survival Forest'] = np.clip(float(prob), 0.0, 1.0)

        if self.models.get('cox') is not None:
            mc = self.models['cox']
            X = self._align_X(processed_data, mc, "cox")
            prob = self._cox_prob(mc, X, self.time_horizon)
            predictions['Cox-PH'] = np.clip(float(prob), 0.0, 1.0)

        if self.models.get("xgboost") is not None:
            m_xgb = self.models["xgboost"]
            scaler = joblib.load(get_resource_path("xgb_scaler.pkl"))
            feature_cols = ['PT',
                            'child分期_A', 'AFP_greater_400',
                            '失血量',
                            '肿瘤是否巨块型分化_1', 'AST', 'ALT', 'WBC', 'INR', 'TBIL',
                            '年龄__y',
                            '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                            'ALB', '是否合并肝硬化_1',
                            '是否大范围切除_1',
                            '是否合并肝炎_1', '肿瘤MVI_M0',
                            'AFP_less_400']
            X_valid = processed_data[feature_cols]
            X_aligned = scaler.transform(X_valid)
            dmatrix = xgb.DMatrix(X_aligned)
            pred_time = m_xgb.predict(dmatrix)
            display_value = float(pred_time[0])
            predictions["XGBoost"] = display_value

        if self.models.get("logistic") is not None:
            clf, scaler = self.models["logistic"]
            available_features = [
                'PT', 'child分期_A', 'AFP_greater_400', '失血量',
                '肿瘤是否巨块型分化_1', 'AST', 'ALT', 'WBC', 'INR', 'TBIL',
                '年龄__y', '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                'ALB', '是否合并肝硬化_1', '是否大范围切除_1', '是否合并肝炎_1',
                '肿瘤MVI_M0', 'AFP_less_400'
            ]
            X_aligned = processed_data[available_features].values
            X_scaled = scaler.transform(X_aligned)
            proba = clf.predict_proba(X_scaled)[:, 1]
            pred_proba = float(proba[0])
            display_value_lr = float(pred_proba)
            predictions["Logistic Regression"] = display_value_lr

        if self.models.get("deepsurv") is not None:
            wrapper = self.models["deepsurv"]
            available_features = [
                'PT', 'child分期_A', 'AFP_greater_400', '失血量',
                '肿瘤是否巨块型分化_1', 'AST', 'ALT', 'WBC', 'INR', 'TBIL',
                '年龄__y', '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                'ALB', '是否合并肝硬化_1', '是否大范围切除_1', '是否合并肝炎_1',
                '肿瘤MVI_M0', 'AFP_less_400'
            ]
            X_aligned = processed_data[available_features].values
            probs = wrapper.predict_proba(X_aligned)[0]
            display_value_nn = float(probs[1])
            predictions["Neural Network"] = display_value_nn

        return predictions, display_value, processed_data, mr, mc, m_xgb, wrapper, clf

@st.cache_resource
def load_predictor():
    return HCCSurvivalPredictor()

predictor = load_predictor()

# --- Main UI ---
st.markdown(current_lang["main_title"], unsafe_allow_html=True)
st.markdown(current_lang["disclaimer"], unsafe_allow_html=True)

if st.session_state.reset_flag:
    for k, v in DEFAULT_INPUTS.items():
        st.session_state[k] = v
    st.session_state.reset_flag = False

for k, v in DEFAULT_INPUTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

with st.sidebar:
    st.header(current_lang["sidebar_header_info"])
    st.subheader(current_lang["sidebar_subheader_basic"])

    age = st.number_input(current_lang["age_label"], min_value=18, max_value=100, key="age")
    gender = st.number_input(current_lang["gender_label"],
                             min_value=0, max_value=1, step=1, key="gender")

    st.subheader(current_lang["sidebar_subheader_lab"])
    alb = st.number_input(current_lang["alb_label"], min_value=20.0, max_value=60.0, step=0.1, key="alb")
    ast = st.number_input(current_lang["ast_label"], min_value=5.0, max_value=500.0, step=1.0, key="ast")
    tbil = st.number_input(current_lang["tbil_label"], min_value=2.0, max_value=200.0, step=0.1, key="tbil")
    albi = st.number_input(current_lang["albi_label"], min_value=-3.0, max_value=0.0, step=0.1, key="albi")
    alt = st.number_input(current_lang["alt_label"], min_value=5.0, max_value=500.0, step=1.0, key="alt")
    afp = st.number_input(current_lang["afp_label"], min_value=0.0, max_value=10000.0, step=1.0, key="afp")

    afp_features = predictor._convert_afp_features(st.session_state["afp"])

    pt = st.number_input(current_lang["pt_label"], min_value=8.0, max_value=30.0, step=0.1, key="pt")
    inr = st.number_input(current_lang["inr_label"], min_value=0.8, max_value=5.0, step=0.1, key="inr")
    wbc = st.number_input(current_lang["wbc_label"], min_value=1.0, max_value=50.0, step=0.1, key="wbc")

    st.subheader(current_lang["sidebar_subheader_surgery"])
    blood_loss = st.number_input(current_lang["blood_loss_label"], min_value=0, max_value=5000, step=50, key="blood_loss")
    wide_resection = st.number_input(current_lang["wide_resection_label"],
                                     min_value=0, max_value=1, step=1, key="wide_resection")

    st.subheader(current_lang["sidebar_subheader_pathology"])
    tumor_mvi = st.number_input(current_lang["tumor_mvi_label"],
                                min_value=0, max_value=1, step=1, key="tumor_mvi")

    tumor_diameter = st.number_input(current_lang["tumor_diameter_label"],
                                     min_value=0.1, max_value=20.0, step=0.1, key="tumor_diameter")

    hepatitis = st.number_input(current_lang["hepatitis_label"],
                                min_value=0, max_value=1, step=1, key="hepatitis")

    cirrhosis = st.number_input(current_lang["cirrhosis_label"],
                                min_value=0, max_value=1, step=1, key="cirrhosis")

    capsule_invasion = st.number_input(current_lang["capsule_invasion_label"],
                                       min_value=0, max_value=1, step=1, key="capsule_invasion")

    tumor_differentiation = st.number_input(
        current_lang["tumor_differentiation_label"],
        min_value=0, max_value=1, step=1, key="tumor_differentiation")

    child_stage = st.number_input(
        current_lang["child_stage_label"],
        min_value=0, max_value=1, step=1, key="child_stage")

    predict_button = st.button(current_lang["predict_button_label"], type="primary", use_container_width=True)

    if st.sidebar.button(current_lang["reset_button_label"], use_container_width=True):
        st.session_state.reset_flag = True
        st.rerun()

if predict_button:
    st.success(current_lang["generating_results"])
    input_data = pd.DataFrame({
        '年龄__y': [age],
        '性别_1': [gender],
        'ALB': [alb],
        'AST': [ast],
        'ALT': [alt],
        'TBIL': [tbil],
        'PT': [pt],
        'INR': [inr],
        'WBC': [wbc],
        'ALBI': [albi],
        '失血量': [blood_loss],
        '肿瘤MVI_M0': [tumor_mvi],
        '肿瘤直径': [tumor_diameter],
        '是否合并肝硬化_1': [cirrhosis],
        '包膜是否受侵犯_未浸及': [capsule_invasion],
        '肿瘤是否巨块型分化_1': [tumor_differentiation],
        '是否合并肝炎_1': [hepatitis],
        '是否大范围切除_1': [wide_resection],
        'child分期_A': [child_stage]
    })
    afp_features = predictor._convert_afp_features(afp)
    input_data['AFP_greater_400'] = [afp_features['AFP_greater_400']]
    input_data['AFP_less_400'] = [afp_features['AFP_less_400']]
    st.subheader(current_lang["input_confirmation"])
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(current_lang["basic_info_title"])
        st.write(current_lang["age_display"].format(age=age))
        st.write(current_lang["gender_display"].format(gender=gender))
        st.write(f"- ALB: {alb} g/L")
        st.write(f"- AST: {ast} U/L")
        st.write(f"- ALT: {alt} U/L")
        st.write(f"- TBIL: {tbil} umol/L")
        st.write(current_lang["albi_display"].format(albi=albi))
        st.write(f"- AFP: {afp} μg/L")
        st.write(current_lang["afp_greater_400_display"].format(afp_greater_400=afp_features['AFP_greater_400']))
        st.write(current_lang["afp_less_400_display"].format(afp_less_400=afp_features['AFP_less_400']))
        st.write(current_lang["pt_display"].format(pt=pt))
        st.write(current_lang["inr_display"].format(inr=inr))

    with col2:
        st.markdown(current_lang["disease_surgery_title"])
        st.write(current_lang["wbc_display"].format(wbc=wbc))
        st.write(current_lang["blood_loss_display"].format(blood_loss=blood_loss))
        st.write(current_lang["tumor_mvi_display"].format(tumor_mvi=tumor_mvi))
        st.write(current_lang["tumor_diameter_display"].format(tumor_diameter=tumor_diameter))
        st.write(current_lang["cirrhosis_display"].format(cirrhosis=cirrhosis))
        st.write(current_lang["capsule_invasion_display"].format(capsule_invasion=capsule_invasion))
        st.write(current_lang["tumor_differentiation_display"].format(tumor_differentiation=tumor_differentiation))
        st.write(current_lang["hepatitis_display"].format(hepatitis=hepatitis))
        st.write(current_lang["wide_resection_display"].format(wide_resection=wide_resection))
        st.write(current_lang["child_stage_display"].format(child_stage=child_stage))

    st.markdown("---")

    st.subheader(current_lang["results_subheader"])

    predictions, display_value, processed_data, mr, mc, m_xgb, wrapper, ml = predictor.predict_survival(input_data)
    if "XGBoost" in predictions:
        del predictions["XGBoost"]
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(current_lang["risk_probability_title"])

        def get_risk_level(risk_prob):
            if risk_prob >= 0.7:
                return current_lang["high_risk"]
            elif risk_prob >= 0.4:
                return current_lang["medium_risk"]
            else:
                return current_lang["low_risk"]

        results_df = pd.DataFrame({
            current_lang["model_col"]: list(predictions.keys()),
            current_lang["risk_col"]: list(predictions.values()),
            current_lang["risk_level_col"]: [get_risk_level(risk) for risk in predictions.values()]
        })

        def color_risk(val):
            if val == current_lang["high_risk"]:
                return 'background-color: #ffcdd2'
            elif val == current_lang["medium_risk"]:
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #c8e6c9'

        styled_df = results_df.style.applymap(
            lambda x: color_risk(x) if isinstance(x, str) else '',
            subset=[current_lang["risk_level_col"]]
        )
        st.dataframe(styled_df, use_container_width=True)

        fig = px.bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            title=current_lang["comparison_chart_title"],
            labels={'x': current_lang["chart_x_label"], 'y': current_lang["chart_y_label"]},
            color=list(predictions.values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_traces(hovertemplate="Model: %{x}<br>Risk: %{y:.6f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(current_lang["comprehensive_risk_assessment"])
        avg_risk = np.mean(list(predictions.values()))
        overall_risk = get_risk_level(avg_risk)

        if overall_risk == current_lang["high_risk"]:
            st.error(current_lang["overall_risk"].format(overall_risk=overall_risk))
            st.write(current_lang["avg_risk_prob"].format(avg_risk=avg_risk))
            st.write(current_lang["recommendation_high"])
        elif overall_risk == current_lang["medium_risk"]:
            st.warning(current_lang["overall_risk"].format(overall_risk=overall_risk))
            st.write(current_lang["avg_risk_prob"].format(avg_risk=avg_risk))
            st.write(current_lang["recommendation_medium"])
        else:
            st.success(f"✅ {current_lang['overall_risk'].format(overall_risk=overall_risk)}")
            st.write(current_lang["avg_risk_prob"].format(avg_risk=avg_risk))
            st.write(current_lang["recommendation_low"])

        st.markdown(current_lang["model_consistency"])
        risk_std = np.std(list(predictions.values()))
        if risk_std < 0.1:
            st.success(current_lang["consistency_high"])
        elif risk_std < 0.2:
            st.warning(current_lang["consistency_medium"])
        else:
            st.error(current_lang["consistency_low"])

        st.write(current_lang["risk_std_dev"].format(risk_std=risk_std))

    st.markdown("---")

    st.subheader(current_lang["survival_analysis_subheader"])

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.session_state.language == "中文":
            from survcurve_cn import plot_survival_curves
        else:
            from survcurve import plot_survival_curves
        muse = joblib.load(get_resource_path('models/deepsurv_strict.joblib'))
        fig = plot_survival_curves(
            processed_data,
            cox_model=mc,
            rsf_model=mr,
            xgb_aft_model=m_xgb,
            deepsurv_model=muse
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if predictor.models.get("xgboost") is not None:
            st.subheader(current_lang["xgboost_rfs_title"])
            st.info(current_lang["predicted_rfs"].format(display_value=int(display_value)))

    st.markdown("---")
    import shap
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    rsf_model = joblib.load(get_resource_path('models/newbest_rsf_model.pkl'))

    st.subheader(current_lang["shap_explainability_subheader"])

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(current_lang["feature_contribution_title"])
        st.image("shap_summary_bar.png", use_container_width=True)
    with col_right:
        st.markdown(current_lang["shap_waterfall_title"])
        try:
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
            plt.rcParams["font.family"] = "DejaVu Sans"
            plt.rcParams["axes.unicode_minus"] = False

            @st.cache_data
            def load_train_data_aa():
                return pd.read_csv("datahx1.csv")
            df_bg = load_train_data_aa()

            if hasattr(rsf_model, "feature_names_in_"):
                model_features = list(rsf_model.feature_names_in_)
            else:
                model_features = df_bg.columns.tolist()
            df_bg = df_bg[model_features].applymap(lambda x: pd.to_numeric(x, errors='coerce'))

            df_bg_sample = df_bg.sample(n=min(20, len(df_bg)), random_state=42)

            df_single = processed_data.copy()

            for f in model_features:
                if f not in df_single.columns:
                    df_single[f] = np.nan

            df_single = df_single[model_features].applymap(lambda x: pd.to_numeric(x, errors='coerce'))
            row = df_single.iloc[[0]]

            TIME_POINT = predictor.time_horizon

            @st.cache_data
            def predict_fn(df):
                surv = rsf_model.predict_survival_function(df)
                return np.array([1 - fn(TIME_POINT) for fn in surv])

            @st.cache_resource
            def get_explainer():
                return shap.PermutationExplainer(predict_fn, df_bg_sample)
            explainer = get_explainer()
            shap_values_single = explainer(row)

            shap_raw = shap_values_single.values
            shap_vals = np.array(shap_raw, dtype=float).reshape(-1)

            abs_vals = np.abs(shap_vals)
            order = np.argsort(abs_vals)[::-1]
            idx_top = order[:8]

            shap_vals_top = shap_vals[idx_top]
            features_top = [model_features[i] for i in idx_top]

            # Always use English feature names for SHAP waterfall plot
            feature_mapping_eng = {
                '年龄__y': 'Age (Years)',
                '性别_1': 'Gender (Male=1, Female=0)',
                'ALB': 'ALB (g/L)',
                'AST': 'AST (U/L)',
                'ALT': 'ALT (U/L)',
                'TBIL': 'TBIL (umol/L)',
                'AFP': 'AFP (μg/L)',
                'AFP_greater_400': 'AFP > 400',
                'AFP_less_400': 'AFP ≤ 400',
                'PT': 'PT (s)',
                'INR': 'INR',
                'WBC': 'WBC (10^9/L)',
                '失血量': 'Blood Loss (mL)',
                '肿瘤MVI_M0': 'Tumor MVI (M0=1, M1 or M2=0)',
                '肿瘤直径': 'Tumor Diameter (cm)',
                '是否合并肝硬化_1': 'Cirrhosis (present=1, absent=0)',
                '包膜是否受侵犯_未浸及': 'Capsule Invasion (present=1, absent=0)',
                '肿瘤是否巨块型分化_1': 'Tumor Differentiation (Massive=1, Others=0)',
                '是否合并肝炎_1': 'Hepatitis (present=1, absent=0)',
                '是否大范围切除_1': 'Wide Resection (yes=1, no=0)',
                'child分期_A': 'Child-Pugh Stage (A=1, B or C=0)',
                'ALBI': 'ALBI Score'
            }
            
            feature_names_eng = [
                feature_mapping_eng.get(f, f) for f in features_top
            ]

            def fig_to_pil(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
                buf.seek(0)
                return Image.open(buf), buf

            colors = ["#d62728" if v > 0 else "#1f77b4" for v in shap_vals_top]

            fig, ax = plt.subplots(figsize=(7, 0.45 * len(feature_names_eng) + 2))

            y = np.arange(len(feature_names_eng))
            ax.barh(y, shap_vals_top, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(feature_names_eng)
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel("SHAP value (impact on predicted recurrence risk)")

            plt.title("Individual Feature Contribution")
            fig.tight_layout()

            img_wf, buf_wf = fig_to_pil(fig)
            plt.close(fig)
            import gc
            gc.collect()

            st.image(img_wf, use_container_width=True)

        except Exception as e:
            st.error(current_lang["waterfall_generation_failed"].format(error=str(e)))

    st.subheader(current_lang["result_export_subheader"])

    report_data = {
        current_lang["report_prediction_time"]: [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        current_lang["report_patient_age"]: [age],
        current_lang["report_patient_gender"]: [gender],
        current_lang["report_albi_score"]: [albi],
        current_lang["report_afp_value"]: [afp],
        current_lang["report_afp_greater_400"]: [afp_features['AFP_greater_400']],
        current_lang["report_afp_less_400"]: [afp_features['AFP_less_400']],
        current_lang["report_tumor_diameter"]: [tumor_diameter],
        current_lang["report_cirrhosis"]: [cirrhosis],
        current_lang["report_avg_risk"]: [avg_risk],
        current_lang["report_overall_risk_level"]: [overall_risk],
        current_lang["report_model_consistency"]: [current_lang["report_consistency_consistent"] if risk_std < 0.1 else current_lang["report_consistency_divergent"]]
    }

    report_df = pd.DataFrame(report_data)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label=current_lang["download_report_button"],
            data=report_df.to_csv(index=False),
            file_name=f"hcc_survival_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        detailed_results = pd.DataFrame({
            current_lang["model_col"]: list(predictions.keys()),
            current_lang["risk_col"]: list(predictions.values()),
            current_lang["risk_level_col"]: [get_risk_level(risk) for risk in predictions.values()]
        })

        st.download_button(
            label=current_lang["download_detailed_button"],
            data=detailed_results.to_csv(index=False),
            file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info(current_lang["initial_info_message"])
    st.subheader(current_lang["afp_conversion_instructions_subheader"])
    st.write(current_lang["afp_conversion_rules_title"])
    st.write(current_lang["afp_conversion_rule_1"])
    st.write(current_lang["afp_conversion_rule_2"])
    st.write(current_lang["afp_conversion_example_title"])
    st.write(current_lang["afp_conversion_example_1"])
    st.write(current_lang["afp_conversion_example_2"])
    st.write(current_lang["afp_conversion_example_3"])
