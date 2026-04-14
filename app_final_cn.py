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
DEFAULT_INPUTS = {
    "age": 47,
    "gender": 1,
    "alb": 42.3,
    "ast": 30.0,
    "tbil": 11.0,
    "albi": -2.91,
    "alt": 42.0,
    "afp": 600.0,
    "pt": 11.5,
    "inr": 0.97,
    "wbc": 8.88,
    "blood_loss": 500,
    "wide_resection": 1,
    "tumor_mvi": 0,
    "tumor_diameter": 4.0,
    "hepatitis": 1,
    "cirrhosis": 0,
    "capsule_invasion": 1,
    "tumor_differentiation": 0,
    "child_stage": 1,
}

if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False
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
def _fail(msg: str):
    st.error(msg)
    raise RuntimeError(msg)
warnings.filterwarnings('ignore')
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost is not installed.")
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch is not installed.")
st.set_page_config(
    page_title="HCC Recurrence-Free Survival Analysis Prediction System",
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
class HCCSurvivalPredictor:

    def __init__(self):
        self.models = {}
        self.feature_mapping = self._create_feature_mapping()
        self.reverse_mapping = self._create_reverse_mapping()
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
        _fail("RSF模型没有可用的预测接口")

    def _cox_prob(self, model, X: pd.DataFrame, time_horizon: float) -> float:
        if hasattr(model, "predict_survival_function"):
            surv_funcs = model.predict_survival_function(X, times=[time_horizon])
            probs = 1.0 - surv_funcs.iloc[0].values
            return probs
        _fail("Cox模型没有可用的预测接口")

    def ensure_feature_orders(self, sample_df: pd.DataFrame):
        json_path = "models/feature_order.json"
        need_build = True
        orders = {}
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
                raise RuntimeError(f"缺少 {model_key} 的训练列顺序，请先生成 models/feature_order.json")
            cols = self.feature_orders[model_key]

        missing = [c for c in cols if c not in df.columns]
        extra = [c for c in df.columns if c not in cols]
        if missing:
            raise RuntimeError(f"{model_key} 缺失特征: {missing}")
        X = df[cols].astype("float32")
        return X


    def _create_feature_mapping(self):
        return {
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

    def _create_reverse_mapping(self):
        return {v: k for k, v in self.feature_mapping.items()}

    def _convert_afp_features(self, afp_value):
        if afp_value > 400:
            return {'AFP_greater_400': 1, 'AFP_less_400': 0}
        else:
            return {'AFP_greater_400': 0, 'AFP_less_400': 1}

    def load_models(self):
        try:
            # 加载随机生存森林模型
            try:
                try:
                    self.models['rsf'] = joblib.load('models/newbest_rsf_model.pkl')
                except:
                    with open('models/newbest_rsf_model.pkl', 'rb') as f:
                        self.models['rsf'] = pickle.load(f)
            except Exception as e:
                st.sidebar.warning(f"⚠️ RSF Model loading failed: {str(e)}")
                self.models['rsf'] = self.create_mock_model('RSF')
            if XGBOOST_AVAILABLE:
                        try:
                            self.models['xgboost'] = xgb.Booster()
                            self.models['xgboost'].load_model('models/aft_model.ubj')
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ XGBoost Model loading failed: {str(e)}")
                            self.models['xgboost'] = self.create_mock_model('XGBoost')
            else:
                        st.sidebar.warning("⚠️ XGBoost not installed")
                        self.models['xgboost'] = self.create_mock_model('XGBoost')

            try:
                    self.models['cox'] = joblib.load('models/cox_model.pkl')

            except Exception as e:
                st.sidebar.warning(f"⚠️ Cox Model loading failed: {str(e)}")
                self.models['cox'] = self.create_mock_model('Cox')

            try:
                    self.models['logistic'] = joblib.load('models/logistic_model.pkl')

            except Exception as e:
                st.sidebar.warning(f"⚠️ LR Model loading failed: {str(e)}")
                self.models['logistic'] = self.create_mock_model('logistic')

            try:
                    self.models['deepsurv'] = joblib.load('models/deepsurv_model.joblib')

            except Exception as e:
                st.sidebar.warning(f"⚠️ NN Model loading failed: {str(e)}")
                self.models['deepsurv'] = self.create_mock_model('deepsurv')

        except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    def create_mock_model(self, model_name):
        class MockModel:
            def predict_proba(self, X):
                _fail(f"{model_name} 未加载成功，且禁止使用随机兜底。请检查模型文件。")

            def predict(self, X):
                _fail(f"{model_name} 未加载成功，且禁止使用随机兜底。请检查模型文件。")

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
            # 假设 self.baseline_surv 是已经计算好的 S0(time_horizon)
            prob = self._cox_prob(mc,X,self.time_horizon)
            predictions['Cox-PH'] = np.clip(float(prob), 0.0, 1.0)

        if self.models.get("xgboost") is not None:
            m_xgb = self.models["xgboost"]
            scaler = joblib.load("xgb_scaler.pkl")
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
            probs = wrapper.predict_proba(X_aligned)[0]  # 可能是一个曲线 (array)
            display_value_nn = float(probs[1])
            predictions["Neural Network"] = display_value_nn

        return predictions, display_value, processed_data, mr, mc, m_xgb, wrapper,clf
    # def _create_feature_mapping(self):
    #     return {
    #         '年龄__y': '年龄',
    #         '性别_1': '性别（男=1，女=0）',
    #         'ALB': 'ALB (g/L)',
    #         'AST': 'AST (U/L)',
    #         'ALT': 'ALT (U/L)',
    #         'TBIL': 'TBIL (umol/L)',
    #         'AFP': 'AFP (μg/L)',
    #         'AFP_greater_400': 'AFP > 400',
    #         'AFP_less_400': 'AFP ≤ 400',
    #         'PT': 'PT (秒)',
    #         'INR': 'INR',
    #         'WBC': 'WBC (10^9/L)',
    #         '失血量': '术中失血量（mL）',
    #         '肿瘤MVI_M0': '肿瘤微血管侵犯（M0=1，M1或M2=0）',
    #         '肿瘤直径': '肿瘤直径（cm）',
    #         '是否合并肝硬化_1': '肝硬化（是=1，否=0）',
    #         '包膜是否受侵犯_未浸及': '包膜侵犯（是=1, 否=0）',
    #         '肿瘤是否巨块型分化_1': '肿瘤分化 (巨块型=1, 结节型或其他=0）',
    #         '是否合并肝炎_1': '肝炎（是=1, 否=0）',
    #         '是否大范围切除_1': '扩大切除 （是=1，否=0）',
    #         'child分期_A': 'Child-Pugh分期（A=1, B或C=0）',
    #         'ALBI': 'ALBI评分'
    #     }
@st.cache_resource
def load_predictor():
    return HCCSurvivalPredictor()


predictor = load_predictor()

st.markdown("<h1 style='text-align: center;'>肝细胞癌无复发生存分析预测系统</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: red;'>声明：仅作为研究目的</h3>", unsafe_allow_html=True)
if st.session_state.reset_flag:
    for k, v in DEFAULT_INPUTS.items():
        st.session_state[k] = v
    st.session_state.reset_flag = False

for k, v in DEFAULT_INPUTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
with st.sidebar:
    st.header("📊 患者信息输入")
    st.subheader("基本信息")

    age = st.number_input("年龄", min_value=18, max_value=100, key="age")
    gender = st.number_input("性别（男=1，女=0）",
                             min_value=0, max_value=1, step=1, key="gender")

    st.subheader("实验室检查指标")
    alb = st.number_input("白蛋白 ALB (g/L)", min_value=20.0, max_value=60.0, step=0.1, key="alb")
    ast = st.number_input("天门冬氨酸氨基转移酶 AST (U/L)", min_value=5.0, max_value=500.0, step=1.0, key="ast")
    tbil = st.number_input("总胆红素 TBIL (umol/L)", min_value=2.0, max_value=200.0, step=0.1, key="tbil")
    albi = st.number_input("ALBI评分", min_value=-3.0, max_value=0.0, step=0.1, key="albi")
    alt = st.number_input("丙氨酸氨基转移酶 ALT (U/L)", min_value=5.0, max_value=500.0, step=1.0, key="alt")
    afp = st.number_input("甲胎蛋白 AFP (μg/L)", min_value=0.0, max_value=10000.0, step=1.0, key="afp")

    afp_features = predictor._convert_afp_features(st.session_state["afp"])

    pt = st.number_input("凝血酶原时间 (s)", min_value=8.0, max_value=30.0, step=0.1, key="pt")
    inr = st.number_input("国际标准化比值 INR", min_value=0.8, max_value=5.0, step=0.1, key="inr")
    wbc = st.number_input("白细胞计数 WBC (10^9/L)", min_value=1.0, max_value=50.0, step=0.1, key="wbc")

    st.subheader("术中指标")
    blood_loss = st.number_input("术中失血量 (mL)", min_value=0, max_value=5000, step=50, key="blood_loss")
    wide_resection = st.number_input("扩大切除 （是=1，否=0）",
                                     min_value=0, max_value=1, step=1, key="wide_resection")

    st.subheader("病理指标")
    tumor_mvi = st.number_input("肿瘤微血管侵犯（M0=1，M1或M2=0）",
                                min_value=0, max_value=1, step=1, key="tumor_mvi")

    tumor_diameter = st.number_input("肿瘤直径（cm）",
                                     min_value=0.1, max_value=20.0, step=0.1, key="tumor_diameter")

    hepatitis = st.number_input("肝炎（是=1, 否=0）",
                                min_value=0, max_value=1, step=1, key="hepatitis")

    cirrhosis = st.number_input("肝硬化（是=1，否=0）",
                                min_value=0, max_value=1, step=1, key="cirrhosis")

    capsule_invasion = st.number_input("包膜侵犯（是=1, 否=0）",
                                       min_value=0, max_value=1, step=1, key="capsule_invasion")

    tumor_differentiation = st.number_input(
        "肿瘤分化 (巨块型=1, 卫星结节型或其他=0）",
        min_value=0, max_value=1, step=1, key="tumor_differentiation")

    child_stage = st.number_input(
        "Child-Pugh分期（A=1, B或C=0）",
        min_value=0, max_value=1, step=1, key="child_stage")

    predict_button = st.button("🚀 开始预测", type="primary", use_container_width=True)

    if st.sidebar.button("🔄 重置输入", use_container_width=True):
        st.session_state.reset_flag = True
        st.rerun()

if predict_button:
    st.success("✅ 正在产出结果...")
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
    st.subheader("输入确认")
    col1, col2 = st.columns(2)
    # def _create_feature_mapping(self):
    #     return {
    #         '年龄__y': '年龄',
    #         '性别_1': '性别（男=1，女=0）',
    #         'ALB': 'ALB (g/L)',
    #         'AST': 'AST (U/L)',
    #         'ALT': 'ALT (U/L)',
    #         'TBIL': 'TBIL (umol/L)',
    #         'AFP': 'AFP (μg/L)',
    #         'AFP_greater_400': 'AFP > 400',
    #         'AFP_less_400': 'AFP ≤ 400',
    #         'PT': 'PT (秒)',
    #         'INR': 'INR',
    #         'WBC': 'WBC (10^9/L)',
    #         '失血量': '术中失血量（mL）',
    #         '肿瘤MVI_M0': '肿瘤微血管侵犯（M0=1，M1或M2=0）',
    #         '肿瘤直径': '肿瘤直径（cm）',
    #         '是否合并肝硬化_1': '肝硬化（是=1，否=0）',
    #         '包膜是否受侵犯_未浸及': '包膜侵犯（是=1, 否=0）',
    #         '肿瘤是否巨块型分化_1': '肿瘤分化 (巨块型=1, 结节型或其他=0）',
    #         '是否合并肝炎_1': '肝炎（是=1, 否=0）',
    #         '是否大范围切除_1': '扩大切除 （是=1，否=0）',
    #         'child分期_A': 'Child-Pugh分期（A=1, B或C=0）',
    #         'ALBI': 'ALBI评分'
    #     }
    with col1:
        #st.write("*:**")
        st.write(f"- 年龄: {age} ")
        st.write(f"- 性别（男=1，女=0—）: {gender}")
        st.write(f"- ALB: {alb} g/L")
        st.write(f"- AST: {ast} U/L")
        st.write(f"- ALT: {alt} U/L")
        st.write(f"- TBIL: {tbil} umol/L")
        st.write(f"- ALBI分期: {albi}")
        st.write(f"- AFP: {afp} μg/L")
        st.write(f"- AFP > 400 μg/L: {afp_features['AFP_greater_400']}")
        st.write(f"- AFP ≤ 400 μg/L: {afp_features['AFP_less_400']}")
        st.write(f"- PT: {pt} s")
        st.write(f"- INR: {inr}")

    with col2:
        #st.write("**病理与手术指标:**")
        st.write(f"- WBC: {wbc} ×10^9/L")
        st.write(f"- 术中失血量: {blood_loss} mL")
        st.write(f"- 肿瘤微血管侵犯（M0=1，M1或M2=0）: {tumor_mvi}")
        st.write(f"- 肿瘤直径: {tumor_diameter} cm")
        st.write(f"- 肝硬化（是=1，否=0）: {cirrhosis}")
        st.write(f"- 包膜侵犯（是=1, 否=0）: {capsule_invasion}")
        st.write(f"- 肿瘤分化 (巨块型=1, 卫星结节型或其他=0）: {tumor_differentiation}")
        st.write(f"- 肝炎（是=1, 否=0）: {hepatitis}")
        st.write(f"- 扩大切除 （是=1，否=0）: {wide_resection}")
        st.write(f"- Child-Pugh分期（A=1, B或C=0）: {child_stage}")


    st.markdown("---")

    st.subheader("结果")

    predictions, display_value, processed_data, mr, mc, m_xgb, wrapper,ml = predictor.predict_survival(input_data)
    if "XGBoost" in predictions:
        del predictions["XGBoost"]
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("**各模型风险概率:**")


        def get_risk_level(risk_prob):
            if risk_prob >= 0.7:
                return '高风险'
            elif risk_prob >= 0.4:
                return '中风险'
            else:
                return '低风险'


        results_df = pd.DataFrame({
            '模型名称': list(predictions.keys()),
            '风险概率': list(predictions.values()),
            '风险等级': [get_risk_level(risk) for risk in predictions.values()]
        })


        def color_risk(val):
            if val == '高风险':
                return 'background-color: #ffcdd2'
            elif val == '中风险':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #c8e6c9'


        styled_df = results_df.style.applymap(
            lambda x: color_risk(x) if isinstance(x, str) else '',
            subset=['风险等级']
        )
        st.dataframe(styled_df, use_container_width=True)

        fig = px.bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            title="各模型风险概率对比",
            labels={'x': '模型', 'y': '风险概率'},
            color=list(predictions.values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_traces(hovertemplate="模型: %{x}<br>风险: %{y:.6f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**综合风险评估:**")
        avg_risk = np.mean(list(predictions.values()))
        overall_risk = get_risk_level(avg_risk)

        if overall_risk == '高风险':
            st.error(f"⚠️ 综合风险: {overall_risk}")
            st.write(f"平均风险概率: {avg_risk:.3f}")
            st.write("建议: 密切监测，积极治疗")
        elif overall_risk == '中风险':
            st.warning(f"⚠️ 综合风险: {overall_risk}")
            st.write(f"平均风险概率: {avg_risk:.3f}")
            st.write("建议: 定期随访，注意观察")
        else:
            st.success(f"✅ 综合风险: {overall_risk}")
            st.write(f"平均风险概率: {avg_risk:.3f}")
            st.write("建议: 继续监测，保持现状")

        st.write("**模型一致性:**")
        risk_std = np.std(list(predictions.values()))
        if risk_std < 0.05:
            st.success("模型预测高度一致")
        elif risk_std < 0.1:
            st.warning("模型预测基本一致")
        else:
            st.error("模型预测存在分歧")

        st.write(f"标准差: {risk_std:.3f}")

    st.markdown("---")

    st.subheader("生存分析")

    col1, col2 = st.columns([3, 1])

    with col1:

        from survcurve_cn import plot_survival_curves
        muse = joblib.load('models/deepsurv_strict.joblib')
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
            st.subheader("无复发生存期（XGBoost预测）")
            st.info(f"预测生存: {int(display_value)} 天")

    st.markdown("---")
    import shap
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import io
    from PIL import Image

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "DejaVu Sans"

    plt.rcParams["axes.unicode_minus"] = False

    rsf_model = joblib.load('models/newbest_rsf_model.pkl')

    st.subheader("SHAP解释")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("特征贡献 (预生成)")
        st.image("shap_summary_bar.png", use_container_width=True)
    with col_right:
        st.markdown("SHAP瀑布图")
        # with st.spinner("Generating SHAP waterfall plot... This may take 10–20 seconds."):
        try:
            import shap
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import io
            from PIL import Image

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

            feature_names_eng = [
                predictor.feature_mapping.get(f, f) for f in features_top
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
            ax.set_xlabel("SHAP value")

           # plt.title("")
            fig.tight_layout()

            img_wf, buf_wf = fig_to_pil(fig)
            plt.close(fig)
            import gc

            gc.collect()

            st.image(img_wf, use_container_width=True)

        except Exception as e:
            st.error(f"Waterfall generation failed: {e}")


    st.subheader(" 结果导出")

    # 生成报告
    # 生成报告
    report_data = {
        '预测时间': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        '患者年龄': [age],
        '患者性别': [gender],
        'ALBI评分': [albi],
        'AFP值': [afp],
        'AFP_greater_400': [afp_features['AFP_greater_400']],
        'AFP_less_400': [afp_features['AFP_less_400']],
        '肿瘤直径': [tumor_diameter],
        'Cirrhosis': [cirrhosis],
        '综合风险概率': [avg_risk],
        '综合风险等级': [overall_risk],
        '模型一致性': ['一致' if risk_std < 0.1 else '存在分歧']
    }

    report_df = pd.DataFrame(report_data)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 下载预测报告 (CSV)",
            data=report_df.to_csv(index=False),
            file_name=f"hcc_survival_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # 创建详细结果
        detailed_results = pd.DataFrame({
            '模型': list(predictions.keys()),
            '风险概率': list(predictions.values()),
            '风险等级': [get_risk_level(risk) for risk in predictions.values()]
        })

        st.download_button(
            label="📥 下载详细结果 (CSV)",
            data=detailed_results.to_csv(index=False),
            file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


else:
    st.info(" 请在左侧输入患者信息，然后点击'开始预测'按钮")

    st.subheader(" AFP指标转换说明")
    st.write("""
       **AFP指标自动转换规则:**
       - 当AFP > 400 μg/L时：AFP_greater_400 = 1, AFP_less_400 = 0
       - 当AFP ≤ 400 μg/L时：AFP_greater_400 = 0, AFP_less_400 = 1

       **示例:**
       - AFP = 500 → AFP_greater_400 = 1, AFP_less_400 = 0
       - AFP = 400 → AFP_greater_400 = 0, AFP_less_400 = 1
       - AFP = 200 → AFP_greater_400 = 0, AFP_less_400 = 1
       """)