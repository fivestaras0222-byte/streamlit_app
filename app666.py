'''
cd ~/Desktop/pythonProject1/streamlit_app
streamlit run app.py
'''
from feature_order import build_and_save_feature_orders, load_feature_orders
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib
import pickle
import json
import warnings
import os
# 固定随机性
import random
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
# ---- shim for main.DeepMLP so joblib can unpickle ----
import sys, types, torch
from torch import nn

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

# ensure import path 'main.DeepMLP' is available for unpickling
if 'main' in sys.modules:
    setattr(sys.modules['main'], 'DeepMLP', DeepMLP)
else:
    _m = types.ModuleType('main')
    _m.DeepMLP = DeepMLP
    sys.modules['main'] = _m
# ------------------------------------------------------

def _fail(msg: str):
    st.error(msg)
    raise RuntimeError(msg)
warnings.filterwarnings('ignore')

# 尝试导入xgboost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost is not installed.")

# 尝试导入torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("⚠️ PyTorch is not installed.")

# 页面配置
st.set_page_config(
    page_title="HCC Recurrence-Free Survival Analysis Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
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

        # 统一的生存时间点（与训练保持一致）
        # 示例：如果训练时用“730天”，就写 730.0；如果用“24个月”，就写 24.0 并在取值处按月
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

    def _load_deepsurv_joblib(self, path="models/deepsurv_model.joblib"):
        try:
            obj = joblib.load(path)
            est = self._coerce_estimator(obj)
            ok = hasattr(est, "predict") or hasattr(est, "predict_proba")
            st.sidebar.write(f"DeepSurv joblib type: {type(est).__name__}")
            st.sidebar.write(f"DeepSurv methods: {[m for m in dir(est) if m in ('predict', 'predict_proba')]}")
            return est if ok else None
        except Exception as e:
            st.sidebar.warning(f"DeepSurv joblib加载失败: {e}")
            return None

        return m

    def _unwrap_estimator(self, model):
        """从 Pipeline/CalibratedClassifierCV 等包装中取最终estimator"""
        m = model
        # sklearn Pipeline
        if hasattr(m, "steps"):
            m = m.steps[-1][1]
        # CalibratedClassifierCV
        if hasattr(m, "base_estimator"):
            m = m.base_estimator
        # GridSearchCV / RandomizedSearchCV
        if hasattr(m, "best_estimator_"):
            m = m.best_estimator_
        return m

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-float(x)))
    def _survival_at(self, surv_obj, t):
        """
        从各种生存函数对象中取在时间t的生存概率S(t)。
        支持:
          - 可调用对象(如scikit-survival的StepFunction)
          - pandas.Series（index为时间，values为S(t)）
          - pandas.DataFrame（第一列为S(t)）
          - ndarray/list（无时间索引则取最后一个值）
        """
        import numpy as np
        import pandas as pd

        # 1) 可调用：直接评估
        if callable(surv_obj):
            return float(surv_obj(t))

        # 2) DataFrame -> Series
        if isinstance(surv_obj, pd.DataFrame):
            if surv_obj.shape[1] == 0:
                return float('nan')
            surv_obj = surv_obj.iloc[:, 0]

        # 3) Series：按时间轴找 t 左侧的最后一个值
        if isinstance(surv_obj, pd.Series):
            times = np.asarray(surv_obj.index, dtype=float)
            vals = np.asarray(surv_obj.values, dtype=float)
            if times.size == 0:
                return float('nan')
            idx = np.searchsorted(times, t, side='right') - 1
            idx = max(0, min(idx, len(vals) - 1))
            return float(vals[idx])

        # 4) ndarray/list：没有时间轴，取末值（常见step结果）
        if isinstance(surv_obj, (list, np.ndarray)):
            arr = np.asarray(surv_obj, dtype=float)
            if arr.size == 0:
                return float('nan')
            return float(arr[-1])

        # 5) 其他标量
        try:
            return float(surv_obj)
        except Exception:
            return float('nan')

    # 在 class HCCSurvivalPredictor 内新增两个工具方法（放在 _survival_at 下方即可）
    def _pos_proba(self, model, proba: np.ndarray) -> float:
        """返回阳性类(标签=1/True)的概率，兼容 classes_ 顺序"""
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
        """优先用RSF的生存函数在固定时间点T取 1-S(T)；否则退化为 predict_proba 或决策分数"""
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
        """
        Cox模型在固定时间点t的事件概率
        :param model: 已训练的 lifelines.CoxPHFitter 模型
        :param X: 样本特征 DataFrame
        :param time_horizon: 固定时间点
        :return: 每个样本在 time_horizon 的事件概率
        """
        if hasattr(model, "predict_survival_function"):
            surv_funcs = model.predict_survival_function(X, times=[time_horizon])
            probs = 1.0 - surv_funcs.iloc[0].values
            return probs
        _fail("Cox模型没有可用的预测接口")

    def ensure_feature_orders(self, sample_df: pd.DataFrame):
        """
        若 models/feature_order.json 不存在或缺键，则根据已加载的模型与sample_df生成。
        """
        json_path = "models/feature_order.json"
        need_build = True
        orders = {}
        if os.path.exists(json_path):
            try:
                orders = load_feature_orders(json_path)
                # 检查关键模型键是否齐全，不齐则重建
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
        """
        按训练时列顺序对齐，若模型本身带feature_names_in_则优先用；
        否则使用 feature_order.json 中的对应顺序。
        """
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
        # 对多余特征不报错，直接丢弃，确保与训练一致
        X = df[cols].astype("float32")
        return X

    def _predict_deepsurv(self, model, Xdf: pd.DataFrame) -> float:
        """DeepSurv稳定推理：joblib/sklearn风格或PyTorch Module"""
        # sklearn/joblib 风格
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xdf)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(proba[0, 1])
            return float(proba[0, 0])
        if hasattr(model, "predict") and not hasattr(model, "forward"):
            yhat = model.predict(Xdf)
            return float(yhat[0]) if hasattr(yhat, "__getitem__") else float(yhat)

        # PyTorch Module
        if hasattr(model, "forward"):
            import torch
            model.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(Xdf.values, dtype=torch.float32)
                out = model(tensor)
                if isinstance(out, torch.Tensor):
                    out = out.squeeze()
                    out = out[0].item() if out.ndim else float(out.item())
                else:
                    out = float(out)
            # 若是风险分数，统一经sigmoid映射为[0,1]；若你的模型本身输出已是概率，可去掉这行
            prob = 1.0 / (1.0 + np.exp(-out))
            return float(prob)

        _fail("DeepSurv模型不支持的类型：缺少predict/predict_proba/forward")


    def _create_feature_mapping(self):
        """创建中文变量名到英文的映射"""
        return {
            # 基本信息
            '年龄__y': 'Age (Years)',
            '性别_1': 'Gender (Male=1, Female=0)',

            # 实验室检查
            'ALB': 'ALB (g/L)',
            'AST': 'AST (U/L)',
            'ALT': 'ALT (U/L)',
            'TBIL': 'TBIL (umol/L)',
            'AFP': 'AFP (μg/L)',
            'AFP_greater_400': 'AFP > 400',  # 衍生变量
            'AFP_less_400': 'AFP ≤ 400',
            'PT': 'PT (s)',
            'INR': 'INR',
            'WBC': 'WBC (10^9/L)',

            # 疾病相关
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
        """创建英文到中文变量名的反向映射"""
        return {v: k for k, v in self.feature_mapping.items()}

    def _convert_afp_features(self, afp_value):
        """将AFP值转换为二分类特征"""
        if afp_value > 400:
            return {'AFP_greater_400': 1, 'AFP_less_400': 0}
        else:
            return {'AFP_greater_400': 0, 'AFP_less_400': 1}

    def _map_chinese_to_english(self, feature_name):
        """将中文特征名映射为英文"""
        return self.feature_mapping.get(feature_name, feature_name)

    def _map_english_to_chinese(self, english_name):
        """将英文特征名映射为中文"""
        return self.reverse_mapping.get(english_name, english_name)

    def load_models(self):
        """加载预训练的模型"""
        try:
            # 加载随机生存森林模型
            try:
                try:
                    self.models['rsf'] = joblib.load('models/newbest_rsf_model.pkl')
                    #st.sidebar.success("✅ RSF模型加载成功 (joblib)")
                except:
                    with open('models/newbest_rsf_model.pkl', 'rb') as f:
                        self.models['rsf'] = pickle.load(f)
                    #st.sidebar.success("✅ RSF模型加载成功 (pickle)")
            except Exception as e:
                st.sidebar.warning(f"⚠️ RSF Model loading failed: {str(e)}")
                self.models['rsf'] = self.create_mock_model('RSF')

        # 加载XGBoost模型 (.ubj文件)
            if XGBOOST_AVAILABLE:
                        try:
                            self.models['xgboost'] = xgb.Booster()
                            self.models['xgboost'].load_model('models/aft_model.ubj')
                            #st.sidebar.success("✅ XGBoost模型加载成功")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ XGBoost Model loading failed: {str(e)}")
                            self.models['xgboost'] = self.create_mock_model('XGBoost')
            else:
                        st.sidebar.warning("⚠️ XGBoost not installed")
                        self.models['xgboost'] = self.create_mock_model('XGBoost')

                        # 加载Cox模型 - 尝试多种加载方式
            try:
                    self.models['cox'] = joblib.load('models/cox_model.pkl')
                    #st.sidebar.success("✅ Cox模型加载成功 (joblib)")

            except Exception as e:
                st.sidebar.warning(f"⚠️ Cox Model loading failed: {str(e)}")
                self.models['cox'] = self.create_mock_model('Cox')

            try:
                    self.models['logistic'] = joblib.load('models/logistic_model.pkl')
                    #st.sidebar.success("✅ LR模型加载成功 (joblib)")

            except Exception as e:
                st.sidebar.warning(f"⚠️ LR Model loading failed: {str(e)}")
                self.models['logistic'] = self.create_mock_model('logistic')

            try:
                    self.models['deepsurv'] = joblib.load('models/deepsurv_model.joblib')
                    ##st.sidebar.success("✅ NN模型加载成功 (joblib)")

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

    def check_rsf_model(self):
        """检查RSF模型的具体问题"""
        if self.models['rsf'] is not None:
            model = self.models['rsf']
            st.write("**RSF模型详细信息:**")

            # 检查模型类型
            st.write(f"模型类型: {type(model)}")
            st.write(f"模型类名: {model.__class__.__name__}")

            # 检查模型属性
            if hasattr(model, 'n_features_in_'):
                st.write(f"输入特征数: {model.n_features_in_}")

            if hasattr(model, 'n_outputs_'):
                st.write(f"输出数: {model.n_outputs_}")

            if hasattr(model, 'estimators_'):
                st.write(f"树的数量: {len(model.estimators_)}")

            # 检查预测方法
            if hasattr(model, 'predict_proba'):
                st.write("支持predict_proba")
            if hasattr(model, 'predict'):
                st.write("支持predict")
            if hasattr(model, 'predict_survival_function'):
                st.write("支持predict_survival_function")

            # 尝试获取特征重要性
            if hasattr(model, 'feature_importances_'):
                st.write(f"特征重要性形状: {model.feature_importances_.shape}")

    def predict_survival(self, input_data):
        """使用所有模型进行预测（严格按训练列顺序对齐；无随机兜底；稳定输出）"""
        processed_data = self.preprocess_input(input_data)

        # 生成/加载训练列顺序（models/feature_order.json）
        self.ensure_feature_orders(processed_data)

        predictions = {}

        # RSF
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

        # --- XGBoost AFT 模型预测 ---
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

            # 构造 DMatrix
            dmatrix = xgb.DMatrix(X_aligned)

            # 预测生存时间（AFT 输出）
            pred_time = m_xgb.predict(dmatrix)  # 返回 array([xxx])

            # 表格显示：只显示预测生存期，不影响概率模块
            display_value = float(pred_time[0])
            predictions["XGBoost"] = display_value

        if self.models.get("logistic") is not None:
            clf, scaler = self.models["logistic"]

            # 对齐特征
            available_features = [
                'PT', 'child分期_A', 'AFP_greater_400', '失血量',
                '肿瘤是否巨块型分化_1', 'AST', 'ALT', 'WBC', 'INR', 'TBIL',
                '年龄__y', '性别_1', 'ALBI', '包膜是否受侵犯_未浸及', '肿瘤直径',
                'ALB', '是否合并肝硬化_1', '是否大范围切除_1', '是否合并肝炎_1',
                '肿瘤MVI_M0', 'AFP_less_400'
            ]
            X_aligned = processed_data[available_features].values
            X_scaled = scaler.transform(X_aligned)

            # 预测 730 天内复发/死亡的概率
            proba = clf.predict_proba(X_scaled)[:, 1]
            pred_proba = float(proba[0])

            # 显示
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
            # 对齐特征
            X_aligned = processed_data[available_features].values
            # 预测概率
            probs = wrapper.predict_proba(X_aligned)[0]  # 可能是一个曲线 (array)
            # 取某个时间点，比如第12个月（假设每列是月）
            display_value_nn = float(probs[1])
            predictions["Neural Network"] = display_value_nn

        return predictions, display_value, processed_data, mr, mc, m_xgb, wrapper,clf

# 初始化预测器
@st.cache_resource
def load_predictor():
    return HCCSurvivalPredictor()


predictor = load_predictor()

# 应用标题
st.markdown("<h1 style='text-align: center;'>HCC Recurrence-Free Survival Analysis Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: red;'>Disclaimer: Only used for research purposes</h3>", unsafe_allow_html=True)

# 侧边栏 - 输入变量
with st.sidebar:
    st.header("Info Input")

    # 基本信息
    st.subheader("Basic Info")
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=47)
    gender = st.number_input("Gender (Male=1, Female=0)",min_value=0, max_value=1, value=1, step=1
                                )

    # 实验室检查
    st.subheader("Laboratory indicators")
    alb = st.number_input("ALB (g/L)", min_value=20.0, max_value=60.0, value=42.3, step=0.1)
    ast = st.number_input("AST (U/L)", min_value=5.0, max_value=500.0, value=30.0, step=1.0)
    tbil = st.number_input("TBIL (umol/L)", min_value=2.0, max_value=200.0, value=11.0, step=0.1)
    albi = st.number_input("ALBI", min_value=-3.0, max_value=0.0, value=-2.91, step=0.1)
    alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=42.0, step=1.0)  # 添加ALT
    afp = st.number_input("AFP (μg/L)", min_value=0.0, max_value=10000.0, value=600.0, step=1.0)

    # 显示AFP转换结果
    afp_features = predictor._convert_afp_features(afp)
    # if afp > 400:
    #     st.info(f"AFP = {afp} μg/L → AFP_greater_400 = 1, AFP_less_400 = 0")
    # else:
    #     st.info(f"AFP = {afp} μg/L → AFP_greater_400 = 0, AFP_less_400 = 1")

    # 其他实验室指标
    pt = st.number_input("PT (s)", min_value=8.0, max_value=30.0, value=11.5, step=0.1)
    inr = st.number_input("INR", min_value=0.8, max_value=5.0, value=0.97, step=0.1)
    wbc = st.number_input("WBC (10^9/L)", min_value=1.0, max_value=50.0, value=8.88, step=0.1)

    st.subheader("Surgery")
    blood_loss = st.number_input("Blood Loss (mL)", min_value=0, max_value=5000, value=500, step=50)

    # 修改大范围切除为0/1输入
    wide_resection = st.number_input("Extensive Resection (yes=1, no=0)", min_value=0, max_value=1, value=1, step=1
                                     )
    st.subheader("Pathology")
    # 修改MVI为0/1输入
    tumor_mvi = st.number_input("Tumor MVI (M0=1, M1 or M2=0)", min_value=0, max_value=1, value=0, step=1)

    tumor_diameter = st.number_input("Tumor Diameter(cm)", min_value=0.1, max_value=20.0, value=4.0, step=0.1)
    # 修改肝炎为0/1输入
    hepatitis = st.number_input("Hepatitis (present=1, absent=0)", min_value=0, max_value=1, value=1, step=1
                                )
    # 修改肝硬化为0/1输入
    cirrhosis = st.number_input("Cirrhosis (present=1, absent=0)", min_value=0, max_value=1, value=0, step=1,
                                )

    # 修改包膜侵犯为0/1输入
    capsule_invasion = st.number_input("Capsule Invasion (present=1, absent=0)", min_value=0, max_value=1, value=1,
                                       step=1)

    # 修改肿瘤分化为0/1输入
    tumor_differentiation = st.number_input("Tumor Differentiation (giant mass type=1, Others=0)", min_value=0, max_value=1,
                                            value=0, step=1)

    # 修改Child-Pugh分期为0/1输入
    child_stage = st.number_input("Child-Pugh Stage (A=1, B or C=0)", min_value=0, max_value=1, value=1, step=1,
                                 )

    # 预测按钮
    predict_button = st.button("🚀 Start Prediction", type="primary", use_container_width=True)

    # 重置按钮
    if st.button("🔄 Reset Input", use_container_width=True):
        st.rerun()

# 主界面
if predict_button:
    st.success("✅ Generating Results...")

    # 创建输入数据框 - 使用模型中的中文变量名
    input_data = pd.DataFrame({
        '年龄__y': [age],
        '性别_1': [gender],
        'ALB': [alb],
        'AST': [ast],
        'ALT': [alt],  # 添加ALT
        'TBIL': [tbil],
        # 'AFP': [afp],
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
    # 添加AFP转换后的两个衍生变量
    afp_features = predictor._convert_afp_features(afp)
    input_data['AFP_greater_400'] = [afp_features['AFP_greater_400']]
    input_data['AFP_less_400'] = [afp_features['AFP_less_400']]
    # 显示输入数据
    st.subheader("Input Confirmation")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Basic Info:**")
        st.write(f"- Age: {age} ")
        st.write(f"- Gender (Male=1, Female=0): {gender}")
        st.write(f"- ALB: {alb} g/L")
        st.write(f"- AST: {ast} U/L")
        st.write(f"- ALT: {alt} U/L")  # 添加ALT显示
        st.write(f"- TBIL: {tbil} umol/L")
        st.write(f"- ALBI Score: {albi}")
        st.write(f"- AFP: {afp} μg/L")
        st.write(f"- AFP > 400 μg/L: {afp_features['AFP_greater_400']}")  # 显示衍生变量
        st.write(f"- AFP ≤ 400 μg/L: {afp_features['AFP_less_400']}")  # 显示衍生变量
        st.write(f"- PT: {pt} s")
        st.write(f"- INR: {inr}")

    with col2:
        st.write("**Disease and Surgery:**")
        st.write(f"- WBC: {wbc} ×10^9/L")
        st.write(f"- Blood Loss: {blood_loss} mL")
        st.write(f"- Tumor MVI (M0=1, M1 or M2=0): {tumor_mvi}")
        st.write(f"- Tumor Diameter: {tumor_diameter} cm")
        st.write(f"- Cirrhosis (present=1, absent=0): {cirrhosis}")
        st.write(f"- Capsule Invasion (present=1, absent=0): {capsule_invasion}")
        st.write(f"- Tumor Differentiation (Massive=1, Others=0): {tumor_differentiation}")
        st.write(f"- Hepatitis (present=1, absent=0): {hepatitis}")
        st.write(f"- Wide Resection (yes=1, no=0): {wide_resection}")
        st.write(f"- Child-Pugh Stage (A=1, B or C=0): {child_stage}")

    # # 显示AFP转换结果
    # st.subheader(" AFP指标转换结果")
    # afp_features = predictor._convert_afp_features(afp)
    # afp_df = pd.DataFrame({
    #     '特征名': ['AFP_greater_400', 'AFP_less_400'],
    #     '数值': [afp_features['AFP_greater_400'], afp_features['AFP_less_400']],
    #     '说明': [
    #         f"AFP > 400 μg/L (当前值: {afp})",
    #         f"AFP ≤ 400 μg/L (当前值: {afp})"
    #     ]
    # })
    # st.dataframe(afp_df, use_container_width=True)

    st.markdown("---")

    # 模型预测部分
    st.subheader("Results")

    # 获取预测结果
    # 获取预测结果（不包含XGBoost）
    predictions, display_value, processed_data, mr, mc, m_xgb, wrapper,ml = predictor.predict_survival(input_data)
    # 删除或忽略 XGBoost 预测
    if "XGBoost" in predictions:
        del predictions["XGBoost"]
    # 创建预测结果展示
    col1, col2 = st.columns([2, 1])

    with col1:
        # 模型预测结果表格
        st.write("**Risk Probability of Each Model:**")


        def get_risk_level(risk_prob):
            """根据风险概率确定风险等级"""
            if risk_prob >= 0.7:
                return 'High risk'
            elif risk_prob >= 0.4:
                return 'Medium risk'
            else:
                return 'Low risk'


        results_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Risk': list(predictions.values()),
            'Risk Level': [get_risk_level(risk) for risk in predictions.values()]
        })
        # 添加颜色编码
        def color_risk(val):
            if val == 'High risk':
                return 'background-color: #ffcdd2'
            elif val == 'Medium risk':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #c8e6c9'


        styled_df = results_df.style.applymap(
            lambda x: color_risk(x) if isinstance(x, str) else '',
            subset=['Risk Level']
        )
        st.dataframe(styled_df, use_container_width=True)

        # 风险概率可视化
        fig = px.bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            title="Comparison of Risk Probability",
            labels={'x': 'Model', 'y': 'Risk'},
            color=list(predictions.values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_traces(hovertemplate="Model: %{x}<br>Risk: %{y:.6f}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Comprehensive Risk Assessment
        st.write("**Comprehensive Risk Assessment:**")
        avg_risk = np.mean(list(predictions.values()))
        overall_risk = get_risk_level(avg_risk)

        if overall_risk == 'High Risk':
            st.error(f"⚠️ Overall Risk: {overall_risk}")
            st.write(f"Average Risk Probability: {avg_risk:.3f}")
            st.write("Recommendation: Close monitoring and active treatment are advised.")
        elif overall_risk == 'Medium Risk':
            st.warning(f"⚠️ Overall Risk: {overall_risk}")
            st.write(f"Average Risk Probability: {avg_risk:.3f}")
            st.write("Recommendation: Regular follow-up and careful observation are recommended.")
        else:
            st.success(f"✅ Overall Risk: {overall_risk}")
            st.write(f"Average Risk Probability: {avg_risk:.3f}")
            st.write("Recommendation: Continue monitoring and maintain current status.")

        # 模型一致性分析
        st.write("**Model Consistency:**")
        risk_std = np.std(list(predictions.values()))
        if risk_std < 0.1:
            st.success("Model predictions are highly consistent")
        elif risk_std < 0.2:
            st.warning("The model predictions are basically consistent")
        else:
            st.error("Model predictions differ")

        st.write(f"Risk Standard Deviation: {risk_std:.3f}")

    st.markdown("---")

    # 详细分析
    st.subheader("Survival analysis")
    # st.write("**预测生存曲线与XGBoost预测时间:**")

    # 创建两列
    col1, col2 = st.columns([3, 1])  # col1占3/4宽度，col2占1/4宽度

    # --- 左边: 生存曲线 ---
    with col1:

        from survcurve import plot_survival_curves
        muse = joblib.load('models/deepsurv_strict.joblib')
        fig = plot_survival_curves(
            processed_data,
            cox_model=mc,
            rsf_model=mr,
            xgb_aft_model=m_xgb,
            deepsurv_model=muse
        )

        st.plotly_chart(fig, use_container_width=True)

    # --- 右边: XGBoost预测时间 ---
    with col2:
        if predictor.models.get("xgboost") is not None:
            st.subheader("XGBoost Predicted Recurrence-Free Survival")
            st.info(f"Predicted RFS: {int(display_value)} days")

    # 导出结果
    st.markdown("---")
    st.subheader("Result Export")

    # 生成报告
    # 生成报告
    report_data = {
        'Prediction time': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Age': [age],
        'Gender (1=Male,0=Female)': [gender],
        'ALBI': [albi],
        # 'AFP': [afp],
        # 'AFP_greater_400': [afp_features['AFP_greater_400']],
        # 'AFP_less_400': [afp_features['AFP_less_400']],
        'Tumor Diameter (cm)': [tumor_diameter],
        # 'Cirrhosis (1=present )': [cirrhosis],
        'Average risk': [avg_risk],
        'Overall risk level': [overall_risk],
        #'模型一致性': ['一致' if risk_std < 0.1 else '存在分歧']
    }

    report_df = pd.DataFrame(report_data)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Prediction Report(CSV)",
            data=report_df.to_csv(index=False),
            file_name=f"hcc_survival_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # 创建详细结果
        detailed_results = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Risk': list(predictions.values()),
            'Risk Level': [get_risk_level(risk) for risk in predictions.values()]
        })

        st.download_button(
            label="📥 Download all model predictions (CSV)",
            data=detailed_results.to_csv(index=False),
            file_name=f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # 初始界面
    st.info("Please enter the patient information on the left and click the 'Start Prediction' button")

    # # 显示应用说明
    # st.subheader("Notes")
    # st.write('View Results: Including the 2-year recurrence-free survival probability predicted from 4 mod ')

    # col1, col2 = st.columns(2)
    #
    # with col1:
    #     st.write("**功能特点:**")
    #     st.write("• 集成5个生存分析模型")
    #     st.write("• 实时风险概率计算")
    #     st.write("• 多维度风险评估")
    #     st.write("• 结果可视化展示")
    #     st.write("• 报告导出功能")
    #     st.write("• 支持中文变量名映射")
    #     st.write("• AFP指标自动转换")
    #
    # with col2:
    #     st.write("**使用步骤:**")
    #     st.write("1. 输入患者基本信息")
    #     st.write("2. 填写临床指标")
    #     st.write("3. 选择疾病特征")
    #     st.write("4. 点击预测按钮")
    #     st.write("5. 查看预测结果")
    #     st.write("6. 下载预测报告")
    #
    # # 模型信息
    # st.subheader("🤖 模型信息")
    # model_info = pd.DataFrame({
    #     '模型名称': ['Cox比例风险模型', '随机生存森林', '逻辑回归', 'XGBoost模型', '深度生存网络'],
    #     '模型类型': ['统计模型', '机器学习', '统计模型', '梯度提升', '深度学习'],
    #     '适用场景': ['线性关系', '非线性关系', '二分类', '复杂模式', '特征学习'],
    #     '优势': ['解释性强', '处理缺失值', '计算简单', '预测稳定', '自动特征']
    # })
    # st.dataframe(model_info, use_container_width=True)

    # AFP转换说明
    st.subheader("AFP Conversion Instructions")
    st.write("""
    **AFP's Automatic Conversion Rules:**
    - When AFP > 400 μg/L：AFP_greater_400 = 1, AFP_less_400 = 0
    - When AFP ≤ 400 μg/L：AFP_greater_400 = 0, AFP_less_400 = 1

    **Example:**
    - AFP = 500 → AFP_greater_400 = 1, AFP_less_400 = 0
    - AFP = 400 → AFP_greater_400 = 0, AFP_less_400 = 1
    - AFP = 200 → AFP_greater_400 = 0, AFP_less_400 = 1
    """)

    # 变量名映射说明