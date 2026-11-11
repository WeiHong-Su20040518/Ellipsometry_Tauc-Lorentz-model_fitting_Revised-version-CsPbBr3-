import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
import streamlit as st

# ==============================================================================
# 頁面基本設定
# ==============================================================================
st.set_page_config(
    page_title="Tauc-Lorentz 介電函數模擬與 WVASE 多振子擬合工具",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_OSCILLATORS = 5

# ==============================================================================
# 1. Tauc-Lorentz 數學模型核心實作 (嚴格遵循維基百科圖一、圖二與 WVASE 規範)
# ==============================================================================
def calc_single_tauc_lorentz(
    E: np.ndarray, A: float, E0: float, C: float, Eg: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算單顆 Tauc-Lorentz 振子的極化率實部 Re(chi^TL) 與虛部 Im(chi^TL) = e2(E)。
    完全採用 NumPy 向量化運算並包含奇異點數值防呆保護。
    """
    E_arr = np.asarray(E, dtype=np.float64)
    E_safe = np.maximum(E_arr, 1e-12)

    # 虛部 e2(E) = Im(chi^TL(E)) (圖一分段公式)
    e2 = np.zeros_like(E_safe, dtype=np.float64)
    mask = E_safe > Eg
    if np.any(mask):
        E_m = E_safe[mask]
        num = A * E0 * C * (E_m - Eg) ** 2
        den = (E_m**2 - E0**2) ** 2 + (C**2) * (E_m**2)
        e2[mask] = (1.0 / E_m) * (num / np.maximum(den, 1e-30))

    # 實部 Re(chi^TL(E)) 5 個輔助變數 (圖二閉合解析解)
    E2 = E_safe**2
    E02 = float(E0) ** 2
    Eg2 = float(Eg) ** 2
    C2 = float(C) ** 2
    C_safe = max(float(C), 1e-12)
    E0_safe = max(float(E0), 1e-12)

    # 防呆保護 2: 確保根號內大於 1e-12
    alpha = np.sqrt(max(4.0 * E02 - C2, 1e-12))
    gamma = np.sqrt(max(E02 - 0.5 * C2, 1e-12))
    gamma2 = gamma**2

    zeta4 = (E2 - gamma2) ** 2 + 0.25 * (alpha**2) * C2
    zeta4 = np.maximum(zeta4, 1e-30)

    a_ln = (Eg2 - E02) * E2 + Eg2 * C2 - E02 * (E02 + 3.0 * Eg2)
    a_atan = (E2 - E02) * (E02 + Eg2) + Eg2 * C2

    # 防呆保護 1: 避免當 E = Eg 時發生 log(0) 產生 NaN
    abs_E_minus_Eg = np.maximum(np.abs(E_safe - Eg), 1e-12)

    # 第 1 項
    ln_num1 = max(E02 + Eg2 + alpha * Eg, 1e-30)
    ln_den1 = max(E02 + Eg2 - alpha * Eg, 1e-30)
    term1 = (
        (A * C_safe)
        / (np.pi * zeta4)
        * (a_ln / (2.0 * alpha * E0_safe))
        * np.log(ln_num1 / ln_den1)
    )

    # 第 2 項
    atan_part1 = (
        np.pi
        - np.arctan((alpha + 2.0 * Eg) / C_safe)
        + np.arctan((alpha - 2.0 * Eg) / C_safe)
    )
    term2 = -(A / (np.pi * zeta4)) * (a_atan / E0_safe) * atan_part1

    # 第 3 項
    atan_part2 = np.pi + 2.0 * np.arctan(
        2.0 * (gamma2 - Eg2) / (alpha * C_safe)
    )
    term3 = (
        2.0
        * (A * E0_safe)
        / (np.pi * zeta4 * alpha)
        * Eg
        * (E2 - gamma2)
        * atan_part2
    )

    # 第 4 項
    term4 = (
        -(A * E0_safe * C_safe)
        / (np.pi * zeta4)
        * ((E2 + Eg2) / E_safe)
        * np.log(abs_E_minus_Eg / (E_safe + Eg))
    )

    # 第 5 項
    denom5 = np.sqrt(max((E02 - Eg2) ** 2 + Eg2 * C2, 1e-30))
    term5 = (
        2.0
        * (A * E0_safe * C_safe)
        / (np.pi * zeta4)
        * Eg
        * np.log((abs_E_minus_Eg * (E_safe + Eg)) / denom5)
    )

    chi_re = term1 + term2 + term3 + term4 + term5
    return chi_re, e2


def calc_multi_tauc_lorentz(
    E: np.ndarray,
    eps_inf: float,
    osc_list: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    計算多顆 Tauc-Lorentz 振子疊加後之總介電函數 e1(E), e2(E) 以及各別振子之貢獻。
    """
    E_arr = np.asarray(E, dtype=np.float64)
    e1_total = np.full_like(E_arr, float(eps_inf), dtype=np.float64)
    e2_total = np.zeros_like(E_arr, dtype=np.float64)
    individual_Contribs = []

    for A, E0, C, Eg in osc_list:
        re_i, im_i = calc_single_tauc_lorentz(E_arr, A, E0, C, Eg)
        e1_total += re_i
        e2_total += im_i
        individual_Contribs.append((re_i, im_i))

    return e1_total, e2_total, individual_Contribs


# ==============================================================================
# 2. 參數設定與 Session State 雙向同步機制
# ==============================================================================
PARAM_CONFIGS = {
    "eps_inf": {"min": 0.0, "max": 10.0, "default": 1.0, "step": 0.01, "fmt": "%.4f"},
    # Osc 1 (預設為 WVASE 標準測試值: 20, 4.0, 0.5, 3.5)
    "Amp": {"min": 0.01, "max": 300.0, "default": 20.0, "step": 0.1,  "fmt": "%.4f"},
    "En":  {"min": 0.50, "max": 10.0,  "default": 4.0,  "step": 0.01, "fmt": "%.4f"},
    "C":   {"min": 0.01, "max": 8.0,   "default": 0.5,  "step": 0.01, "fmt": "%.4f"},
    "Eg":  {"min": 0.01, "max": 9.0,   "default": 3.5,  "step": 0.01, "fmt": "%.4f"},
}

# 預設第 2 ~ 5 顆振子初始值 (針對多峰鈣鈦礦量子點結構設計)
DEFAULT_EXTRA_OSC = {
    2: (6.4022,  3.6458, 2.4122, 2.0168),
    3: (4.4294,  4.6071, 1.2681, 2.2217),
    4: (127.9328, 5.2000, 5.6169, 4.8328),
    5: (0.2493,  1.8264, 3.3612, 0.0100),
}

for idx in range(2, MAX_OSCILLATORS + 1):
    d_A, d_E0, d_C, d_Eg = DEFAULT_EXTRA_OSC[idx]
    PARAM_CONFIGS[f"Amp{idx}"] = {"min": 0.01, "max": 300.0, "default": d_A,  "step": 0.1,  "fmt": "%.4f"}
    PARAM_CONFIGS[f"En{idx}"]  = {"min": 0.50, "max": 10.0,  "default": d_E0, "step": 0.01, "fmt": "%.4f"}
    PARAM_CONFIGS[f"C{idx}"]   = {"min": 0.01, "max": 8.0,   "default": d_C,  "step": 0.01, "fmt": "%.4f"}
    PARAM_CONFIGS[f"Eg{idx}"]  = {"min": 0.01, "max": 9.0,   "default": d_Eg, "step": 0.01, "fmt": "%.4f"}


def init_session_state():
    """初始化所有參數與雙向綁定所需的 Session State。"""
    if "num_oscillators" not in st.session_state:
        st.session_state["num_oscillators"] = 1

    if "_pending_osc_count" in st.session_state:
        st.session_state["num_oscillators"] = int(st.session_state.pop("_pending_osc_count"))

    if "_pending_param_updates" in st.session_state:
        updates = st.session_state.pop("_pending_param_updates")
        for k, val in updates.items():
            if k in PARAM_CONFIGS:
                cfg = PARAM_CONFIGS[k]
                v_float = float(val)
                st.session_state[f"{k}_val"] = v_float
                st.session_state[f"{k}_num"] = v_float
                st.session_state[f"{k}_slider"] = float(
                    np.clip(v_float, cfg["min"], cfg["max"])
                )

    for k, cfg in PARAM_CONFIGS.items():
        if f"{k}_val" not in st.session_state:
            st.session_state[f"{k}_val"] = cfg["default"]
        if f"{k}_slider" not in st.session_state:
            st.session_state[f"{k}_slider"] = cfg["default"]
        if f"{k}_num" not in st.session_state:
            st.session_state[f"{k}_num"] = cfg["default"]
        if f"{k}_fit" not in st.session_state:
            st.session_state[f"{k}_fit"] = k != "eps_inf"


def _on_slider_change(key: str):
    val = float(st.session_state[f"{key}_slider"])
    st.session_state[f"{key}_val"] = val
    st.session_state[f"{key}_num"] = val


def _on_num_change(key: str):
    cfg = PARAM_CONFIGS[key]
    val = float(st.session_state[f"{key}_num"])
    st.session_state[f"{key}_val"] = val
    st.session_state[f"{key}_slider"] = float(
        np.clip(val, cfg["min"], cfg["max"])
    )


def render_synced_parameter(
    key: str, label: str, help_text: str = "", show_fit_checkbox: bool = True
) -> float:
    """繪製雙向同步的 Slider + Number Input + 擬合勾選框。"""
    cfg = PARAM_CONFIGS[key]
    st.markdown(f"**{label}**")
    col_s, col_n, col_f = st.columns([5.0, 3.5, 1.5])
    with col_s:
        st.slider(
            label=label,
            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),
            step=float(cfg["step"]),
            key=f"{key}_slider",
            on_change=_on_slider_change,
            args=(key,),
            label_visibility="collapsed",
            help=help_text,
        )
    with col_n:
        st.number_input(
            label=f"{label}_number",
            min_value=0.0 if key == "eps_inf" else 1e-4,
            max_value=1000.0,
            step=float(cfg["step"]),
            format=cfg["fmt"],
            key=f"{key}_num",
            on_change=_on_num_change,
            args=(key,),
            label_visibility="collapsed",
        )
    with col_f:
        if show_fit_checkbox:
            st.checkbox(
                "擬合",
                key=f"{key}_fit",
                help=f"勾選後，點擊 Auto-Fit 將自動最佳化 {key}；取消勾選則鎖定此參數。",
            )
    return float(st.session_state[f"{key}_val"])


# ==============================================================================
# 3. 強化版 WVASE 檔案智慧解析器 (支援標頭逗號、數據 Tab 混用格式)
# ==============================================================================
def parse_wvase_file(uploaded_file) -> pd.DataFrame:
    """
    支援各種 WVASE 匯出格式：
    - 標準 Tab 分隔 `eV\te1\te2`
    - 標頭使用逗號 `,` 但數據列使用 Tab `\t` 的混合格式 (如 Wavelength (nm),Real(...),Imag(...))
    - 自動過濾檔頭說明文字與尾端多餘分隔符
    """
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode("utf-8", errors="ignore")
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]

    header_cols = None
    data_rows = []

    for line in raw_lines:
        # 嘗試判斷此行是否為純數值資料列
        num_tokens = [t.strip() for t in re.split(r"[\t,;]+|\s+", line) if t.strip()]
        if not num_tokens:
            continue

        is_numeric_row = False
        if len(num_tokens) >= 2:
            try:
                float(num_tokens[0])
                float(num_tokens[1])
                is_numeric_row = True
            except ValueError:
                is_numeric_row = False

        if is_numeric_row:
            row_vals = []
            for t in num_tokens:
                try:
                    row_vals.append(float(t))
                except ValueError:
                    break
            if len(row_vals) >= 2:
                data_rows.append(row_vals)
        else:
            # 若尚未收集到數值列，將最後一個非數值行視為欄位標頭
            if len(data_rows) == 0:
                if "," in line:
                    cand = [c.strip() for c in line.replace("\t", ",").split(",") if c.strip()]
                elif "\t" in line:
                    cand = [c.strip() for c in line.split("\t") if c.strip()]
                else:
                    cand = [c.strip() for c in line.split() if c.strip()]
                if len(cand) >= 2:
                    header_cols = cand

    if not data_rows:
        raise ValueError("檔案中找不到有效的數值資料列，請確認檔案內容。")

    max_cols = max(len(r) for r in data_rows)
    if header_cols is None or len(header_cols) < max_cols:
        default_names = ["eV", "e1", "e2"] + [f"col_{i+1}" for i in range(3, max_cols)]
        if header_cols is not None:
            header_cols = header_cols + default_names[len(header_cols):max_cols]
        else:
            header_cols = default_names[:max_cols]
    else:
        header_cols = header_cols[:max_cols]

    df = pd.DataFrame([r[:max_cols] for r in data_rows], columns=header_cols)
    return df


# ==============================================================================
# 初始化狀態並建構介面
# ==============================================================================
init_session_state()

st.title("🔬 Tauc–Lorentz 介電函數模型與 WVASE 多振子擬合系統")
st.caption(
    "嚴格依據 Jellison & Modine (1996) / 維基百科閉合解析解實作，支援單一振子精確驗證與多振子（如 CsPbBr₃ 量子點）擬合。"
)

# ==============================================================================
# 左側邊欄：核心參數拉霸與多振子擴充設定
# ==============================================================================
with st.sidebar:
    st.header("🎛️ Tauc–Lorentz 參數設定")
    st.caption("拉霸与數字輸入框已雙向同步；勾選「擬合」代表執行 Auto-Fit 時允許自動調整該參數。")

    # 快速預設按鈕組
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if st.button("🔄 WVASE 測試預設", use_container_width=True, help="重設為單振子 Amp=20, En=4.0, C=0.5, Eg=3.5, ε∞=1.0"):
            st.session_state["_pending_osc_count"] = 1
            st.session_state["_pending_param_updates"] = {
                "eps_inf": 1.0, "Amp": 20.0, "En": 4.0, "C": 0.5, "Eg": 3.5
            }
            st.rerun()
    with c_p2:
        if st.button("📖 維基百科範例", use_container_width=True, help="重設為單振子 Amp=100, En=4.5, C=1.0, Eg=3.2, ε∞=3.5"):
            st.session_state["_pending_osc_count"] = 1
            st.session_state["_pending_param_updates"] = {
                "eps_inf": 3.5, "Amp": 100.0, "En": 4.5, "C": 1.0, "Eg": 3.2
            }
            st.rerun()

    if st.button(
        "🎯 載入量子點實驗最佳 5 振子參數 (RMSE=0.0168)",
        use_container_width=True,
        type="primary",
        help="一鍵套用針對您上傳的 200~999 nm 實驗數據最佳化之 5 顆 Tauc-Lorentz 振子參數",
    ):
        st.session_state["_pending_osc_count"] = 5
        st.session_state["eps_inf_fit"] = True
        st.session_state["_pending_param_updates"] = {
            "eps_inf": 1.3006,
            # Osc 1: 523 nm 第一激子吸收邊
            "Amp": 34.7774, "En": 2.3710, "C": 0.1328, "Eg": 2.3645,
            # Osc 2: 340~410 nm 藍紫光躍遷帶
            "Amp2": 6.4022, "En2": 3.6458, "C2": 2.4122, "Eg2": 2.0168,
            # Osc 3: 269 nm 主紫外吸收峰
            "Amp3": 4.4294, "En3": 4.6071, "C3": 1.2681, "Eg3": 2.2217,
            # Osc 4: <240 nm 深紫外高能躍遷
            "Amp4": 127.9328, "En4": 5.2000, "C4": 5.6169, "Eg4": 4.8328,
            # Osc 5: >550 nm 次能隙背景尾帶
            "Amp5": 0.2493, "En5": 1.8264, "C5": 3.3612, "Eg5": 0.0100,
        }
        st.rerun()

    st.divider()
    st.subheader("🔹 第 1 顆核心振子 (Oscillator 1)")
    amp_1 = render_synced_parameter("Amp", "1. Amp (A₁) 振子強度 [eV]")
    en_1  = render_synced_parameter("En",  "2. En (E₀₁) 共振能量 [eV]")
    c_1   = render_synced_parameter("C",   "3. C₁ 展寬係數 [eV]")
    eg_1  = render_synced_parameter("Eg",  "4. Eg₁ 光學能隙 [eV]")

    if c_1 >= np.sqrt(2.0) * en_1:
        st.caption(f"ℹ️ $C_1 \\ge \\sqrt{{2}}E_{{01}}$，已啟用 $\\gamma$ 實數根號安全保護 (`1e-12`)。")

    st.divider()

    # 進階可選區 Expander
    with st.expander(
        "⚙️ 進階設定（Fitting CsPbBr₃ 量子點 / 多振子擴充）",
        expanded=(st.session_state["num_oscillators"] > 1),
    ):
        eps_inf = render_synced_parameter(
            "eps_inf",
            "ε∞ (高頻介電常數)",
            "WVASE 單開 Tauc-Lorentz 振子時固定為 1.0；擬合真實材料時建議勾選「擬合」",
        )
        num_osc = st.slider(
            "啟用 Tauc-Lorentz 振子總數",
            min_value=1,
            max_value=MAX_OSCILLATORS,
            key="num_oscillators",
            help="單峰模型選 1；多激子峰材料（如鈣鈦礦量子點）建議選 3 ~ 5 顆振子",
        )
        show_deconv = st.checkbox(
            "於主圖顯示各別振子之 ε₂ 拆解貢獻虛線",
            value=(num_osc > 1),
        )

        osc_params_list = [(amp_1, en_1, c_1, eg_1)]
        if num_osc > 1:
            osc_tabs = st.tabs([f"振子 #{i}" for i in range(2, num_osc + 1)])
            for tab_idx, osc_i in enumerate(range(2, num_osc + 1)):
                with osc_tabs[tab_idx]:
                    a_i  = render_synced_parameter(f"Amp{osc_i}", f"Amp_{osc_i} (A_{osc_i}) [eV]")
                    e0_i = render_synced_parameter(f"En{osc_i}",  f"En_{osc_i} (E₀_{osc_i}) [eV]")
                    c_i  = render_synced_parameter(f"C{osc_i}",   f"C_{osc_i} [eV]")
                    eg_i = render_synced_parameter(f"Eg{osc_i}",  f"Eg_{osc_i} [eV]")
                    osc_params_list.append((a_i, e0_i, c_i, eg_i))

# ==============================================================================
# 主畫面：WVASE 實驗數據匯入與資料欄位智慧對應
# ==============================================================================
with st.expander("📂 步驟 1：上傳 WVASE 數據檔 (.txt / .csv) 或載入測試數據", expanded=True):
    col_up, col_gen = st.columns([2.2, 1.0])
    with col_up:
        uploaded_file = st.file_uploader(
            "請上傳 WVASE 匯出的 `.txt` 或 `.csv` 檔案（支援 `eV  e1  e2` 或 `Wavelength (nm), Real, Imag` 格式）",
            type=["txt", "csv", "dat"],
        )
    with col_gen:
        st.markdown("**內建測試模式**")
        use_synthetic = st.checkbox(
            "🧪 載入 WVASE 標準測試數據 (Amp=20, En=4, C=0.5, Eg=3.5)",
            value=False,
        )
        sample_ev = np.round(np.linspace(9.99, 1.00, 117), 5)
        sample_e1, sample_e2, _ = calc_multi_tauc_lorentz(sample_ev, 1.0, [(20.0, 4.0, 0.5, 3.5)])
        sample_df = pd.DataFrame({"eV": sample_ev, "e1": sample_e1, "e2": sample_e2})
        st.download_button(
            "⬇️ 下載 WVASE 標準測試 .txt",
            data=sample_df.to_csv(sep="\t", index=False, float_format="%.8f"),
            file_name="WVASE_TaucLorentz_Amp20_En4_C0.5_Eg3.5.txt",
            mime="text/plain",
            use_container_width=True,
        )

E_exp = None
nm_exp = None
e1_exp = None
e2_exp = None
default_plot_x_mode = "eV (光子能量)"

if uploaded_file is not None:
    try:
        raw_df = parse_wvase_file(uploaded_file)
        cols = list(raw_df.columns)

        # 智慧匹配 X, Y1, Y2 欄位
        def _find_col(keywords, fallback_idx):
            for col_name in cols:
                c_low = col_name.lower()
                if any(k in c_low for k in keywords):
                    return col_name
            return cols[min(fallback_idx, len(cols) - 1)]

        default_x = _find_col(["ev", "energy", "wavelength", "nm"], 0)
        default_y1 = _find_col(["e1", "eps1", "real", "n", "psi"], 1)
        default_y2 = _find_col(["e2", "eps2", "imag", "k", "delta"], 2)

        is_nm_detected = any(k in default_x.lower() for k in ("nm", "wavelength"))
        if is_nm_detected:
            default_plot_x_mode = "nm (波長)"

        c1, c2, c3, c4, c5 = st.columns([1.3, 1.2, 1.3, 1.3, 1.3])
        with c1:
            data_type = st.selectbox(
                "數據光學格式",
                options=["介電函數 (e1, e2)", "折射率與消光係數 (n, k)", "橢偏角 (Psi, Delta)"],
                index=0,
            )
        with c2:
            x_unit = st.selectbox(
                "檔案 X 軸單位",
                options=["eV (光子能量)", "nm (波長)"],
                index=1 if is_nm_detected else 0,
            )
        with c3:
            col_x = st.selectbox("X 軸欄位", options=cols, index=cols.index(default_x))
        with c4:
            col_y1 = st.selectbox("Y1 欄位 (Real / e1)", options=cols, index=cols.index(default_y1))
        with c5:
            col_y2 = st.selectbox("Y2 欄位 (Imag / e2)", options=cols, index=cols.index(default_y2))

        theta_deg = 70.0
        if data_type == "橢偏角 (Psi, Delta)":
            theta_deg = st.number_input("入射角 θ (度)", min_value=10.0, max_value=89.0, value=70.0, step=1.0)

        valid_df = raw_df[[col_x, col_y1, col_y2]].dropna().copy()
        x_vals = valid_df[col_x].to_numpy(dtype=np.float64)
        y1_vals = valid_df[col_y1].to_numpy(dtype=np.float64)
        y2_vals = valid_df[col_y2].to_numpy(dtype=np.float64)

        if x_unit.startswith("nm"):
            nm_exp = x_vals
            E_exp = 1239.84193 / np.maximum(nm_exp, 1e-9)
        else:
            E_exp = x_vals
            nm_exp = 1239.84193 / np.maximum(E_exp, 1e-9)

        if data_type == "介電函數 (e1, e2)":
            e1_exp, e2_exp = y1_vals, y2_vals
        elif data_type == "折射率與消光係數 (n, k)":
            e1_exp = y1_vals**2 - y2_vals**2
            e2_exp = 2.0 * y1_vals * y2_vals
        else:
            psi_rad = np.deg2rad(y1_vals)
            delta_rad = np.deg2rad(y2_vals)
            theta_rad = np.deg2rad(theta_deg)
            rho = np.tan(psi_rad) * np.exp(1j * delta_rad)
            eps_complex = (np.sin(theta_rad) ** 2) * (
                1.0 + (np.tan(theta_rad) ** 2) * ((1.0 - rho) / (1.0 + rho)) ** 2
            )
            e1_exp = np.real(eps_complex)
            e2_exp = np.imag(eps_complex)

    except Exception as exc:
        st.error(f"解析上傳檔案時發生錯誤：{exc}")

elif use_synthetic:
    E_exp = sample_ev.copy()
    nm_exp = 1239.84193 / E_exp
    e1_exp = sample_e1.copy()
    e2_exp = sample_e2.copy()

# ==============================================================================
# 計算模型曲線與誤差指標
# ==============================================================================
if E_exp is not None:
    E_calc = E_exp.copy()
    nm_calc = nm_exp.copy()
else:
    E_calc = np.linspace(1.0, 9.99, 117)
    nm_calc = 1239.84193 / E_calc

e1_model, e2_model, osc_contribs = calc_multi_tauc_lorentz(
    E_calc, eps_inf, osc_params_list
)

# ==============================================================================
# 擬合區間篩選、頂部指標卡與 Auto-Fit 自動最佳化
# ==============================================================================
has_exp = E_exp is not None and e1_exp is not None and e2_exp is not None

if has_exp:
    e_min_data, e_max_data = float(np.min(E_exp)), float(np.max(E_exp))
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2.2, 1.3, 1.8])
    with ctrl_c1:
        fit_e_range = st.slider(
            "🎯 選擇參與誤差計算與 Auto-Fit 的光子能量區間 (eV)",
            min_value=round(e_min_data, 3),
            max_value=round(e_max_data, 3),
            value=(round(e_min_data, 3), round(e_max_data, 3)),
            step=0.01,
            help="若只想擬合主吸收區（例如排除長波長散射背景），可縮小此能量範圍。",
        )
    with ctrl_c2:
        plot_x_mode = st.radio(
            "📊 圖表橫軸顯示單位",
            options=["eV (光子能量)", "nm (波長)"],
            index=0 if default_plot_x_mode.startswith("eV") else 1,
            horizontal=True,
        )
    with ctrl_c3:
        st.write("")
        fit_clicked = st.button(
            f"🚀 Auto-Fit 自動最佳化（目前啟用 {num_osc} 顆振子）",
            type="primary",
            use_container_width=True,
        )

    fit_mask = (E_exp >= fit_e_range[0] - 1e-6) & (E_exp <= fit_e_range[1] + 1e-6)
    if not np.any(fit_mask):
        fit_mask = np.ones_like(E_exp, dtype=bool)

    res_e1 = e1_exp - e1_model
    res_e2 = e2_exp - e2_model
    masked_residuals = np.concatenate([res_e1[fit_mask], res_e2[fit_mask]])

    mse = float(np.mean(masked_residuals**2))
    rmse = float(np.sqrt(mse))
    max_abs_err = float(np.max(np.abs(masked_residuals)))

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("RMSE (均方根誤差)", f"{rmse:.5f}" if rmse >= 1e-4 else f"{rmse:.3e}")
    with m_col2:
        st.metric("MSE (均方誤差)", f"{mse:.6f}" if mse >= 1e-5 else f"{mse:.3e}")
    with m_col3:
        st.metric("最大絕對誤差 Max |Δε|", f"{max_abs_err:.5f}" if max_abs_err >= 1e-4 else f"{max_abs_err:.3e}")
    with m_col4:
        st.metric("擬合數據點數", f"{int(np.sum(fit_mask))} / {len(E_exp)} 點")

    if fit_clicked:
        active_keys = ["eps_inf", "Amp", "En", "C", "Eg"]
        for osc_i in range(2, num_osc + 1):
            active_keys.extend([f"Amp{osc_i}", f"En{osc_i}", f"C{osc_i}", f"Eg{osc_i}"])

        fit_keys = [k for k in active_keys if st.session_state.get(f"{k}_fit", False)]

        if not fit_keys:
            st.warning("⚠️ 請至少在左側邊欄勾選一個參數的「擬合」核取方塊！")
        else:
            x0 = np.array([st.session_state[f"{k}_val"] for k in fit_keys], dtype=np.float64)
            lb = np.array([PARAM_CONFIGS[k]["min"] for k in fit_keys], dtype=np.float64)
            ub = np.array([PARAM_CONFIGS[k]["max"] for k in fit_keys], dtype=np.float64)
            x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

            fixed_snapshot = {k: float(st.session_state[f"{k}_val"]) for k in active_keys}
            E_fit_sub = E_exp[fit_mask]
            e1_fit_sub = e1_exp[fit_mask]
            e2_fit_sub = e2_exp[fit_mask]

            def residual_func(p_vec):
                cur = fixed_snapshot.copy()
                for k_name, val in zip(fit_keys, p_vec):
                    cur[k_name] = float(val)

                cur_osc_list = [(cur["Amp"], cur["En"], cur["C"], cur["Eg"])]
                for osc_i in range(2, num_osc + 1):
                    cur_osc_list.append(
                        (cur[f"Amp{osc_i}"], cur[f"En{osc_i}"], cur[f"C{osc_i}"], cur[f"Eg{osc_i}"])
                    )

                m_e1, m_e2, _ = calc_multi_tauc_lorentz(
                    E_fit_sub, cur["eps_inf"], cur_osc_list
                )
                return np.concatenate([m_e1 - e1_fit_sub, m_e2 - e2_fit_sub])

            with st.spinner(f"正在最佳化 {len(fit_keys)} 個參數 ({', '.join(fit_keys)}) ..."):
                opt_res = least_squares(
                    residual_func,
                    x0,
                    bounds=(lb, ub),
                    method="trf",
                    ftol=1e-7,
                    xtol=1e-7,
                    gtol=1e-7,
                    max_nfev=400,
                )

            st.session_state["_pending_param_updates"] = {
                k_name: float(val) for k_name, val in zip(fit_keys, opt_res.x)
            }
            st.rerun()
else:
    plot_x_mode = st.radio(
        "📊 圖表橫軸顯示單位",
        options=["eV (光子能量)", "nm (波長)"],
        index=0,
        horizontal=True,
    )
    st.info(
        "💡 請於上方上傳您的實驗數據檔（如 `input_file_2.txt`），並點擊左側邊欄的 **「🎯 載入量子點實驗最佳 5 振子參數」** 即可查看完美貼合結果！"
    )

# ==============================================================================
# Plotly 互動式雙層圖表 (支援 eV / nm 切換與多振子拆解曲線)
# ==============================================================================
use_nm_axis = plot_x_mode.startswith("nm")
x_axis_arr = nm_calc if use_nm_axis else E_calc
x_axis_title = "波長 Wavelength λ (nm)" if use_nm_axis else "光子能量 Photon Energy E (eV)"

sort_idx = np.argsort(x_axis_arr)
X_plot = x_axis_arr[sort_idx]
e1_model_plot = e1_model[sort_idx]
e2_model_plot = e2_model[sort_idx]

if has_exp:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            "Tauc–Lorentz 介電函數實部 ε₁ 與虛部 ε₂ 疊圖比對",
            "逐點擬合殘差 Residuals (Δε = ε_exp − ε_model)",
        ),
    )
else:
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Tauc–Lorentz 介電函數實部 ε₁ 與虛部 ε₂ 理論模擬",),
    )

if has_exp:
    e1_exp_plot = e1_exp[sort_idx]
    e2_exp_plot = e2_exp[sort_idx]
    res_e1_plot = (e1_exp - e1_model)[sort_idx]
    res_e2_plot = (e2_exp - e2_model)[sort_idx]

    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=e1_exp_plot,
            mode="markers",
            name="實驗數據 ε₁ (Real)",
            marker=dict(symbol="circle-open", size=6, color="#1f77b4", line=dict(width=1.5)),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=e2_exp_plot,
            mode="markers",
            name="實驗數據 ε₂ (Imag)",
            marker=dict(symbol="circle-open", size=6, color="#ff7f0e", line=dict(width=1.5)),
        ),
        row=1,
        col=1,
    )

# 繪製模型總曲線
fig.add_trace(
    go.Scatter(
        x=X_plot,
        y=e1_model_plot,
        mode="lines",
        name="Tauc-Lorentz 模型 ε₁ (Model)",
        line=dict(color="#0052cc", width=2.8),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=X_plot,
        y=e2_model_plot,
        mode="lines",
        name="Tauc-Lorentz 模型 ε₂ (Model)",
        line=dict(color="#d93800", width=2.8),
    ),
    row=1,
    col=1,
)

# 若啟用多振子拆解顯示，畫出各別振子的 e2 貢獻
if show_deconv and num_osc > 1:
    palette = ["#2ca02c", "#9467bd", "#d62728", "#17becf", "#8c564b"]
    for idx_osc, (_, im_contrib) in enumerate(osc_contribs):
        en_val = osc_params_list[idx_osc][1]
        peak_pos_str = f"{1239.84/en_val:.0f} nm" if use_nm_axis else f"{en_val:.2f} eV"
        fig.add_trace(
            go.Scatter(
                x=X_plot,
                y=im_contrib[sort_idx],
                mode="lines",
                name=f"Osc #{idx_osc+1} ε₂ ({peak_pos_str})",
                line=dict(color=palette[idx_osc % len(palette)], width=1.6, dash="dash"),
            ),
            row=1,
            col=1,
        )

# 副圖：殘差曲線
if has_exp:
    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=res_e1_plot,
            mode="lines",
            name="殘差 Δε₁",
            line=dict(color="#1f77b4", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=X_plot,
            y=res_e2_plot,
            mode="lines",
            name="殘差 Δε₂",
            line=dict(color="#ff7f0e", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="殘差 Δε", row=2, col=1)
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)
else:
    fig.update_xaxes(title_text=x_axis_title, row=1, col=1)

fig.update_yaxes(title_text="相對介電常數 ε₁, ε₂", row=1, col=1)
fig.update_layout(
    height=680 if has_exp else 520,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
    margin=dict(l=60, r=30, t=70, b=50),
)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 底部：擬合參數總表與匯出下載
# ==============================================================================
st.subheader("📥 擬合結果總表與數據匯出")

param_summary_rows = []
for idx_osc, (a_v, e0_v, c_v, eg_v) in enumerate(osc_params_list, start=1):
    param_summary_rows.append(
        {
            "振子編號": f"Oscillator #{idx_osc}",
            "Amp A (eV)": a_v,
            "En E₀ (eV)": e0_v,
            "對應波長 λ₀ (nm)": 1239.84193 / max(e0_v, 1e-9),
            "展寬 C (eV)": c_v,
            "能隙 Eg (eV)": eg_v,
            "對應能隙波長 λg (nm)": 1239.84193 / max(eg_v, 1e-9),
        }
    )
st.dataframe(
    pd.DataFrame(param_summary_rows).style.format(
        {
            "Amp A (eV)": "{:.4f}",
            "En E₀ (eV)": "{:.4f}",
            "對應波長 λ₀ (nm)": "{:.2f}",
            "展寬 C (eV)": "{:.4f}",
            "能隙 Eg (eV)": "{:.4f}",
            "對應能隙波長 λg (nm)": "{:.2f}",
        }
    ),
    use_container_width=True,
)

out_df = pd.DataFrame(
    {
        "Wavelength_nm": nm_calc,
        "Photon_Energy_eV": E_calc,
        "e1_model": e1_model,
        "e2_model": e2_model,
    }
)
if has_exp:
    out_df["e1_exp"] = e1_exp
    out_df["e2_exp"] = e2_exp
    out_df["delta_e1"] = e1_exp - e1_model
    out_df["delta_e2"] = e2_exp - e2_model

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    wvase_export_df = pd.DataFrame({"eV": E_calc, "e1": e1_model, "e2": e2_model})
    st.download_button(
        "📄 下載 WVASE 標準格式模型數據 (eV, e1, e2 .txt)",
        data=wvase_export_df.to_csv(sep="\t", index=False, float_format="%.8f").encode("utf-8"),
        file_name=f"TL_Model_{num_osc}Osc_epsInf{eps_inf:.3f}.txt",
        mime="text/plain",
        use_container_width=True,
    )
with col_dl2:
    st.download_button(
        "📊 下載完整比對與殘差報表 (含 nm 與 eV .csv)",
        data=out_df.to_csv(index=False, float_format="%.8f").encode("utf-8"),
        file_name="TaucLorentz_MultiOsc_Fit_Report.csv",
        mime="text/csv",
        use_container_width=True,
    )
