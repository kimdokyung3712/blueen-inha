import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_csv(file):
    """CSV 파일을 인코딩을 바꿔가며 읽기 (utf-8-sig → cp949)"""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    """열 이름을 표준 이름으로 바꾸고 숫자형으로 정리"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


def get_school_name(filename):
    """업로드한 파일명에서 학교명 추출"""
    if filename is None:
        return "우리 학교"
    name = filename.replace(".csv", "").strip()
    if "_" in name:
        return name.split("_")[0]
    return name


def make_histogram(series, title, xlabel, key):
    """히스토그램 생성 (고유 key 전달)"""
    vals = series.dropna()
    if len(vals) < 1:
        return None, key
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals.tolist(),
        nbinsx=max(10, min(30, len(vals) // 2)),
        marker_color="#4C78A8",
        marker_line_color="white",
        marker_line_width=0.5,
        opacity=0.85,
    ))
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left"},
        xaxis_title=xlabel,
        yaxis_title="빈도",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=20, t=45, b=35),
        height=250,
    )
    return fig, key


def make_scatter_trend(df, x_col, y_col, title, xlabel, ylabel, key):
    """산점도 + np.polyfit 추세선 + 상관계수 r 반환 (고유 key 전달)"""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return None, key
    x = sub[x_col].values
    y = sub[y_col].values
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    line_y = slope * line_x + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x.tolist(),
        y=y.tolist(),
        mode="markers",
        marker=dict(color="#4C78A8", size=10, line=dict(width=0.5, color="white")),
        name="데이터",
    ))
    fig.add_trace(go.Scatter(
        x=line_x.tolist(),
        y=line_y.tolist(),
        mode="lines",
        line=dict(color="#E45756", width=2.5, dash="dash"),
        name=f"추세선 (r = {r:.3f})",
    ))
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left"},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=True,
        legend=dict(x=0.97, y=0.97, xanchor="right", yanchor="top"),
        margin=dict(l=40, r=20, t=45, b=35),
        height=320,
    )
    return fig, key


@st.cache_data
def compute_growth_correlation(df, x_col, y_col):
    """두 변수 간 상관계수 계산 (캐시)"""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return None
    return float(np.corrcoef(sub[x_col].values, sub[y_col].values)[0, 1])


# =========================== Streamlit 앱 ===========================

st.set_page_config(page_title="green-inha", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("green-inha")
st.caption("극지식물 개체별 생육 차이 분석 대시보드")

# ---------------- 사이드바: CSV 업로드 ----------------

st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

school_name = get_school_name(
    env_file.name if env_file is not None else (
        growth_file.name if growth_file is not None else None
    )
)

if env_file is None and growth_file is None:
    st.info("⬆️ 사이드바에서 환경 데이터와 생육 결과 CSV를 업로드해 주세요.")
    st.stop()

# 데이터 읽기 (한 번만)
with st.spinner("데이터를 읽는 중이에요..."):
    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    if env_file is not None:
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
    if growth_file is not None:
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)

st.sidebar.success(f"✅ {school_name} 학교 데이터를 불러왔어요.")

st.sidebar.header("📊 생육 측정 항목 선택")
growth_cols = st.sidebar.multiselect(
    "표시할 항목",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이"],
    default=["생중량", "잎 수", "지상부 길이", "지하부 길이"],
)

# ---------------- 탭 ----------------

tab1, tab2, tab3 = st.tabs([
    "📖 실험 목적",
    "📋 원본 데이터",
    "📊 분석 그래프",
])


# ======================= Tab 1: 실험 목적 =======================

with tab1:
    try:
        st.header("🔬 실험 목적")
        st.markdown(
            f"""
**{school_name} 극지식물 실험**은 같은 환경에서 자라는 극지식물 개체마다
생육에 어떤 차이가 있는지 알아보기 위한 관찰이에요.

극지식물은 harsh한 환경에서 살아남기 위해 독특한 성장 전략을 가지고 있어요.
같은 조건에서도 개체마다 잎 수, 줄기 길이, 뿌리 길이, 무게가 달라질 수 있어요.
이런 차이를 숫자로 비교하면 **어떤 개체가 더 잘 자랐는지**,
**개체 간 차이가 얼마나 큰지** 알 수 있어요.

이 실험에서 알아보고자 하는 것은 다음과 같아요.
            """
        )

        st.header("❓ 실험 질문")
        st.markdown(
            """
**질문 1.** 지하부의 길이가 길수록 생중량이 클까?

**질문 2.** 지상부의 길이가 길수록 생중량이 클까?

**질문 3.** 지상부보다 지하부의 길이가 생중량에 더 큰 영향을 미칠까?
            """
        )

        st.info(
            "💡 통계 용어 미리보기\n"
            "- **상관계수(r)**: 두 값이 함께 변하는 정도 (−1 ~ 1)\n"
            "  1에 가까우면 함께 증가, −1에 가까우면 반대 방향, 0에 가까우면 관계 없음\n"
            "- **히스토그램**: 값의 분포(개수가 얼마나 모여 있는지)를 막대 그래프로 보여줌\n"
            "- **산점도**: 두 변수의 관계를 점으로 찍어 보여줌",
            icon="📐",
        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ======================= Tab 2: 원본 데이터 =======================

with tab2:
    try:
        if env_df.empty and growth_df.empty:
            st.info("데이터를 업로드하면 원본 테이블을 볼 수 있어요.")
        else:
            if not env_df.empty:
                st.header("🌡️ 환경 데이터 원본")
                st.dataframe(env_df, width="stretch")
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="환경 데이터 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )

            if not growth_df.empty:
                st.header("🌱 생육 결과 데이터 원본")
                st.dataframe(growth_df, width="stretch")
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="생육 결과 CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ======================= Tab 3: 분석 그래프 =======================

with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 분석 그래프가 표시돼요.")
        else:
            has_fresh = "생중량" in growth_df.columns
            has_shoot = "지상부 길이" in growth_df.columns
            has_root = "지하부 길이" in growth_df.columns

            # ----- 히스토그램: 생중량 + 지상부 길이 -----
            st.header("📊 생중량 & 지상부 길이 분포")
            c1, c2 = st.columns(2)
            with c1:
                if has_fresh:
                    fig, key = make_histogram(
                        growth_df["생중량"], "생중량 분포", "생중량 (g)", "hist_fresh_1"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch", key=key)
                    else:
                        st.info("생중량 데이터가 부족해요.")
                else:
                    st.info("생중량 열이 없어요.")
            with c2:
                if has_shoot:
                    fig, key = make_histogram(
                        growth_df["지상부 길이"], "지상부 길이 분포", "지상부 길이 (mm)", "hist_shoot_1"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch", key=key)
                    else:
                        st.info("지상부 길이 데이터가 부족해요.")
                else:
                    st.info("지상부 길이 열이 없어요.")

            # ----- 히스토그램: 생중량 + 지하부 길이 -----
            st.header("📊 생중량 & 지하부 길이 분포")
            c3, c4 = st.columns(2)
            with c3:
                if has_fresh:
                    fig, key = make_histogram(
                        growth_df["생중량"], "생중량 분포", "생중량 (g)", "hist_fresh_2"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch", key=key)
                    else:
                        st.info("생중량 데이터가 부족해요.")
                else:
                    st.info("생중량 열이 없어요.")
            with c4:
                if has_root:
                    fig, key = make_histogram(
                        growth_df["지하부 길이"], "지하부 길이 분포", "지하부 길이 (mm)", "hist_root_1"
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch", key=key)
                    else:
                        st.info("지하부 길이 데이터가 부족해요.")
                else:
                    st.info("지하부 길이 열이 없어요.")

            # ----- 산점도: 지상부 길이 vs 생중량 -----
            st.header("🔗 지상부 길이 vs 생중량")
            if has_shoot and has_fresh:
                fig, key = make_scatter_trend(
                    growth_df,
                    "지상부 길이", "생중량",
                    "지상부 길이 vs 생중량",
                    "지상부 길이 (mm)",
                    "생중량 (g)",
                    "scatter_shoot",
                )
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key=key)
                    if r_shoot > 0.7:
                        strength = "강한 양의 상관관계"
                    elif r_shoot > 0.3:
                        strength = "보통 상관관계"
                    else:
                        strength = "약한 상관관계"
                    st.caption(
                        f"상관계수 r = {r_shoot:.3f}  —  {strength}"
                    )
                else:
                    st.info("데이터가 2개 미만이라 산점도를 그릴 수 없어요.")
            else:
                st.info("지상부 길이와 생중량 열이 모두 필요해요.")

            # ----- 산점도: 지하부 길이 vs 생중량 -----
            st.header("🔗 지하부 길이 vs 생중량")
            if has_root and has_fresh:
                fig, key = make_scatter_trend(
                    growth_df,
                    "지하부 길이", "생중량",
                    "지하부 길이 vs 생중량",
                    "지하부 길이 (mm)",
                    "생중량 (g)",
                    "scatter_root",
                )
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key=key)
                    if r_root > 0.7:
                        strength = "강한 양의 상관관계"
                    elif r_root > 0.3:
                        strength = "보통 상관관계"
                    else:
                        strength = "약한 상관관계"
                    st.caption(
                        f"상관계수 r = {r_root:.3f}  —  {strength}"
                    )
                else:
                    st.info("데이터가 2개 미만이라 산점도를 그릴 수 없어요.")
            else:
                st.info("지하부 길이와 생중량 열이 모두 필요해요.")

            # ----- 상관관계 비교 -----
            if has_shoot and has_root and has_fresh:
                r_shoot = compute_growth_correlation(
                    growth_df, "지상부 길이", "생중량"
                )
                r_root = compute_growth_correlation(
                    growth_df, "지하부 길이", "생중량"
                )
                if r_shoot is not None and r_root is not None:
                    st.header("📈 상관관계 비교")
                    if abs(r_root) > abs(r_shoot):
                        st.success(
                            f"✅ 지하부 길이(r = {r_root:.3f})가 "
                            f"지상부 길이(r = {r_shoot:.3f})보다 "
                            f"생중량과 더 강한 상관관계를 보여요. "
                            f"→ **지하부의 길이가 생중량에 더 큰 영향을 미친다**는 "
                            f"가설을 지지해요."
                        )
                    elif abs(r_shoot) > abs(r_root):
                        st.info(
                            f"지상부 길이(r = {r_shoot:.3f})가 "
                            f"지하부 길이(r = {r_root:.3f})보다 "
                            f"생중량과 더 강한 상관관계를 보여요."
                        )
                    else:
                        st.info(
                            f"지상부 길이(r = {r_shoot:.3f})와 "
                            f"지하부 길이(r = {r_root:.3f})가 "
                            f"비슷한 수준의 상관관계를 보여요."
                        )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
