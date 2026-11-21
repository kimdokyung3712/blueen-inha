import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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
    """업로드한 파일명에서 학교명 추출 (첫 번째 '_' 앞부분)"""
    if filename is None:
        return "우리 학교"
    name = filename.replace(".csv", "").strip()
    if "_" in name:
        return name.split("_")[0]
    return name


def make_bar_chart(df, col, title, ylabel):
    """개체별 막대 그래프 (생중량은 최대·최소 강조)"""
    df_sorted = df.sort_values("개체번호")
    ids = df_sorted["개체번호"].tolist()
    vals = df_sorted[col].tolist()
    colors = ["#4C78A8"] * len(vals)
    if len(vals) > 0:
        max_idx = vals.index(max(vals))
        min_idx = vals.index(min(vals))
        colors[max_idx] = "#E45756"
        colors[min_idx] = "#54A24B"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(i) for i in ids],
        y=vals,
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=0.5,
    ))
    fig.update_layout(
        title={"text": title, "x": 0.05, "xanchor": "left"},
        xaxis_title="개체",
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_box_plot(df, col, title, ylabel):
    """생중량 분포 박스플롯 (fillcolor는 go.Box 바깥에 지정)"""
    vals = df[col].dropna().tolist()
    if len(vals) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=vals,
        name="분포",
        boxpoints="outliers",
        fillcolor="#4C78A8",
        line=dict(color="#313695", width=1.5),
        marker=dict(color="#4C78A8", size=6),
    ))
    fig.update_layout(
        title={"text": title, "x": 0.05, "xanchor": "left"},
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def make_scatter_with_trend(df, x_col, y_col, title, xlabel, ylabel):
    """산점도 + np.polyfit 추세선 + 상관계수 r 반환"""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 2:
        return None, None
    x = sub[x_col].values
    y = sub[y_col].values
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.array([float(x.min()), float(x.max())])
    line_y = slope * line_x + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x.tolist(),
        y=y.tolist(),
        mode="markers",
        marker=dict(color="#4C78A8", size=9, line=dict(width=0.5, color="white")),
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
        title={"text": title, "x": 0.05, "xanchor": "left"},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=True,
        legend=dict(x=0.97, y=0.97, xanchor="right", yanchor="top"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig, r


@st.cache_data
def compute_env_summary(df):
    """환경 데이터 요약 통계 (캐시)"""
    cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in df.columns]
    if not cols:
        return pd.DataFrame()
    label_map = {
        "temperature": "온도 (°C)",
        "humidity": "습도 (%)",
        "ph": "pH",
        "ec": "EC (mS/cm)",
    }
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) > 0:
            rows.append({
                "항목": label_map.get(c, c),
                "평균": round(float(s.mean()), 2),
                "최소": round(float(s.min()), 2),
                "최대": round(float(s.max()), 2),
            })
    return pd.DataFrame(rows)


@st.cache_data
def compute_growth_stats(df):
    """생육 데이터 핵심 통계 (캐시)"""
    if "생중량" not in df.columns:
        return None
    s = df["생중량"].dropna()
    if len(s) == 0:
        return None
    mean_val = round(float(s.mean()), 2)
    std_val = round(float(s.std()), 2)
    cv_val = round(float(s.std() / s.mean()), 3) if s.mean() != 0 else None
    max_val = round(float(s.max()), 2)
    min_val = round(float(s.min()), 2)
    max_idx = s.idxmax()
    min_idx = s.idxmin()
    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv_val,
        "max_val": max_val,
        "min_val": min_val,
        "max_id": df.loc[max_idx, "개체번호"],
        "min_id": df.loc[min_idx, "개체번호"],
    }


# =========================================================================
# Streamlit 앱 시작
# =========================================================================

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

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

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
st.caption("극지식물 개체별 생육 데이터를 비교하고, 환경과 생육의 관계를 살펴보는 대시보드예요.")
st.caption("평균·표준편차·상관계수 등 통계 용어에는 짧은 설명을 붙였어요.")

# ---------------------------------------------------------------------------
# 사이드바: CSV 업로드
# ---------------------------------------------------------------------------

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

# 데이터 읽기 (한 번만 — 사이드바와 탭 모두 재사용)
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

# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
])


# ========================== Tab 1: 실험 개요 ==========================

with tab1:
    try:
        st.header("🔬 연구 배경 및 목적")
        st.markdown(
            f"""
**{school_name} 극지식물 실험**은 서로 다른 개체가 같은 환경에서 자랄 때
생육에 어떤 차이가 생기는지 알아보기 위한 관찰이에요.

같은 조건에서 자란 극지식물도 개체마다 잎 수, 줄기 길이, 무게가 다를 수 있어요.
이런 차이를 숫자로 비교하면 **어떤 개체가 잘 자랐는지**,
**개체 간 차이가 얼마나 큰지** 알 수 있어요.

- **평균**: 모든 값을 더해 개수로 나눈 값 — 데이터의 대표값
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 수 — 클수록 들쭉날쭉
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값 — 단위가 다른 것도 비교 가능
- **상관계수(r)**: 두 값이 함께 변하는 정도 (−1 ~ 1) — 1에 가까우면 함께 증가
            """
        )

        st.header("📋 데이터 요약")

        has_env = not env_df.empty
        has_growth = not growth_df.empty

        if has_env:
            env_cols_det = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            env_item_names = {
                "temperature": "온도", "humidity": "습도",
                "ph": "pH", "ec": "EC",
            }
            env_items = ", ".join([env_item_names.get(c, c) for c in env_cols_det])
            t_min = env_df["time"].min()
            t_max = env_df["time"].max()
            if pd.notna(t_min) and pd.notna(t_max):
                period = f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"
            else:
                period = "알 수 없음"
            n_env = len(env_df)
        else:
            env_items = "-"
            period = "-"
            n_env = 0

        if has_growth:
            n_growth = len(growth_df)
            g_cols_det = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                          if c in growth_df.columns]
            g_item_names = {
                "잎 수": "잎 수", "지상부 길이": "지상부 길이",
                "지하부 길이": "지하부 길이", "생중량": "생중량",
            }
            growth_items = ", ".join([g_item_names.get(c, c) for c in g_cols_det])
        else:
            n_growth = 0
            growth_items = "-"

        summary_rows = []
        summary_rows.append({
            "구분": "환경 데이터",
            "기록 수": n_env,
            "측정 기간": period,
            "측정 항목": env_items,
        })
        summary_rows.append({
            "구분": "생육 결과 데이터",
            "기록 수": n_growth,
            "측정 기간": "1회 측정 (최종 생육)",
            "측정 항목": growth_items,
        })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.header("🏆 주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if has_growth:
                st.metric("총 개체 수", f"{n_growth}개체")
            else:
                st.metric("총 개체 수", "-")

        with c2:
            if has_env and "temperature" in env_df.columns:
                avg_t = round(env_df["temperature"].dropna().mean(), 1)
                st.metric("평균 온도", f"{avg_t} °C")
            else:
                st.metric("평균 온도", "-")

        with c3:
            if has_env and "humidity" in env_df.columns:
                avg_h = round(env_df["humidity"].dropna().mean(), 1)
                st.metric("평균 습도", f"{avg_h} %")
            else:
                st.metric("평균 습도", "-")

        with c4:
            if has_growth and "생중량" in growth_df.columns:
                gs = compute_growth_stats(growth_df)
                if gs:
                    st.metric("평균 생중량", f"{gs['mean']} g")
                else:
                    st.metric("평균 생중량", "-")
            else:
                st.metric("평균 생중량", "-")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ========================== Tab 2: 환경 데이터 ==========================

with tab2:
    try:
        if env_df.empty:
            st.info("환경 데이터 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            st.header("🌡️ 환경 데이터 요약")

            env_summary = compute_env_summary(env_df)
            if env_summary.empty:
                st.info("표시할 환경 측정 항목이 없어요.")
            else:
                st.dataframe(env_summary, use_container_width=True)

            st.header("📈 날짜별 환경 변화 (일별 평균)")

            env_cols_avail = [c for c in ["temperature", "humidity", "ph", "ec"]
                              if c in env_df.columns]
            if not env_cols_avail:
                st.info("표시할 환경 측정 항목이 없어요.")
            else:
                env_clean = env_df.dropna(subset=["time"]).copy()
                env_clean["date"] = env_clean["time"].dt.date
                daily = env_clean.groupby("date").mean(numeric_only=True).reset_index()
                daily["date"] = pd.to_datetime(daily["date"])

                plot_list = [
                    ("temperature", "온도 (°C)", "#FF6B6B"),
                    ("humidity", "습도 (%)", "#4ECDC4"),
                    ("ph", "pH", "#45B7D1"),
                    ("ec", "EC (mS/cm)", "#96CEB4"),
                ]

                n_col = min(2, len(env_cols_avail))
                cols = st.columns(n_col)

                for i, (c, label, color) in enumerate(plot_list):
                    if c not in env_cols_avail:
                        continue
                    with cols[i % n_col]:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=daily["date"],
                            y=daily[c].dropna(),
                            mode="lines",
                            line=dict(color=color, width=2),
                            name=label,
                        ))
                        y_title = label.split(" ")[0] if " " in label else label
                        fig.update_layout(
                            title={"text": label, "x": 0.03, "xanchor": "left"},
                            xaxis_title="날짜",
                            yaxis_title=y_title,
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=10, t=40, b=30),
                            height=220,
                        )
                        st.plotly_chart(fig, width="stretch")

            with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                st.dataframe(env_df, use_container_width=True)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ========================== Tab 3: 생육 결과 ==========================

with tab3:
    try:
        if growth_df.empty:
            st.info("생육 결과 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            has_fresh = "생중량" in growth_df.columns
            has_leaf = "잎 수" in growth_df.columns
            has_shoot = "지상부 길이" in growth_df.columns
            has_root = "지하부 길이" in growth_df.columns

            # ---- 핵심 결과 카드 ----
            st.header("🥇 핵심 결과")

            if has_fresh:
                gs = compute_growth_stats(growth_df)
                if gs:
                    col_max, col_mean, col_min = st.columns(3)
                    with col_mean:
                        st.metric("평균 생중량", f"{gs['mean']} g")
                    with col_max:
                        st.metric(
                            f"최대 개체 (#{gs['max_id']})",
                            f"{gs['max_val']} g",
                        )
                    with col_min:
                        st.metric(
                            f"최소 개체 (#{gs['min_id']})",
                            f"{gs['min_val']} g",
                        )
                    cv_text = f"{gs['cv']}" if gs['cv'] is not None else "계산 불가"
                    st.caption(
                        f"표준편차: {gs['std']} g  |  "
                        f"CV(변동계수): {cv_text}  —  "
                        f"CV가 0.5보다 크면 개체 간 차이가 큰 편이에요."
                    )

            # ---- 개체별 생육 비교 막대 그래프 ----
            st.header("📊 개체별 생육 비교")

            avail_cols = [c for c in ["생중량", "잎 수", "지상부 길이", "지하부 길이"]
                          if c in growth_df.columns]
            if not avail_cols:
                st.info("표시할 생육 측정 항목이 없어요.")
            else:
                n_col = min(2, len(avail_cols))
                plot_cols = st.columns(n_col)

                y_label_map = {
                    "생중량": "생중량 (g)",
                    "잎 수": "잎 수 (장)",
                    "지상부 길이": "지상부 길이 (mm)",
                    "지하부 길이": "지하부 길이 (mm)",
                }
                title_map = {
                    "생중량": "🍃 개체별 생중량 비교 (⭐ 핵심 지표)",
                    "잎 수": "🍃 개체별 잎 수 비교",
                    "지상부 길이": "📏 개체별 지상부 길이 비교",
                    "지하부 길이": "📏 개체별 지하부 길이 비교",
                }

                for i, col in enumerate(avail_cols):
                    with plot_cols[i % n_col]:
                        fig = make_bar_chart(growth_df, col, title_map[col], y_label_map[col])
                        st.plotly_chart(fig, width="stretch")

            # ---- 생중량 분포 ----
            st.header("📦 생중량 분포 (박스플롯)")
            if has_fresh:
                bp_fig = make_box_plot(growth_df, "생중량", "생중량 분포", "생중량 (g)")
                if bp_fig is not None:
                    st.plotly_chart(bp_fig, width="stretch")
                else:
                    st.info("생중량 값이 2개 미만이라 분포를 그릴 수 없어요.")
            else:
                st.info("생중량 열이 없어서 분포를 그릴 수 없어요.")

            # ---- 상관관계 분석 ----
            st.header("🔗 상관관계 분석")
            st.caption(
                "상관계수(r)는 −1 ~ 1 사이 값이에요. "
                "1에 가까우면 두 값이 함께 증가, −1에 가까우면 하나는 증가하는데 다른 하나는 감소해요. "
                "0에 가까우면 두 값 사이에 뚜렷한 관계가 없어요."
            )

            corr_pairs = [
                ("잎 수", "생중량", "잎 수 vs 생중량", "잎 수 (장)", "생중량 (g)"),
                ("지상부 길이", "생중량", "지상부 길이 vs 생중량", "지상부 길이 (mm)", "생중량 (g)"),
            ]

            for x_col, y_col, title, xlabel, ylabel in corr_pairs:
                if x_col not in growth_df.columns or y_col not in growth_df.columns:
                    st.info(f"{title} — 필요한 열이 없어서 표시할 수 없어요.")
                    continue
                fig, r = make_scatter_with_trend(growth_df, x_col, y_col, title, xlabel, ylabel)
                if fig is None:
                    st.info(f"{title} — 데이터가 2개 미만이라 표시할 수 없어요.")
                else:
                    st.plotly_chart(fig, width="stretch")

            # ---- 원본 테이블 + 다운로드 ----
            with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                st.dataframe(growth_df, use_container_width=True)
                csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{school_name}_생육결과데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형태 때문에 표시하지 못했어요: {e}")
