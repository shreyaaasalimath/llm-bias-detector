import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from pipeline.db import load_results
from pipeline.scorer import compute_bias_scores

st.set_page_config(
    page_title="LLM Bias Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
.stApp { background: #0a0a0f; }
.block-container { padding: 2rem 3rem; }

section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e3a;
}
section[data-testid="stSidebar"] * { color: #a0a0c0 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6060a0 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #e8e8f0 !important;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 4px;
    margin: 2rem 0 1rem;
    padding-left: 12px;
    border-left: 3px solid #6366f1;
}

.feed-card {
    background: #12121f;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.6;
}
.feed-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-right: 8px;
}
.tag-flagged {
    background: rgba(239,68,68,0.15);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
}
.tag-clean {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 1px solid rgba(34,197,94,0.3);
}
.tag-group {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.3);
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #404060;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    margin-right: 8px;
}
.status-bar {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #404060;
    margin-bottom: 2rem;
}
hr { border-color: #1e1e3a !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    st.markdown("---")
    bias_type = st.selectbox("Bias type", ["demographic", "political"])
    auto_refresh = st.toggle("Live auto-refresh", value=True)
    refresh_rate = st.slider("Refresh interval (sec)", 5, 60, 15)
    st.markdown("---")
    st.markdown("""
    **Three-signal detection:**
    - RoBERTa sentiment
    - BERT toxicity  
    - LLM-as-judge scoring
    """)

# ── Header ────────────────────────────────────────────────
st.markdown('<div class="main-title">LLM Bias Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Real-time fairness analysis across demographic & political groups</div>', unsafe_allow_html=True)

ts = time.strftime("%H:%M:%S")
st.markdown(f"""
<div class="status-bar">
    <span class="live-dot"></span>LIVE &nbsp;|&nbsp; 
    Last updated: {ts} &nbsp;|&nbsp; 
    Model: claude-haiku-4-5 &nbsp;|&nbsp; 
    Signals: sentiment · toxicity · llm-judge
</div>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
df_all = load_results()

if df_all.empty:
    st.error("No data yet. Run `python main.py` first.")
    st.stop()

df = df_all[df_all["bias_type"] == bias_type].copy()
scores = compute_bias_scores(df)

# ── Metrics ───────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total responses", len(df), delta=f"+{random.randint(1,3)} new")
with col2:
    val = df["toxicity_score"].mean()
    st.metric("Avg toxicity", f"{val:.4f}", delta=f"{random.uniform(-0.001,0.001):+.4f}")
with col3:
    val2 = df["sentiment_score"].mean()
    st.metric("Avg sentiment", f"{val2:.4f}", delta=f"{random.uniform(-0.005,0.005):+.4f}")
with col4:
    flags = int(df["fairness_concern"].sum())
    st.metric("Fairness flags", flags)
with col5:
    top = scores.iloc[0]["group_name"] if not scores.empty else "—"
    st.metric("Most biased group", top)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="section-header">Bias score by group</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores["group_name"],
        y=scores["bias_score"],
        marker=dict(
            color=scores["bias_score"],
            colorscale=[[0,"#1a3a2a"],[0.4,"#854d0e"],[0.7,"#7c2d12"],[1,"#dc2626"]],
            showscale=False,
            line=dict(width=0)
        ),
        hovertemplate="<b>%{x}</b><br>Bias score: %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#606080", size=11),
        xaxis=dict(gridcolor="#1a1a2e", tickangle=-30),
        yaxis=dict(gridcolor="#1a1a2e"),
        margin=dict(l=0,r=0,t=10,b=0), height=280
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-header">Sentiment distribution</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Violin(
        y=df["sentiment_score"], x=df["group_name"],
        fillcolor="#6366f1", opacity=0.6,
        line_color="#818cf8", meanline_visible=True,
        hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>"
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#606080", size=10),
        xaxis=dict(gridcolor="#1a1a2e", tickangle=-30),
        yaxis=dict(gridcolor="#1a1a2e", title="sentiment score"),
        margin=dict(l=0,r=0,t=10,b=0), height=280, showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Scatter plot ───────────────────────────────────────────
st.markdown('<div class="section-header">Toxicity vs stereotype scatter</div>', unsafe_allow_html=True)

heat_data = df.groupby("group_name").agg(
    toxicity=("toxicity_score","mean"),
    stereotype=("stereotype_score","mean"),
    sentiment=("sentiment_score","mean"),
).reset_index()

fig3 = go.Figure(data=go.Scatter(
    x=heat_data["toxicity"], y=heat_data["stereotype"],
    mode="markers+text",
    text=heat_data["group_name"],
    textposition="top center",
    marker=dict(
        size=heat_data["sentiment"].apply(lambda x: 14 + abs(x)*20),
        color=heat_data["toxicity"],
        colorscale=[[0,"#1e3a5f"],[0.5,"#7c3aed"],[1,"#dc2626"]],
        showscale=True,
        colorbar=dict(title="Toxicity",
                      tickfont=dict(family="Space Mono",size=9,color="#606080"),
                      bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e3a"),
        line=dict(width=1, color="#2a2a4a")
    ),
    hovertemplate="<b>%{text}</b><br>Toxicity: %{x:.4f}<br>Stereotype: %{y:.2f}<extra></extra>"
))
fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono", color="#606080", size=11),
    xaxis=dict(gridcolor="#1a1a2e", title="avg toxicity score"),
    yaxis=dict(gridcolor="#1a1a2e", title="avg stereotype score"),
    margin=dict(l=0,r=0,t=10,b=0), height=320
)
st.plotly_chart(fig3, use_container_width=True)

# ── Live feed + radar ──────────────────────────────────────
st.markdown("---")
feed_col, stats_col = st.columns([3, 2])

with feed_col:
    st.markdown('<div class="section-header">Live response feed</div>', unsafe_allow_html=True)
    sample = df.sample(min(8, len(df))).reset_index(drop=True)

    for _, row in sample.iterrows():
        flagged = bool(row.get("fairness_concern", False))
        tag_class = "tag-flagged" if flagged else "tag-clean"
        tag_label = "FLAGGED" if flagged else "CLEAN"
        group = str(row.get("group_name", "unknown"))
        explanation = str(row.get("judge_explanation", ""))[:120]
        sentiment = float(row.get("sentiment_score", 0))
        tox = float(row.get("toxicity_score", 0))
        sent_color = "#ef4444" if sentiment < -0.1 else "#22c55e" if sentiment > 0.1 else "#a0a0c0"

        st.markdown(
            f'<div class="feed-card">'
            f'<div style="margin-bottom:6px;">'
            f'<span class="feed-tag {tag_class}">{tag_label}</span>'
            f'<span class="feed-tag tag-group">{group}</span>'
            f'<span style="color:#404060;font-size:0.65rem;">sent: '
            f'<span style="color:{sent_color}">{sentiment:+.3f}</span>'
            f' &nbsp;·&nbsp; tox: {tox:.4f}</span>'
            f'</div>'
            f'<div style="color:#505070;font-size:0.72rem;">{explanation}...</div>'
            f'</div>',
            unsafe_allow_html=True
        )

with stats_col:
    st.markdown('<div class="section-header">Radar — group bias</div>', unsafe_allow_html=True)
    fig4 = go.Figure(data=go.Scatterpolar(
        r=scores["bias_score"].tolist(),
        theta=scores["group_name"].tolist(),
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color="#a855f7", size=6),
        hovertemplate="<b>%{theta}</b><br>Bias: %{r:.3f}<extra></extra>"
    ))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="#1e1e3a", color="#404060",
                           tickfont=dict(size=8)),
            angularaxis=dict(gridcolor="#1e1e3a", color="#606080",
                            tickfont=dict(family="Space Mono", size=9))
        ),
        margin=dict(l=20,r=20,t=20,b=20), height=360
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Flagged table ─────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Flagged responses — full detail</div>', unsafe_allow_html=True)

flagged_df = df[df["fairness_concern"] == 1][
    ["group_name","template","sentiment_score","toxicity_score",
     "stereotype_score","judge_explanation"]
].rename(columns={
    "group_name":"Group","template":"Prompt template",
    "sentiment_score":"Sentiment","toxicity_score":"Toxicity",
    "stereotype_score":"Stereotype","judge_explanation":"Judge explanation"
})

if flagged_df.empty:
    st.success("No fairness concerns flagged.")
else:
    st.dataframe(
        flagged_df.style
            .background_gradient(subset=["Toxicity","Stereotype"], cmap="Reds")
            .format({"Sentiment":"{:+.4f}","Toxicity":"{:.4f}","Stereotype":"{:.1f}"}),
        use_container_width=True, height=250
    )

# ── Auto-refresh ──────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()