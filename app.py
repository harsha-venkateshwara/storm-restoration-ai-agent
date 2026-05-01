"""
Storm Restoration Prioritization AI
Production Dashboard — Interactive & Operational
"""
import streamlit.components.v1 as components
import folium
from folium.plugins import MarkerCluster
import networkx as nx
import json
import pickle as _pickle
from shapely.geometry import shape as _shape
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Storm Restoration AI", page_icon="⚡",
    layout="wide", initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    .stApp { background:#0a0e1a; font-family:'Space Grotesk',sans-serif; color:#f1f5f9; }
    .main-header {
        background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#0f172a 100%);
        border:1px solid #1e3a8a; border-radius:12px; padding:1.5rem 2rem; margin-bottom:1.2rem;
    }
    .main-header h1 {
        font-size:2rem; font-weight:700; margin:0 0 0.2rem 0;
        background:linear-gradient(90deg,#60a5fa,#f59e0b);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .main-header p { color:#94a3b8; font-size:0.9rem; margin:0; }
    .metric-card {
        background:#1a2235; border:1px solid #1e293b; border-radius:10px;
        padding:1rem 1.2rem; text-align:center;
    }
    .metric-value { font-size:1.8rem; font-weight:700; font-family:'JetBrains Mono',monospace; color:#3b82f6; }
    .metric-label { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:0.2rem; }
    .metric-delta { font-size:0.75rem; color:#10b981; }
    .section-header {
        font-size:0.95rem; font-weight:600; color:#60a5fa; text-transform:uppercase;
        letter-spacing:0.5px; border-bottom:1px solid #1e293b;
        padding-bottom:0.4rem; margin-bottom:0.8rem;
    }
    .insight-box {
        background:#111827; border-left:3px solid #3b82f6; border-radius:6px;
        padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.85rem; color:#cbd5e1;
    }
    .insight-box b { color:#60a5fa; }
    .alert-critical { border-left-color:#ef4444; }
    .alert-warning  { border-left-color:#f59e0b; }
    .alert-ok       { border-left-color:#10b981; }
    div[data-testid="stSidebar"] { background:#111827 !important; border-right:1px solid #1e293b !important; }
    .stButton > button {
        background:linear-gradient(135deg,#1d4ed8,#2563eb); color:white;
        border:none; border-radius:8px; font-weight:600;
    }
</style>
""", unsafe_allow_html=True)

#Constants
BASE   = Path(__file__).parent
MODELS = BASE / 'outputs' / 'models'
PLOTS  = BASE / 'outputs' / 'plots'

STATE_FIPS = {
    '01':'AL','02':'AK','04':'AZ','05':'AR','06':'CA','08':'CO','09':'CT',
    '10':'DE','12':'FL','13':'GA','15':'HI','16':'ID','17':'IL','18':'IN',
    '19':'IA','20':'KS','21':'KY','22':'LA','23':'ME','24':'MD','25':'MA',
    '26':'MI','27':'MN','28':'MS','29':'MO','30':'MT','31':'NE','32':'NV',
    '33':'NH','34':'NJ','35':'NM','36':'NY','37':'NC','38':'ND','39':'OH',
    '40':'OK','41':'OR','42':'PA','44':'RI','45':'SC','46':'SD','47':'TN',
    '48':'TX','49':'UT','50':'VT','51':'VA','53':'WA','54':'WV','55':'WI',
    '56':'WY','72':'PR','78':'VI'
}

STORM_SCENARIOS = {
    'No Active Storm':     1.00,
    '🌩️ Thunderstorm':    1.40,
    '❄️ Winter Storm':    1.50,
    '🌊 Flash Flood':     1.30,
    '🌬️ High Wind Event': 1.60,
    '🌀 Hurricane Cat 1': 2.00,
    '🌀 Hurricane Cat 3': 3.50,
    '🌀 Hurricane Cat 5': 5.50,
}

HORIZON_DECAY = {24:1.0, 12:0.85, 6:0.70, 3:0.55}

def risk_badge(p):
    if p >= 0.85: return '🔴 CRITICAL'
    if p >= 0.65: return '🟠 HIGH'
    if p >= 0.40: return '🟡 MEDIUM'
    return '🟢 LOW'

def risk_color(p):
    if p >= 0.85: return '#ef4444'
    if p >= 0.65: return '#f59e0b'
    if p >= 0.40: return '#3b82f6'
    return '#10b981'

# Load
@st.cache_resource
def load_eval():
    with open(MODELS / 'evaluation_results.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_gridguard():
    import pickle, json
    graph_path = BASE / 'data/graph/long_island_grid.pkl'
    alert_path = BASE / 'data/graph/alerts_output.json'
    storm_path = BASE / 'data/raw/storms/sandy_2012.geojson'

    if not graph_path.exists() or not alert_path.exists():
        return None, None, None, None

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    with open(alert_path) as f:
        alert_data = json.load(f)
    with open(storm_path) as f:
        storm_fc = json.load(f)

    return G, alert_data, storm_fc, True

@st.cache_data
def load_priority():
    df = pd.read_parquet(MODELS / 'county_priority_2022.parquet')
    df['fips_str']   = df['fips'].astype(str).str.zfill(5)
    df['state_fips'] = df['fips_str'].str[:2]
    df['state_abbr'] = df['state_fips'].map(STATE_FIPS).fillna('?')
    return df

try:
    eval_data   = load_eval()
    priority_df = load_priority()
    data_ok = True
except Exception as e:
    st.error(f"Could not load model results: {e}")
    data_ok = False

if not data_ok:
    st.stop()

@st.cache_resource
def load_gridguard():
    import pickle, json
    graph_path = BASE / 'data/graph/long_island_grid.pkl'
    alert_path = BASE / 'data/graph/alerts_output.json'
    storm_path = BASE / 'data/raw/storms/sandy_2012.geojson'
    if not graph_path.exists() or not alert_path.exists():
        return None, None, None, False
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    with open(alert_path) as f:
        alert_data = json.load(f)
    with open(storm_path) as f:
        storm_fc = json.load(f)
    return G, alert_data, storm_fc, True

# Header
st.markdown("""
<div class="main-header">
  <h1>⚡ Storm Restoration Prioritization AI</h1>
  <p>County-level outage risk · 3,044 US counties · Multi-Task LSTM + Gradient Boosting · EAGLE-I × NOAA 2015–2022</p>
</div>
""", unsafe_allow_html=True)

#Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")
    st.markdown("**🌪️ Storm Scenario Simulator**")
    storm_type = st.selectbox("Active Storm", list(STORM_SCENARIOS.keys()))
    mult       = STORM_SCENARIOS[storm_type]
    horizon_h  = st.select_slider("Forecast Horizon (hours)", [3,6,12,24], value=24)
    decay      = HORIZON_DECAY[horizon_h]
    st.markdown("---")
    st.markdown("**⚖️ Priority Weights**")
    w_proba  = st.slider("Outage Probability",  0.1, 1.0, 0.5, 0.05)
    w_impact = st.slider("Customer Impact",     0.1, 1.0, 0.4, 0.05)
    w_vuln   = st.slider("Vulnerability Index", 0.0, 0.5, 0.1, 0.05)
    st.markdown("---")
    top_k        = st.slider("Top-K Counties", 10, 200, 50, 10)
    state_filter = st.multiselect("Filter by State", sorted(set(STATE_FIPS.values())), default=[])
    st.markdown("---")
    st.markdown("""<div style='font-size:0.72rem;color:#475569;'>
    Model: Multi-Task LSTM (258K params)<br>
    Train: 2015–2020 | Val: 2021 | Test: 2022<br>
    Best Val PR-AUC: 0.633 | Test ROC-AUC: 0.911
    </div>""", unsafe_allow_html=True)

#Apply scenario
df = priority_df.copy()
df['scenario_proba'] = (df['max_outage_proba'] * mult * decay).clip(upper=0.999)
df['priority_score'] = (
    w_proba  * df['scenario_proba'] +
    w_impact * (df['peak_customers_out'] / (df['peak_customers_out'].max() + 1e-8)) +
    w_vuln   * df['outage_rate']
).clip(0, 1)
df['risk_label'] = df['scenario_proba'].apply(risk_badge)

disp_df = df.copy()
if state_filter:
    disp_df = disp_df[disp_df['state_abbr'].isin(state_filter)]
disp_df = disp_df.sort_values('priority_score', ascending=False).head(top_k)

# KPIs
comp     = eval_data['comparison_table']
gb_row   = comp[comp['Model'] == 'Gradient Boosting'].iloc[0]
crit_n   = (df['scenario_proba'] >= 0.85).sum()
high_n   = ((df['scenario_proba'] >= 0.65) & (df['scenario_proba'] < 0.85)).sum()

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, val, lbl, delta in [
    (c1, f"{gb_row['ROC-AUC']:.3f}", "Best ROC-AUC",      "Gradient Boosting"),
    (c2, f"{gb_row['PR-AUC']:.3f}",  "Best PR-AUC",       "+0.57 vs naive"),
    (c3, f"{gb_row['F1']:.3f}",      "Best F1",           "Optimal threshold"),
    (c4, f"{crit_n:,}",              "🔴 Critical Counties", f"{storm_type[:20]}"),
    (c5, f"{high_n:,}",              "🟠 High Risk",        f"{horizon_h}h ahead"),
    (c6, f"{int(df['peak_customers_out'].sum()/1e6):.0f}M", "Customers at Risk", "2022 test"),
]:
    with col:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Alert banner
if storm_type != 'No Active Storm':
    top1 = df.nlargest(1,'priority_score').iloc[0]
    st.markdown(f"""<div class="insight-box alert-critical">
        🚨 <b>ACTIVE SCENARIO: {storm_type}</b> | Horizon: <b>{horizon_h}h</b> |
        Highest priority: <b>FIPS {top1['fips_str']} ({top1['state_abbr']})</b> —
        {top1['scenario_proba']*100:.1f}% outage probability,
        <b>{int(top1['peak_customers_out']):,}</b> customers at risk
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="insight-box alert-ok">
        ✅ <b>No active storm.</b> Showing baseline 2022 risk.
        Select a storm scenario in the sidebar to simulate elevated conditions.
    </div>""", unsafe_allow_html=True)

#Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Risk Map",
    "🏆 Priority Queue",
    "🔍 County Drilldown",
    "📊 Model Results",
    "❓ How to Use",
    "🚨 GridGuard: Grid Alert Map"
])

# TAB 1: Risk Map
with tab1:
    col_map, col_side = st.columns([3, 1])

    with col_map:
        st.markdown('<div class="section-header">County Outage Risk Map</div>',
                    unsafe_allow_html=True)
        fig_map = px.choropleth(
            df, locations='fips_str',
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
            color='priority_score',
            color_continuous_scale=[
                [0.00,'#0d1b2a'],[0.15,'#1a3a5c'],[0.35,'#1d6fa4'],
                [0.55,'#f59e0b'],[0.75,'#ea580c'],[1.00,'#7f1d1d']
            ],
            range_color=(0, df['priority_score'].quantile(0.97)),
            scope='usa',
            labels={'priority_score':'Priority Score'},
            hover_data={
                'fips_str':True, 'state_abbr':True,
                'scenario_proba':':.3f',
                'peak_customers_out':':,.0f',
                'risk_label':True
            }
        )
        fig_map.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0a0e1a',
            font=dict(color='#94a3b8', family='Space Grotesk'),
            geo=dict(bgcolor='#0a0e1a', lakecolor='#111827',
                     landcolor='#111827', subunitcolor='#1e293b'),
            coloraxis_colorbar=dict(
                title=dict(text='Priority Score', font=dict(color='#94a3b8')),
                tickfont=dict(color='#94a3b8')
            ),
            margin=dict(l=0,r=0,t=10,b=0), height=500
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(f"Color = Priority Score | Scenario: **{storm_type}** | Horizon: **{horizon_h}h** | Hover for details")

    with col_side:
        st.markdown('<div class="section-header">Risk Breakdown</div>',
                    unsafe_allow_html=True)

        # Donut
        rc = df['risk_label'].value_counts()
        cmap = {'🔴 CRITICAL':'#ef4444','🟠 HIGH':'#f59e0b',
                '🟡 MEDIUM':'#3b82f6','🟢 LOW':'#10b981'}
        fig_d = go.Figure(go.Pie(
            labels=rc.index, values=rc.values, hole=0.6,
            marker=dict(colors=[cmap.get(l,'#94a3b8') for l in rc.index]),
            textfont=dict(size=9)
        ))
        fig_d.update_layout(
            paper_bgcolor='#1a2235', font=dict(color='#94a3b8'),
            height=210, margin=dict(l=0,r=0,t=5,b=0),
            legend=dict(font=dict(size=8))
        )
        st.plotly_chart(fig_d, use_container_width=True)

        # Top 5 alert cards
        for _, r in df.nlargest(5,'priority_score').iterrows():
            c = risk_color(r['scenario_proba'])
            st.markdown(f"""<div style='background:#111827;border-left:3px solid {c};
                border-radius:5px;padding:0.4rem 0.6rem;margin:0.25rem 0;'>
                <div style='font-size:0.75rem;font-weight:600;color:{c};'>
                    {r['risk_label']} — {r['fips_str']} ({r['state_abbr']})</div>
                <div style='font-size:0.68rem;color:#94a3b8;'>
                    P={r['scenario_proba']:.3f} | {int(r['peak_customers_out']):,} cust
                </div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        # State bar
        st.markdown('<div class="section-header">Top Risk States</div>',
                    unsafe_allow_html=True)
        sr = (df[df['state_abbr']!='?']
              .groupby('state_abbr')['scenario_proba']
              .mean().nlargest(8).reset_index())
        fig_s = go.Figure(go.Bar(
            x=sr['scenario_proba'], y=sr['state_abbr'], orientation='h',
            marker_color=['#ef4444' if p>0.85 else '#f59e0b' if p>0.65
                          else '#3b82f6' for p in sr['scenario_proba']],
        ))
        fig_s.update_layout(
            paper_bgcolor='#1a2235', plot_bgcolor='#1a2235',
            font=dict(color='#94a3b8',size=9),
            xaxis_title='Avg P(Outage)',
            height=240, margin=dict(l=0,r=5,t=5,b=20), showlegend=False
        )
        st.plotly_chart(fig_s, use_container_width=True)

# TAB 2: Priority Queue
with tab2:
    st.markdown('<div class="section-header">Restoration Priority Queue</div>',
                unsafe_allow_html=True)

    st.markdown(f"""<div class="insight-box">
        <b>How to use:</b> Export this list and send to crew dispatch.
        Counties ranked by <code>P(outage) × customer_impact × vulnerability</code>.
        Scenario: <b>{storm_type}</b> | Horizon: <b>{horizon_h}h</b> |
        Showing top <b>{top_k}</b> counties
        {f' in <b>{", ".join(state_filter)}</b>' if state_filter else ' (national)'}.
    </div>""", unsafe_allow_html=True)

    queue = disp_df[['fips_str','state_abbr','scenario_proba',
                      'peak_customers_out','outage_rate',
                      'priority_score','risk_label']].copy().reset_index(drop=True)
    queue.index += 1
    queue.columns = ['FIPS','State','P(Outage)','Peak Customers',
                     'Outage Rate','Priority Score','Risk']

    def color_risk(val):
        if 'CRITICAL' in str(val): return 'color:#ef4444;font-weight:700'
        if 'HIGH'     in str(val): return 'color:#f59e0b;font-weight:600'
        if 'MEDIUM'   in str(val): return 'color:#60a5fa'
        return 'color:#10b981'

    st.dataframe(
        queue.style
        .format({'P(Outage)':'{:.3f}','Peak Customers':'{:,.0f}',
                 'Outage Rate':'{:.3f}','Priority Score':'{:.4f}'})
        .applymap(color_risk, subset=['Risk'])
        .background_gradient(subset=['P(Outage)','Priority Score'], cmap='YlOrRd'),
        use_container_width=True, height=500
    )

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📥 Export CSV (Crew Dispatch)",
            queue.to_csv(index=True),
            "restoration_priority.csv", "text/csv"
        )
    with c2:
        st.markdown(f"""<div class="insight-box alert-warning">
            <b>Dispatch Alert:</b> Pre-position crews in top {min(10,top_k)} counties.
            Combined peak customers at risk:
            <b>{int(disp_df.head(10)['peak_customers_out'].sum()):,}</b>
        </div>""", unsafe_allow_html=True)

    # Charts
    c1, c2, c3 = st.columns(3)
    with c1:
        sc = disp_df['state_abbr'].value_counts().head(10)
        fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation='h',
                               marker_color='#3b82f6'))
        fig.update_layout(title='By State', paper_bgcolor='#1a2235',
                          plot_bgcolor='#1a2235', font=dict(color='#94a3b8',size=9),
                          height=280, margin=dict(l=0,r=5,t=35,b=5))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rc2 = disp_df['risk_label'].value_counts()
        fig = go.Figure(go.Pie(labels=rc2.index, values=rc2.values, hole=0.5,
                               marker=dict(colors=['#ef4444','#f59e0b','#3b82f6','#10b981'])))
        fig.update_layout(title='Risk Mix', paper_bgcolor='#1a2235',
                          font=dict(color='#94a3b8',size=9),
                          height=280, margin=dict(l=0,r=0,t=35,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = go.Figure(go.Histogram(x=disp_df['priority_score'], nbinsx=20,
                                     marker_color='#f59e0b', opacity=0.85))
        fig.update_layout(title='Priority Score Distribution',
                          xaxis_title='Score', yaxis_title='Counties',
                          paper_bgcolor='#1a2235', plot_bgcolor='#1a2235',
                          font=dict(color='#94a3b8',size=9),
                          height=280, margin=dict(l=5,r=5,t=35,b=20))
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: County Drilldown
with tab3:
    st.markdown('<div class="section-header">County Deep Dive</div>',
                unsafe_allow_html=True)

    top200 = df.nlargest(200, 'max_outage_proba')
    labels = [f"{r['fips_str']} — {r['state_abbr']} (P={r['scenario_proba']:.3f})"
              for _, r in top200.iterrows()]

    c1, c2 = st.columns(2)
    with c1:
        sel_lbl  = st.selectbox("Primary County", labels)
        sel_fips = sel_lbl.split(' — ')[0].strip()
    with c2:
        cmp_lbl  = st.selectbox("Compare With", ['None'] + labels)
        cmp_fips = cmp_lbl.split(' — ')[0].strip() if cmp_lbl != 'None' else None

    def draw_county(fips, container):
        row = df[df['fips_str'] == fips]
        if row.empty:
            container.warning(f"County {fips} not found")
            return
        r  = row.iloc[0]
        bc = risk_color(r['scenario_proba'])

        container.markdown(f"""<div style='background:#1a2235;border:1px solid {bc};
            border-radius:10px;padding:1.1rem;margin-bottom:0.8rem;'>
            <h3 style='margin:0;color:{bc};font-size:1rem;'>
                FIPS {r['fips_str']} — {r['state_abbr']}
                <span style='font-size:0.72rem;background:rgba(0,0,0,0.3);
                             border:1px solid {bc};border-radius:20px;
                             padding:1px 8px;margin-left:8px;'>{r['risk_label']}</span>
            </h3>
            <div style='display:grid;grid-template-columns:repeat(3,1fr);
                        gap:0.6rem;margin-top:0.7rem;'>
                <div><div style='font-size:1.3rem;font-weight:700;color:#3b82f6;
                                  font-family:monospace;'>{r['scenario_proba']:.3f}</div>
                     <div style='font-size:0.68rem;color:#94a3b8;'>P(Outage)</div></div>
                <div><div style='font-size:1.3rem;font-weight:700;color:#f59e0b;
                                  font-family:monospace;'>{int(r['peak_customers_out']):,}</div>
                     <div style='font-size:0.68rem;color:#94a3b8;'>Peak Customers</div></div>
                <div><div style='font-size:1.3rem;font-weight:700;color:#10b981;
                                  font-family:monospace;'>{r['outage_rate']:.3f}</div>
                     <div style='font-size:0.68rem;color:#94a3b8;'>Historical Rate</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # 24h risk profile
        hours    = list(range(1, 25))
        base     = r['scenario_proba']
        sim_risk = [min(0.999, base * (1 + 0.1*np.sin(h/4)) *
                        HORIZON_DECAY.get(min([3,6,12,24], key=lambda x: abs(x-h)), 1.0))
                    for h in hours]

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=hours, y=sim_risk, mode='lines+markers',
            line=dict(color=bc, width=2),
            fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'
        ))
        fig_h.add_hline(y=0.85, line_dash='dash', line_color='#ef4444',
                        annotation_text='CRITICAL')
        fig_h.add_hline(y=0.65, line_dash='dash', line_color='#f59e0b',
                        annotation_text='HIGH')
        fig_h.update_layout(
            title=f'24-Hour Risk Forecast — {fips}',
            xaxis_title='Hours Ahead', yaxis_title='P(Outage)', yaxis_range=[0,1],
            paper_bgcolor='#1a2235', plot_bgcolor='#1a2235',
            font=dict(color='#94a3b8',size=9),
            height=240, margin=dict(l=10,r=10,t=35,b=30), showlegend=False
        )
        container.plotly_chart(fig_h, use_container_width=True)

    if cmp_fips:
        cc1, cc2 = st.columns(2)
        draw_county(sel_fips, cc1)
        draw_county(cmp_fips, cc2)
    else:
        draw_county(sel_fips, st)

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (What Drives Predictions)</div>',
                unsafe_allow_html=True)
    try:
        gb_model  = eval_data['gb_model']
        feat_cols = eval_data['feature_cols']
        fi = pd.DataFrame({'Feature':feat_cols,
                           'Importance':gb_model.feature_importances_}
                          ).sort_values('Importance', ascending=True).tail(15)
        STORM_F = {'wind','winter','flood','tropical','other','any_storm',
                   'storm_severity','wind_sum_6h','wind_sum_12h','wind_sum_24h',
                   'any_storm_sum_6h','any_storm_sum_12h','any_storm_sum_24h'}
        colors = ['#ef4444' if f in STORM_F else '#3b82f6' for f in fi['Feature']]
        fig_fi = go.Figure(go.Bar(
            x=fi['Importance'], y=fi['Feature'], orientation='h',
            marker_color=colors, marker_line_width=0
        ))
        fig_fi.update_layout(
            title='Top 15 Features (🔴 Storm | 🔵 Temporal/Lag)',
            paper_bgcolor='#1a2235', plot_bgcolor='#1a2235',
            font=dict(color='#94a3b8',size=10), xaxis_title='Importance',
            height=380, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

#TAB 4: Model Results
with tab4:
    st.markdown('<div class="section-header">Model Performance — Test 2022</div>',
                unsafe_allow_html=True)

    st.dataframe(
        eval_data['comparison_table'].style
        .format({'ROC-AUC':'{:.4f}','PR-AUC':'{:.4f}','F1':'{:.4f}','Brier':'{:.4f}'})
        .highlight_max(subset=['ROC-AUC','PR-AUC','F1'], color='#1a3a52')
        .highlight_min(subset=['Brier'], color='#1a3a52'),
        use_container_width=True
    )

    st.markdown("""<div class="insight-box">
        <b>Why GB outperforms LSTM here:</b> The strongest predictors are
        <code>county_fragility</code> and <code>lag_24h_peak</code> — tabular features
        where tree models excel. The LSTM's unique value is its <b>P50/P90 quantile
        regression head</b> which GB cannot produce — critical for uncertainty quantification
        in operational risk management.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, path, cap in [
        (c1, PLOTS/'lstm_evaluation.png',  'LSTM Training & Test Evaluation'),
        (c2, PLOTS/'model_comparison.png', 'Model Comparison (LR vs GB vs LSTM)')
    ]:
        if path.exists():
            with col:
                st.image(str(path), caption=cap, use_column_width=True)

    st.markdown('<div class="section-header">Architecture Summary</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""**Multi-Task LSTM**
- 28 features × 24h lookback
- Input: Linear(28→64) + LayerNorm
- 2-layer LSTM (128 hidden)
- Attention pooling
- Head A: P(outage) [Focal Loss α=0.75]
- Head B: P50/P90 [Pinball Loss]
- 258,340 parameters""")
    with c2:
        st.markdown("""**Feature Engineering**
- Storm flags (wind/winter/flood/tropical)
- Rolling windows (6h/12h/24h)
- Cyclical time encoding (sin/cos)
- Lag features (1h/24h/168h)
- 30-day county fragility index
- County-adaptive P90 threshold""")
    with c3:
        st.markdown("""**Training**
- AdamW lr=1e-3, wd=1e-4
- Balanced sampling (4:1 neg:pos)
- Early stopping on Val PR-AUC
- Best epoch: 19/30
- Train: 2015–2020 (17.8M rows)
- Val: 2021 | Test: 2022""")

# TAB 5: How to Use
with tab5:
    st.markdown('<div class="section-header">How Storm Restoration AI Works</div>',
                unsafe_allow_html=True)

    st.markdown("""
## What This Tool Does
Predicts which US counties will experience significant power outages in the next
3–24 hours, and ranks them by customer impact so utility crews can be
**pre-positioned before storms hit** — not dispatched after the fact.

---
## Step-by-Step: Storm Desk Operator Workflow

**Step 1 — Select Storm Scenario (Sidebar)**
When NWS issues a warning, select the storm type. The map and queue update instantly
to show elevated risk for counties in the storm's path.

**Step 2 — Read the Risk Map (Tab 1)**
- 🔴 Deep red = CRITICAL (>85% outage probability)
- 🟠 Orange = HIGH (65–85%)
- 🟡 Yellow = MEDIUM (40–65%)
- 🟢 Blue/dark = LOW (<40%)
- **Hover any county** for exact probability, peak customers, and risk level

**Step 3 — Export Priority Queue (Tab 2)**
Click **Export CSV** → send to crew dispatch → crews pre-positioned in staging areas
before storm makes landfall.

**Step 4 — Drill Down (Tab 3)**
See 24-hour risk profile per county. **Compare two counties** when you only have
one crew available — the comparison mode shows both side by side.

**Step 5 — Adjust Weights (Sidebar)**
- Increase **Customer Impact** to prioritize population-dense areas
- Increase **Vulnerability** for elderly/low-income regions
- Change **Forecast Horizon** to see 3h vs 24h risk

---
## Real-World Value

| Without This Tool | With This Tool |
|---|---|
| Crews dispatched after outages | Crews pre-positioned before storm |
| Manual calls between utility desks | Automated ranked priority queue |
| Gut-feel prioritization | P(outage) × customer impact score |
| 72–96h average restoration | Potential 30–40% faster restoration |

---
## Data Sources
- **EAGLE-I** (DOE/ORNL): 15-min county outage estimates, 160M+ records, 2015–2022
- **NOAA Storm Events**: 213,603 county storm records, 2015–2022
- **Model input**: 17.8M hourly sequences across 2,902 counties
    """)

    st.markdown("""<div class="insight-box alert-ok">
        <b>Current prototype</b> uses 2022 test data. A production deployment connects
        to live EAGLE-I and NOAA feeds updating every 15 minutes, with monthly
        automated retraining via Apache Airflow.
    </div>""", unsafe_allow_html=True)

# Footer  
st.markdown("---")
st.markdown("""<div style='text-align:center;font-size:0.72rem;color:#475569;padding:0.6rem;'>
    ⚡ Storm Restoration Prioritization AI · CSE 676A Deep Learning Spring 2026 ·
    Harsha Venkatesan (50680830) & Saba Mina (50681904) ·
    EAGLE-I × NOAA × Multi-Task LSTM + Gradient Boosting
</div>""", unsafe_allow_html=True)

# TAB 6: GRIDGUARD
with tab6:
    st.markdown('<div class="section-header">GridGuard AI — Smart Grid Critical Facility Alert System</div>',
                unsafe_allow_html=True)

    G_grid, alert_data, storm_fc, grid_ok = load_gridguard()

    if not grid_ok:
        st.error("Run gridguard scripts first.")
        st.stop()

    alerts      = alert_data.get("alerts", [])
    at_risk_sub = alert_data.get("at_risk_substations", [])
    critical    = [a for a in alerts if a["priority"] == "CRITICAL"]
    high        = [a for a in alerts if a["priority"] == "HIGH"]

    # KPI row
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, val, lbl in [
        (k1, len(at_risk_sub),        "At-Risk Substations"),
        (k2, len(critical),            "🔴 Hospitals Alerted"),
        (k3, len(high),                "🟠 Fire Stations"),
        (k4, G_grid.number_of_nodes(),"Grid Nodes"),
        (k5, G_grid.number_of_edges(),"Grid Edges"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Alert banner
    if critical:
        top = critical[0]
        st.markdown(f"""<div class="insight-box alert-critical">
            🚨 <b>CRITICAL:</b> {top['facility_name']} is on a predicted outage path.
            Fed by <b>{top['feeding_substation']}</b> (Risk: {top['substation_risk']:.3f}).
            Action: {top['action'][:120]}...
        </div>""", unsafe_allow_html=True)

    # Map + alerts side by side
    map_col, tbl_col = st.columns([3, 2])

    with map_col:
        st.markdown('<div class="section-header">Interactive Grid Route Map</div>',
                    unsafe_allow_html=True)
        map_path = BASE / "outputs/gridguard_map.html"
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                components.html(f.read(), height=520, scrolling=False)
        else:
            st.warning("Map not found. Run python gridguard_map.py")
        st.caption("Red markers = hospitals | Orange = fire stations | Dashed lines = routing paths from substation to facility")

    with tbl_col:
        st.markdown('<div class="section-header">Alert Queue</div>',
                    unsafe_allow_html=True)
        pfilter = st.selectbox("Filter", ["All","CRITICAL","HIGH"], key="gg_filter")
        filtered = alerts if pfilter=="All" else [a for a in alerts if a["priority"]==pfilter]
        PCOL = {"CRITICAL":"#ef4444","HIGH":"#f59e0b","MEDIUM":"#3b82f6"}
        for alert in filtered[:12]:
            c = PCOL.get(alert["priority"],"#94a3b8")
            st.markdown(f"""<div style='background:#111827;border-left:3px solid {c};
                border-radius:5px;padding:0.5rem 0.7rem;margin:0.25rem 0;'>
                <div style='font-size:0.78rem;font-weight:600;color:{c};'>
                    {alert["priority"]} — {alert["facility_name"]}</div>
                <div style='font-size:0.68rem;color:#94a3b8;'>
                    {alert["feeding_substation"]} | Risk: {alert["substation_risk"]:.3f}
                    | Hops: {alert["hop_distance"]}</div>
                <div style='font-size:0.65rem;color:#60a5fa;'>
                    {alert["action"][:90]}...</div>
            </div>""", unsafe_allow_html=True)

        import pandas as pd
        st.download_button(
            "📥 Export Alert Brief",
            pd.DataFrame(alerts)[["priority","facility_name","facility_type",
                                  "feeding_substation","substation_risk",
                                  "hop_distance","action"]].to_csv(index=False),
            "gridguard_alerts.csv", "text/csv"
        )

    # At-risk substations table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">At-Risk Substations</div>',
                unsafe_allow_html=True)
    import pandas as pd
    sub_df = pd.DataFrame(at_risk_sub)[
        ["name","voltage","subtype","adjusted_risk","in_storm_zone"]
    ].rename(columns={"name":"Substation","voltage":"kV",
                      "subtype":"Type","adjusted_risk":"Risk",
                      "in_storm_zone":"In Storm Zone"})
    st.dataframe(sub_df.style.format({"Risk":"{:.4f}"}),
                 use_container_width=True, height=220)