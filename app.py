import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Color scheme
BARCA_BLUE = "#004D98"
BARCA_RED = "#A50044"
BARCA_YELLOW = "#EDBB00"

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

# load dataset
DATA_PATH = "dataset_fcb_final_app.csv"

st.set_page_config(page_title="Dashboard Scouting Barça", layout="wide")

df = load_data(DATA_PATH)

# sidebar sliders for weights
st.sidebar.header("Ajuste de pesos")
weight_attack = st.sidebar.slider("Ataque y Creación", 0.0, 10.0, 4.0, 0.1)
weight_pressure = st.sidebar.slider("Presión Alta", 0.0, 10.0, 2.0, 0.1)
weight_consistency = st.sidebar.slider("Consistencia/Minutos", 0.0, 10.0, 3.0, 0.1)
weight_penalty = st.sidebar.slider("Penalizaciones", 0.0, 10.0, 1.0, 0.1)

# scaling metrics for index computation
scaler = MinMaxScaler()
# columns needing scaling
cols_to_scale = ['xg', 'xa', 'toques_area_rival', 'oport_creadas',
                 'gr_oport_creadas', 'regates_pct',
                 'recuperaciones', 'pos_gan_tercio_of',
                 'minutos', 'perdidas', 'edad']

# ensure columns exist
for c in cols_to_scale:
    if c not in df.columns:
        df[c] = 0

scaled = pd.DataFrame(scaler.fit_transform(df[cols_to_scale]),
                      columns=cols_to_scale,
                      index=df.index)

# calculate index components
attack_components = scaled[['xg','xa','toques_area_rival','oport_creadas','gr_oport_creadas','regates_pct']].sum(axis=1)
pressure_components = scaled[['recuperaciones','pos_gan_tercio_of']].sum(axis=1)
consistency_component = scaled['minutos']
penalty_components = scaled[['perdidas','edad']].sum(axis=1)

# raw index
raw_index = (weight_attack * attack_components +
             weight_pressure * pressure_components +
             weight_consistency * consistency_component -
             weight_penalty * penalty_components)

# final scaling to 0-100
df['Indice_Barca_Final'] = MinMaxScaler(feature_range=(0,100)).fit_transform(raw_index.values.reshape(-1,1))

df_display = df.sort_values('Indice_Barca_Final', ascending=False)

# tabs
tab1, tab2 = st.tabs(["Ranking y Análisis", "Cara a Cara (1vs1)"])

with tab1:
    st.header("Top 10 jugadores según índice Barça")
    top10 = df_display.head(10).copy()
    # custom gradient from blue to red
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("barca_grad", [BARCA_BLUE, BARCA_RED])
    styled = top10.style.background_gradient(cmap=cmap, subset=['Indice_Barca_Final'])
    st.dataframe(styled, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(df, x='edad', y='Indice_Barca_Final', size='minutos',
                          color='Indice_Barca_Final',
                          color_continuous_scale=[BARCA_BLUE, BARCA_RED])
        fig1.update_layout(title="Edad vs Índice Barça", plot_bgcolor='white')
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.scatter(df, x='recuperaciones', y='oport_creadas',
                          color='Indice_Barca_Final',
                          color_continuous_scale=[BARCA_BLUE, BARCA_RED])
        fig2.update_layout(title="Recuperaciones vs Oportunidades creadas",
                           plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Cara a Cara de jugadores")
    players = df['nombre'].unique() if 'nombre' in df.columns else df.index.astype(str)
    player1 = st.selectbox("Jugador 1", players, index=0)
    player2 = st.selectbox("Jugador 2", players, index=1)
    p1 = df[df['nombre']==player1].iloc[0]
    p2 = df[df['nombre']==player2].iloc[0]

    # gauges
    gauge1 = go.Figure(go.Indicator(
        mode="gauge+number", value=p1['Indice_Barca_Final'],
        gauge={'axis':{'range':[0,100]},
               'bar':{'color':BARCA_BLUE}},
        title={'text':player1}))
    gauge2 = go.Figure(go.Indicator(
        mode="gauge+number", value=p2['Indice_Barca_Final'],
        gauge={'axis':{'range':[0,100]},
               'bar':{'color':BARCA_RED}},
        title={'text':player2}))
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(gauge1, use_container_width=True)
    with col2:
        st.plotly_chart(gauge2, use_container_width=True)

    # radar percentiles
    radar_metrics = ['xg','xa','oport_creadas','regates_pct','recuperaciones','toques_area_rival']
    pct = df[radar_metrics].rank(pct=True) * 100
    p1_vals = pct[df['nombre']==player1].iloc[0].tolist()
    p2_vals = pct[df['nombre']==player2].iloc[0].tolist()

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=p1_vals, theta=radar_metrics, fill='none',
                                       name=player1, line_color=BARCA_BLUE))
    fig_radar.add_trace(go.Scatterpolar(r=p2_vals, theta=radar_metrics, fill='none',
                                       name=player2, line_color=BARCA_RED))
    fig_radar.update_layout(polar=dict(bgcolor=BARCA_YELLOW,
                                      radialaxis=dict(range=[0,100])),
                            showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # scatter xg vs xa highlighting
    base = px.scatter(df, x='xg', y='xa', color_discrete_sequence=['lightgray'],
                      opacity=0.4)
    base.add_scatter(x=[p1['xg']], y=[p1['xa']], mode='markers',
                     marker=dict(size=12, color=BARCA_BLUE), name=player1)
    base.add_scatter(x=[p2['xg']], y=[p2['xa']], mode='markers',
                     marker=dict(size=12, color=BARCA_RED), name=player2)
    base.update_layout(title='xG vs xA (jugadores resaltados)')
    st.plotly_chart(base, use_container_width=True)



    
