import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image

# --- 1. CONFIGURACIÓN Y CSS (Fondo verde y Tipografía) ---
st.set_page_config(page_title="Dashboard Scouting Barça", layout="wide", page_icon="🔵🔴")

# Inyección de CSS para la fuente y el fondo verde
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Fondo verde muy suave y elegante para la app */
    .stApp {
        background-color: #E6F4EA; 
    }

    /* Fondo de la barra lateral con un verde ligeramente distinto */
    [data-testid="stSidebar"] {
        background-color: #D4EDDA !important; 
    }
    
    /* Hacer que las tablas tengan fondo blanco para que se lean bien */
    .stDataFrame {
        background-color: #FFFFFF !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

BARCA_BLUE = "#004D98"
BARCA_RED = "#A50044"
BARCA_YELLOW = "#EDBB00"
DARK_GREEN = "#144D22" # Para los gráficos nuevos

# --- 2. CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("dataset_fcb_final_app.csv")
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- 3. BARRA LATERAL (NAVEGACIÓN) ---
st.sidebar.title("⚽ FCB Scouting")
st.sidebar.markdown("---")
opcion_menu = st.sidebar.radio(
    "Navegación:",
    ["🏆 Ranking Top Fichajes", "📈 Gráficos e Insights", "⚔️ Cara a Cara (1vs1)"]
)
st.sidebar.markdown("---")

# --- 4. LÓGICA DE CÁLCULO (PESOS FIJOS) ---
stats_ataque = ['xg', 'xa', 'toques_area_rival', 'oport_creadas', 'gr_oport_creadas', 'regates_pct']
stats_presion = ['recuperaciones', 'pos_gan_tercio_of']
stats_consistencia = ['minutos']
stats_negativas = ['perdidas', 'edad']

todas_las_metricas = stats_ataque + stats_presion + stats_consistencia + stats_negativas
columnas_validas = [col for col in todas_las_metricas if col in df.columns]

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[columnas_validas] = scaler.fit_transform(df[columnas_validas])

weight_attack, weight_pressure, weight_consistency, weight_penalty = 4.0, 2.0, 3.0, 1.0

score_ataque = df_scaled[[c for c in stats_ataque if c in df.columns]].sum(axis=1) * weight_attack
score_presion = df_scaled[[c for c in stats_presion if c in df.columns]].sum(axis=1) * weight_pressure
score_consistencia = df_scaled[[c for c in stats_consistencia if c in df.columns]].sum(axis=1) * weight_consistency
score_negativo = df_scaled[[c for c in stats_negativas if c in df.columns]].sum(axis=1) * weight_penalty

df['Indice_Bruto'] = (score_ataque + score_presion + score_consistencia) - score_negativo
indice_scaler = MinMaxScaler(feature_range=(0, 100))
df['Indice_Barca_Final'] = indice_scaler.fit_transform(df[['Indice_Bruto']])

df_ranking = df.sort_values(by='Indice_Barca_Final', ascending=False).reset_index(drop=True)

# --- 5. VISTAS DE LA APP ---

if opcion_menu == "🏆 Ranking Top Fichajes":
    st.header("🏆 Top Fichajes (Algoritmo Barça)")
    cols_vista = ['jugador', 'Indice_Barca_Final', 'edad', 'minutos', 'xg', 'oport_creadas', 'recuperaciones']
    cols_vista = [c for c in cols_vista if c in df_ranking.columns]
    
    st.dataframe(
        df_ranking[cols_vista].style.background_gradient(cmap='Greens', subset=['Indice_Barca_Final']),
        use_container_width=True, height=450
    )

elif opcion_menu == "📈 Gráficos e Insights":
    
    # NUEVO: Gráfico dinámico de Top 5
    st.header("📊 Top 5 Jugadores por Categoría")
    
    diccionario_metricas = {
        'xg': 'Goles Esperados (xG)',
        'xa': 'Asistencias Esperadas (xA)',
        'oport_creadas': 'Oportunidades Creadas',
        'recuperaciones': 'Balones Recuperados',
        'regates_pct': 'Porcentaje Acierto Regates',
        'pos_gan_tercio_of': 'Recuperaciones en Ataque',
        'toques_area_rival': 'Toques en Área Rival'
    }
    
    metrica_seleccionada = st.selectbox(
        "Selecciona la estadística a analizar:", 
        options=list(diccionario_metricas.keys()), 
        format_func=lambda x: diccionario_metricas[x]
    )
    
    if metrica_seleccionada in df.columns:
        # Cogemos el top 5 y le damos la vuelta para que el mejor salga arriba en el gráfico
        top5 = df.nlargest(5, metrica_seleccionada).sort_values(by=metrica_seleccionada, ascending=True)
        
        fig_bar = px.bar(
            top5, x=metrica_seleccionada, y='jugador', orientation='h',
            text=metrica_seleccionada,
            labels={'jugador': '', metrica_seleccionada: diccionario_metricas[metrica_seleccionada]}
        )
        fig_bar.update_traces(marker_color=DARK_GREEN, texttemplate='%{text:.2f}', textposition='outside')
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Rendimiento vs Juventud")
        fig1 = px.scatter(
            df_ranking, x='edad', y='Indice_Barca_Final', text='jugador', 
            size='minutos', color_discrete_sequence=[BARCA_BLUE],
            labels={'edad': 'Edad', 'Indice_Barca_Final': 'Score Barça'}
        )
        fig1.update_traces(textposition='top center')
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("ADN Barça: Presión vs Creación")
        fig2 = px.scatter(
            df_ranking, x='recuperaciones', y='oport_creadas', text='jugador',
            size='Indice_Barca_Final', color_discrete_sequence=[BARCA_RED],
            labels={'recuperaciones': 'Recuperaciones', 'oport_creadas': 'Oportunidades Creadas'}
        )
        fig2.update_traces(textposition='top center')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

elif opcion_menu == "⚔️ Cara a Cara (1vs1)":
    st.header("⚔️ Comparativa Directa")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        jugador1 = st.selectbox("🔴 Jugador 1", df_ranking['jugador'].tolist(), index=0)
    with col_s2:
        idx_j2 = 1 if len(df_ranking) > 1 else 0
        jugador2 = st.selectbox("🔵 Jugador 2", df_ranking['jugador'].tolist(), index=idx_j2)
        
    p1_data = df_ranking[df_ranking['jugador'] == jugador1].iloc[0]
    p2_data = df_ranking[df_ranking['jugador'] == jugador2].iloc[0]

    st.markdown("---")

    # --- FOTOS ROBUSTAS CON LIBRERÍA PIL ---
    col_img1, col_img2 = st.columns(2)
    silueta_default = "https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png"
    
    def obtener_foto(nombre_jugador):
        extensiones = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
        for ext in extensiones:
            # Buscamos en la carpeta local "fotos/"
            ruta = os.path.join("fotos", f"{nombre_jugador}{ext}")
            if os.path.exists(ruta):
                try:
                    # Usamos Image de PIL para asegurar la lectura local
                    img = Image.open(ruta)
                    return img
                except:
                    pass
        return silueta_default

    with col_img1:
        _, c1, _ = st.columns([1, 2, 1])
        with c1: st.image(obtener_foto(jugador1), use_container_width=True, caption=jugador1)

    with col_img2:
        _, c2, _ = st.columns([1, 2, 1])
        with c2: st.image(obtener_foto(jugador2), use_container_width=True, caption=jugador2)

    # --- VELOCÍMETROS ---
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_g1 = go.Figure(go.Indicator(
            mode = "gauge+number", value = p1_data['Indice_Barca_Final'],
            title = {'text': f"Score: {jugador1}", 'font': {'color': BARCA_RED}},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_RED}}
        ))
        fig_g1.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g1, use_container_width=True)
        
    with col_g2:
        fig_g2 = go.Figure(go.Indicator(
            mode = "gauge+number", value = p2_data['Indice_Barca_Final'],
            title = {'text': f"Score: {jugador2}", 'font': {'color': BARCA_BLUE}},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_BLUE}}
        ))
        fig_g2.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_g2, use_container_width=True)

    st.markdown("---")

    # --- RADAR Y SCATTER ---
    col_rad, col_sca = st.columns(2)
    
    with col_rad:
        st.subheader("Radar de Perfil (Percentiles)")
        radar_cols = ['xg', 'xa', 'oport_creadas', 'regates_pct', 'recuperaciones', 'toques_area_rival']
        nombres_ejes = ['Goles Esp. (xG)', 'Asist. Esp. (xA)', 'Creación', 'Regate %', 'Recuperaciones', 'Toques Área']
        
        radar_cols_validas = [c for c in radar_cols if c in df.columns]
        
        if radar_cols_validas:
            df_percentiles = df[radar_cols_validas].rank(pct=True) * 100
            df_percentiles['jugador'] = df['jugador']
            
            p1_radar = df_percentiles[df_percentiles['jugador'] == jugador1].iloc[0][radar_cols_validas].tolist()
            p2_radar = df_percentiles[df_percentiles['jugador'] == jugador2].iloc[0][radar_cols_validas].tolist()
            
            p1_radar += p1_radar[:1]
            p2_radar += p2_radar[:1]
            ejes_validos = [nombres_ejes[radar_cols.index(c)] for c in radar_cols_validas]
            ejes_validos += ejes_validos[:1]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=p1_radar, theta=ejes_validos, fill='toself', name=jugador1, line_color=BARCA_RED, fillcolor='rgba(165, 0, 68, 0.4)'))
            fig_radar.add_trace(go.Scatterpolar(r=p2_radar, theta=ejes_validos, fill='toself', name=jugador2, line_color=BARCA_BLUE, fillcolor='rgba(0, 77, 152, 0.4)'))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(255, 255, 255, 0.5)'), 
                showlegend=True, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with col_sca:
        st.subheader("Creadores vs Rematadores")
        if 'xg' in df.columns and 'xa' in df.columns:
            fig_xgxa = go.Figure()
            
            fig_xgxa.add_trace(go.Scatter(x=df_ranking['xg'], y=df_ranking['xa'], mode='markers', marker=dict(size=8, color='gray', opacity=0.5), hovertext=df_ranking['jugador'], name="Resto"))
            fig_xgxa.add_trace(go.Scatter(x=[p1_data['xg']], y=[p1_data['xa']], mode='markers+text', text=[jugador1], textposition='top center', marker=dict(size=18, color=BARCA_RED, line=dict(width=2, color=BARCA_YELLOW)), name=jugador1))
            fig_xgxa.add_trace(go.Scatter(x=[p2_data['xg']], y=[p2_data['xa']], mode='markers+text', text=[jugador2], textposition='bottom center', marker=dict(size=18, color=BARCA_BLUE, line=dict(width=2, color=BARCA_YELLOW)), name=jugador2))
            
            fig_xgxa.update_layout(
                xaxis_title="Goles Esperados (xG)", yaxis_title="Asistencias Esperadas (xA)", 
                showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_xgxa, use_container_width=True)









