import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata

# --- 1. CONFIGURACIÓN Y CSS (Fondo Degradado) ---
st.set_page_config(page_title="Dashboard Scouting Barça", layout="wide", page_icon="🔵🔴")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Fondo principal: Degradado Grana a Azul */
    .stApp {
        background: linear-gradient(135deg, #A50044 0%, #004D98 100%) !important;
        background-attachment: fixed !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 0, 0, 0.3) !important; 
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #FFFFFF !important; 
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important; 
        border-bottom: 4px solid #EDBB00 !important; 
    }
    
    h1, h2, h3, h4, .stMarkdown p {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }
    </style>
""", unsafe_allow_html=True)

BARCA_BLUE = "#004D98"
BARCA_RED = "#A50044"
BARCA_YELLOW = "#EDBB00"

# --- ESCUDO DEL BARÇA ---
col_logo1, col_logo2, col_logo3 = st.columns([4, 1, 4])
with col_logo2:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/300px-FC_Barcelona_%28crest%29.svg.png", use_container_width=True)

# --- 2. CARGA Y FUSIÓN DE DATOS ---
def limpiar_nombre(nombre):
    if pd.isna(nombre): return ""
    nombre_limpio = unicodedata.normalize('NFKD', str(nombre)).encode('ASCII', 'ignore').decode('utf-8')
    return nombre_limpio.strip().lower()

@st.cache_data
def load_data():
    try:
        # Obtener el directorio del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_main = os.path.join(script_dir, "dataset_fcb_actualizado_con_goles.csv")
        path_goles = os.path.join(script_dir, "evolucio_gols_dc.csv")
        
        df_main = pd.read_csv(path_main)
        try:
            df_goles = pd.read_csv(path_goles)
            
            # Limpiamos nombres para unirlos perfectamente
            df_main['match_name'] = df_main['jugador'].apply(limpiar_nombre)
            df_goles['match_name'] = df_goles['Jugadores'].apply(limpiar_nombre)
            
            # Columnas a traernos del excel de goles
            cols_goles = ['match_name', 'Goles_5_y', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
            
            df = pd.merge(df_main, df_goles[cols_goles], on='match_name', how='left')
            df = df.drop(columns=['match_name'])
            
            # Rellenamos huecos con 0 si algún jugador no tiene datos
            df['Goles_5_y'] = df['Goles_5_y'].fillna(0)
            for col in ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']:
                df[col] = df[col].fillna(0)
                
        except Exception as e:
            st.warning("No se encontró el archivo de goles históricos. Usando solo datos base.")
            df = df_main
            df['Goles_5_y'] = 0
            for col in ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']: df[col] = 0
            
        return df
    except Exception as e:
        st.error(f"Error cargando datos principales: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- 3. CÁLCULO DEL ALGORITMO ---
stats_ataque = ['xg', 'xa', 'toques_area_rival', 'oport_creadas', 'gr_oport_creadas', 'regates_pct', 'Goles_5_y']
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
df['Indice_Barca_Final'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df[['Indice_Bruto']])

df_ranking = df.sort_values(by='Indice_Barca_Final', ascending=False).reset_index(drop=True)

# --- 4. ESTRUCTURA DE PESTAÑAS (TABS) ---
tab1, tab2, tab3 = st.tabs(["🏆 Ranking Top Fichajes", "📈 Gráficos e Insights", "⚔️ Cara a Cara (1vs1)"])

# ==================================
# PESTAÑA 1: RANKING (ESTILO LEADERBOARD)
# ==================================
with tab1:
    st.header("🏆 Clasificación Definitiva")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Creamos un diseño de lista con tarjetas en HTML/CSS
    for i, row in df_ranking.head(10).iterrows():
        puntos = row['Indice_Barca_Final']
        jugador = row['jugador']
        
        # Color diferente para el Top 1 (Oro), Top 2 (Plata), Top 3 (Bronce) y el resto
        if i == 0: color_borde = "#FFD700" # Oro
        elif i == 1: color_borde = "#C0C0C0" # Plata
        elif i == 2: color_borde = "#CD7F32" # Bronce
        else: color_borde = BARCA_BLUE
            
        st.markdown(f"""
        <div style="background-color: rgba(0, 0, 0, 0.4); padding: 15px 25px; border-radius: 10px; margin-bottom: 12px; border-left: 8px solid {color_borde}; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h3 style="margin: 0; font-size: 1.5rem;">#{i+1} &nbsp;&nbsp; {jugador}</h3>
            <h3 style="margin: 0; color: {BARCA_YELLOW}; font-size: 1.6rem;">{puntos:.1f} Pts</h3>
        </div>
        """, unsafe_allow_html=True)

# ==================================
# PESTAÑA 2: GRÁFICOS
# ==================================
with tab2:
    st.header("📊 Top 5 Jugadores por Categoría")
    
    diccionario_metricas = {
        'xg': 'Goles Esperados (xG)',
        'xa': 'Asistencias Esperadas (xA)',
        'oport_creadas': 'Oportunidades Creadas',
        'recuperaciones': 'Balones Recuperados',
        'regates_pct': 'Porcentaje Acierto Regates',
        'Goles_5_y': 'Goles Totales (Últimos 5 Años)'
    }
    
    metrica_seleccionada = st.selectbox("Selecciona la estadística:", options=list(diccionario_metricas.keys()), format_func=lambda x: diccionario_metricas[x])
    
    if metrica_seleccionada in df.columns:
        top5 = df.nlargest(5, metrica_seleccionada).sort_values(by=metrica_seleccionada, ascending=True)
        
        fig_bar = px.bar(
            top5, x=metrica_seleccionada, y='jugador', orientation='h',
            text=metrica_seleccionada,
            labels={'jugador': '', metrica_seleccionada: diccionario_metricas[metrica_seleccionada]}
        )
        fig_bar.update_traces(marker_color=BARCA_YELLOW, texttemplate='%{text:.2f}', textposition='outside', textfont=dict(color='white'))
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0.4)', height=350, font=dict(color='white'))
        st.plotly_chart(fig_bar, use_container_width=True)

# ==================================
# PESTAÑA 3: CARA A CARA
# ==================================
with tab3:
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

    # --- FOTOS ---
    col_img1, col_img2 = st.columns(2)
    silueta_default = "https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png"
    
    def obtener_ruta_foto(nombre):
        nombre_formateado = limpiar_nombre(nombre).replace(" ", "_")
        for ext in ['.png', '.jpg', '.jpeg']:
            ruta_formateada = f"fotos/{nombre_formateado}{ext}"
            if os.path.exists(ruta_formateada): return ruta_formateada
            ruta_original = f"fotos/{nombre.strip()}{ext}"
            if os.path.exists(ruta_original): return ruta_original
        return silueta_default

    with col_img1:
        _, c1, _ = st.columns([1, 2, 1])
        with c1: st.image(obtener_ruta_foto(jugador1), use_container_width=True, caption=jugador1)

    with col_img2:
        _, c2, _ = st.columns([1, 2, 1])
        with c2: st.image(obtener_ruta_foto(jugador2), use_container_width=True, caption=jugador2)

    # --- VELOCÍMETROS (AHORA MÁS GRANDES) ---
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_g1 = go.Figure(go.Indicator(
            mode = "gauge+number", value = p1_data['Indice_Barca_Final'],
            title = {'text': f"Puntos Barça", 'font': {'color': 'white'}},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_RED}}
        ))
        # Ajustamos height y margin para que no se corten los números
        fig_g1.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40), paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
        st.plotly_chart(fig_g1, use_container_width=True)
        
    with col_g2:
        fig_g2 = go.Figure(go.Indicator(
            mode = "gauge+number", value = p2_data['Indice_Barca_Final'],
            title = {'text': f"Puntos Barça", 'font': {'color': 'white'}},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_BLUE}}
        ))
        fig_g2.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40), paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
        st.plotly_chart(fig_g2, use_container_width=True)

    st.markdown("---")

    # --- NUEVO: GRÁFICO DE LÍNEAS DE EVOLUCIÓN GOLEADORA ---
    st.subheader("📈 Evolución Goleadora (Últimas 5 Temporadas)")
    temporadas = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    
    # Comprobamos que existan las columnas para no dar error
    if all(temp in df.columns for temp in temporadas):
        goles_p1 = p1_data[temporadas].tolist()
        goles_p2 = p2_data[temporadas].tolist()
        
        # Preparamos los datos para Plotly Line Chart
        df_lineas = pd.DataFrame({
            'Temporada': temporadas + temporadas,
            'Goles': goles_p1 + goles_p2,
            'Jugador': [jugador1]*5 + [jugador2]*5
        })
        
        fig_line = px.line(df_lineas, x='Temporada', y='Goles', color='Jugador', markers=True, 
                           color_discrete_sequence=[BARCA_RED, BARCA_BLUE],
                           labels={'Goles': 'Goles Marcados', 'Temporada': 'Temporada'})
        
        # Estética de la línea y marcadores
        fig_line.update_traces(line=dict(width=4), marker=dict(size=10, line=dict(width=2, color='white')))
        fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'), hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No hay datos de evolución por temporadas disponibles para estos jugadores.")

    st.markdown("---")

    # --- RADAR ---
    col_rad, col_sca = st.columns(2)
    with col_rad:
        st.subheader("Radar de Perfil")
        radar_cols = ['xg', 'xa', 'oport_creadas', 'regates_pct', 'recuperaciones', 'toques_area_rival']
        nombres_ejes = ['Goles Esp. (xG)', 'Asist. Esp. (xA)', 'Creación', 'Regate %', 'Recuperaciones', 'Toques Área']
        
        radar_cols_validas = [c for c in radar_cols if c in df.columns]
        
        if radar_cols_validas:
            df_percentiles = df[radar_cols_validas].rank(pct=True) * 100
            df_percentiles['jugador'] = df['jugador']
            
            p1_radar = df_percentiles[df_percentiles['jugador'] == jugador1].iloc[0][radar_cols_validas].tolist()
            p2_radar = df_percentiles[df_percentiles['jugador'] == jugador2].iloc[0][radar_cols_validas].tolist()
            
            p1_radar += p1_radar[:1]; p2_radar += p2_radar[:1]
            ejes_validos = [nombres_ejes[radar_cols.index(c)] for c in radar_cols_validas] + [nombres_ejes[radar_cols.index(radar_cols_validas[0])]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=p1_radar, theta=ejes_validos, fill='toself', name=jugador1, line_color=BARCA_RED, fillcolor='rgba(165, 0, 68, 0.6)'))
            fig_radar.add_trace(go.Scatterpolar(r=p2_radar, theta=ejes_validos, fill='toself', name=jugador2, line_color=BARCA_BLUE, fillcolor='rgba(0, 77, 152, 0.6)'))
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor='rgba(255, 255, 255, 0.9)'), showlegend=True, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
            st.plotly_chart(fig_radar, use_container_width=True)

    with col_sca:
        st.subheader("Creadores vs Rematadores")
        if 'xg' in df.columns and 'xa' in df.columns:
            fig_xgxa = go.Figure()
            fig_xgxa.add_trace(go.Scatter(x=df_ranking['xg'], y=df_ranking['xa'], mode='markers', marker=dict(size=8, color='white', opacity=0.3), hovertext=df_ranking['jugador'], name="Resto"))
            fig_xgxa.add_trace(go.Scatter(x=[p1_data['xg']], y=[p1_data['xa']], mode='markers+text', text=[jugador1], textposition='top center', marker=dict(size=18, color=BARCA_RED, line=dict(width=2, color=BARCA_YELLOW)), name=jugador1))
            fig_xgxa.add_trace(go.Scatter(x=[p2_data['xg']], y=[p2_data['xa']], mode='markers+text', text=[jugador2], textposition='bottom center', marker=dict(size=18, color=BARCA_BLUE, line=dict(width=2, color=BARCA_YELLOW)), name=jugador2))
            
            fig_xgxa.update_layout(xaxis_title="Goles Esperados (xG)", yaxis_title="Asistencias Esperadas (xA)", showlegend=False, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
            st.plotly_chart(fig_xgxa, use_container_width=True)










