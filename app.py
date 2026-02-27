import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata
import base64

# --- 1. CONFIGURACIÓN Y CSS (Fondo Degradado) ---
st.set_page_config(page_title="Dashboard Scouting Barça", layout="wide", page_icon="🔵🔴")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_main = os.path.join(script_dir, "dataset_fcb_actualizado_con_goles.csv")
        path_goles = os.path.join(script_dir, "evolucio_gols_dc.csv")
        
        df_main = pd.read_csv(path_main)
        
        df_goles = None
        if os.path.exists(path_goles):
            try:
                df_goles = pd.read_csv(path_goles)
                df_goles.columns = df_goles.columns.str.strip()
            except:
                df_goles = None
        
        if df_goles is not None:
            try:
                df_main['match_name'] = df_main['jugador'].apply(limpiar_nombre)
                df_goles['match_name'] = df_goles['Jugadores'].apply(limpiar_nombre)
                cols_goles = ['match_name', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
                df = pd.merge(df_main, df_goles[cols_goles], on='match_name', how='left')
                df = df.drop(columns=['match_name'])
                
                for col in ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']:
                    if col in df.columns:
                        df[col] = df[col].fillna(0).astype(int)
            except:
                df = df_main
                for col in ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']: df[col] = 0
        else:
            df = df_main
            for col in ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']: df[col] = 0
            
        return df
    except Exception as e:
        st.error(f"Error cargando datos principales: {e}")
        return pd.DataFrame()

@st.cache_data
def load_top5_data():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_top5 = os.path.join(script_dir, "dataset_top5.csv")
        
        if os.path.exists(path_top5):
            df_t5 = pd.read_csv(path_top5)
            df_t5.columns = df_t5.columns.str.strip()
            
            rename_dict = {
                'grandes_oport_creadas': 'gr_oport_creadas',
                'regates_realizados_pct': 'regates_pct',
                'posesion_tercio_ofen': 'pos_gan_tercio_of'
            }
            df_t5 = df_t5.rename(columns=rename_dict)
            
            for col in df_t5.columns:
                if col != 'jugador':
                    df_t5[col] = df_t5[col].astype(str).str.replace('%', '').str.replace(',', '.').astype(float)
            return df_t5
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error cargando top 5: {e}")
        return pd.DataFrame()

df = load_data()
df_top5_data = load_top5_data()

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

# --- FUNCIONES PARA FOTOS ---
silueta_default = "https://upload.wikimedia.org/wikipedia/commons/7/7c/Profile_avatar_placeholder_large.png"

def obtener_ruta_foto(nombre):
    # DICCIONARIO TRADUCTOR AUTOMÁTICO PARA EL TOP 5
    diccionario_top5 = {
        "K. Mbappé": "Kylian Mbappe",
        "E. Haaland": "Erling Haaland",
        "H. Kane": "Harry Kane",
        "V. Osimhen": "Victor Osimhen",
        "H. Ekitike": "Hugo Ekitike"
    }
    
    # Si el nombre viene abreviado, coge el nombre completo del diccionario
    nombre_final = diccionario_top5.get(nombre, nombre)
    
    nombre_formateado = limpiar_nombre(nombre_final).replace(" ", "_")
    for ext in ['.png', '.jpg', '.jpeg']:
        ruta_formateada = f"fotos/{nombre_formateado}{ext}"
        if os.path.exists(ruta_formateada): return ruta_formateada
        ruta_original = f"fotos/{nombre_final.strip()}{ext}"
        if os.path.exists(ruta_original): return ruta_original
    return silueta_default

def get_img_html(ruta):
    if ruta.startswith("http"): return ruta
    try:
        with open(ruta, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    except:
        return silueta_default

# --- 4. ESTRUCTURA DE PESTAÑAS (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Ranking", "📈 Gráficos", "⚔️ Cara a Cara", "🏅 Insignias", "🧬 Clonador Perfiles"])

# ==================================
# PESTAÑA 1: RANKING
# ==================================
with tab1:
    st.header("🏆 Clasificación Definitiva")
    st.markdown("<br>", unsafe_allow_html=True)
    
    for i, row in df_ranking.head(10).iterrows():
        puntos = row['Indice_Barca_Final']
        jugador = row['jugador']
        
        if i == 0: color_borde = "#FFD700"
        elif i == 1: color_borde = "#C0C0C0"
        elif i == 2: color_borde = "#CD7F32"
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
        fig_bar = px.bar(top5, x=metrica_seleccionada, y='jugador', orientation='h', text=metrica_seleccionada, labels={'jugador': '', metrica_seleccionada: diccionario_metricas[metrica_seleccionada]})
        fig_bar.update_traces(marker_color=BARCA_YELLOW, texttemplate='%{text:.2f}', textposition='outside', textfont=dict(color='white'))
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0.4)', height=350, font=dict(color='white'))
        st.plotly_chart(fig_bar, use_container_width=True)

# ==================================
# PESTAÑA 3: CARA A CARA
# ==================================
with tab3:
    st.header("⚔️ Comparativa Directa")
    col_s1, col_s2 = st.columns(2)
    with col_s1: jugador1 = st.selectbox("🔴 Jugador 1", df_ranking['jugador'].tolist(), index=0)
    with col_s2:
        idx_j2 = 1 if len(df_ranking) > 1 else 0
        jugador2 = st.selectbox("🔵 Jugador 2", df_ranking['jugador'].tolist(), index=idx_j2)
        
    p1_data = df_ranking[df_ranking['jugador'] == jugador1].iloc[0]
    p2_data = df_ranking[df_ranking['jugador'] == jugador2].iloc[0]

    st.markdown("---")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        _, c1, _ = st.columns([1, 2, 1])
        with c1: st.image(obtener_ruta_foto(jugador1), use_container_width=True, caption=jugador1)
    with col_img2:
        _, c2, _ = st.columns([1, 2, 1])
        with c2: st.image(obtener_ruta_foto(jugador2), use_container_width=True, caption=jugador2)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_g1 = go.Figure(go.Indicator(mode = "gauge+number", value = p1_data['Indice_Barca_Final'], title = {'text': f"Puntos Barça", 'font': {'color': 'white'}}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_RED}}))
        fig_g1.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40), paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
        st.plotly_chart(fig_g1, use_container_width=True)
    with col_g2:
        fig_g2 = go.Figure(go.Indicator(mode = "gauge+number", value = p2_data['Indice_Barca_Final'], title = {'text': f"Puntos Barça", 'font': {'color': 'white'}}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': BARCA_BLUE}}))
        fig_g2.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40), paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'))
        st.plotly_chart(fig_g2, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Evolución Goleadora (Últimas 5 Temporadas)")
    temporadas = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    if all(temp in df.columns for temp in temporadas):
        goles_p1 = p1_data[temporadas].tolist()
        goles_p2 = p2_data[temporadas].tolist()
        df_lineas = pd.DataFrame({'Temporada': temporadas + temporadas, 'Goles': goles_p1 + goles_p2, 'Jugador': [jugador1]*5 + [jugador2]*5})
        fig_line = px.line(df_lineas, x='Temporada', y='Goles', color='Jugador', markers=True, color_discrete_sequence=[BARCA_RED, BARCA_BLUE], labels={'Goles': 'Goles Marcados', 'Temporada': 'Temporada'})
        fig_line.update_traces(line=dict(width=4), marker=dict(size=10, line=dict(width=2, color='white')))
        fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0.4)', font=dict(color='white'), hovermode="x unified", yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
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

# ==================================
# PESTAÑA 4: INSIGNIAS
# ==================================
with tab4:
    st.header("🏅 Salón de Insignias")
    
    ganador_francotirador = df_ranking.nlargest(1, 'Goles_5_y').iloc[0] if 'Goles_5_y' in df_ranking.columns else df_ranking.iloc[0]
    ganador_mago = df_ranking.nlargest(1, 'oport_creadas').iloc[0] if 'oport_creadas' in df_ranking.columns else df_ranking.iloc[0]
    ganador_pulpo = df_ranking.nlargest(1, 'recuperaciones').iloc[0] if 'recuperaciones' in df_ranking.columns else df_ranking.iloc[0]
    
    df_jovenes = df_ranking[df_ranking['edad'] <= 22]
    ganador_joya = df_jovenes.nlargest(1, 'Indice_Barca_Final').iloc[0] if not df_jovenes.empty else df_ranking.iloc[0]
    ganador_motor = df_ranking.nlargest(1, 'minutos').iloc[0] if 'minutos' in df_ranking.columns else df_ranking.iloc[0]

    def dibujar_tarjeta(icono, titulo, jugador_row, nombre_stat, valor_stat, color_borde):
        foto_base64 = get_img_html(obtener_ruta_foto(jugador_row['jugador']))
        tarjeta_html = f"""
        <div style="background: rgba(0,0,0,0.5); border: 2px solid {color_borde}; border-radius: 15px; padding: 25px 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.4); margin-bottom: 20px;">
            <div style="font-size: 3.5rem; margin-bottom: 5px;">{icono}</div>
            <h3 style="color: {color_borde}; margin: 0 0 15px 0; text-transform: uppercase; letter-spacing: 2px; font-size: 1.2rem;">{titulo}</h3>
            <img src="{foto_base64}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 50%; border: 4px solid {color_borde}; margin-bottom: 15px; background-color: #fff;">
            <h2 style="color: white; margin: 0 0 10px 0; font-size: 1.6rem;">{jugador_row['jugador']}</h2>
            <div style="background-color: rgba(255,255,255,0.1); border-radius: 8px; padding: 10px;">
                <p style="color: #ccc; font-size: 0.9rem; margin: 0;">{nombre_stat}</p>
                <p style="color: {BARCA_YELLOW}; font-size: 1.5rem; font-weight: bold; margin: 0;">{valor_stat}</p>
            </div>
        </div>
        """
        st.markdown(tarjeta_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1: dibujar_tarjeta("🎯", "Francotirador", ganador_francotirador, "Goles (Últimos 5 años)", int(ganador_francotirador.get('Goles_5_y', 0)), BARCA_RED)
    with col2: dibujar_tarjeta("🎩", "El Mago", ganador_mago, "Oportunidades Creadas", round(ganador_mago.get('oport_creadas', 0), 2), BARCA_YELLOW)
    with col3: dibujar_tarjeta("🐙", "El Pulpo", ganador_pulpo, "Balones Recuperados", round(ganador_pulpo.get('recuperaciones', 0), 2), BARCA_BLUE)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col4, col5, _ = st.columns([1, 2, 2, 1])
    with col4: dibujar_tarjeta("💎", "La Joya", ganador_joya, f"Score Barça ({ganador_joya.get('edad', '-')} años)", round(ganador_joya.get('Indice_Barca_Final', 0), 1), "#00FF7F")
    with col5: dibujar_tarjeta("🏃", "El Motor", ganador_motor, "Minutos Jugados", int(ganador_motor.get('minutos', 0)), "#FF4500")

# ==================================
# PESTAÑA 5: BUSCADOR DE CLONES
# ==================================
with tab5:
    st.header("🧬 Buscador de Clones (Machine Learning)")
    st.markdown("Usando **Similitud del Coseno**, cruzamos todas las estadísticas (xG, asistencias, creación, recuperaciones) de las estrellas mundiales para encontrar a sus espejos exactos en tu base de datos.")
    
    if df_top5_data.empty:
        st.warning("⚠️ No se ha encontrado el archivo `dataset_top5.csv`. Asegúrate de subirlo a GitHub.")
    else:
        jugador_elite = st.selectbox("🌟 Selecciona la Estrella a clonar:", df_top5_data['jugador'].tolist())
        st.markdown("---")
        
        cols_comunes = [c for c in df_top5_data.columns if c in df.columns and c != 'jugador' and pd.api.types.is_numeric_dtype(df_top5_data[c])]
        
        if cols_comunes:
            df_calc = df[cols_comunes].fillna(0)
            df_t5_calc = df_top5_data[cols_comunes].fillna(0)
            
            data_to_scale = pd.concat([df_calc, df_t5_calc])
            scaler_clones = MinMaxScaler()
            scaled_data = scaler_clones.fit_transform(data_to_scale)
            
            df_scaled = scaled_data[:len(df_calc)]
            t5_scaled = scaled_data[len(df_calc):]
            
            idx_elite = df_top5_data[df_top5_data['jugador'] == jugador_elite].index[0]
            vector_elite = t5_scaled[idx_elite].reshape(1, -1)
            
            similitudes = cosine_similarity(df_scaled, vector_elite).flatten()
            
            df_clones = df.copy()
            df_clones['Similitud'] = similitudes * 100
            
            apellido = limpiar_nombre(jugador_elite).split()[-1]
            df_clones = df_clones[~df_clones['jugador'].apply(lambda x: apellido in limpiar_nombre(x))]
            
            top3 = df_clones.sort_values(by='Similitud', ascending=False).head(3)
            
            col_target, col_c1, col_c2, col_c3 = st.columns(4)
            
            with col_target:
                st.markdown("<h4 style='text-align: center; color: #FFF;'>🌟 MOLDE IDEAL</h4>", unsafe_allow_html=True)
                foto_t = get_img_html(obtener_ruta_foto(jugador_elite))
                tarjeta_html_t = f"""
                <div style="background: rgba(165,0,68,0.7); border: 2px solid {BARCA_YELLOW}; border-radius: 15px; padding: 25px 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.5);">
                    <img src="{foto_t}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 50%; border: 4px solid {BARCA_YELLOW}; margin-bottom: 15px; background-color: #fff;">
                    <h3 style="color: white; margin: 0; font-size: 1.5rem;">{jugador_elite}</h3>
                    <p style="color: {BARCA_YELLOW}; margin-top: 5px; font-weight: bold;">Élite Mundial</p>
                </div>
                """
                st.markdown(tarjeta_html_t, unsafe_allow_html=True)
                
            for i, (idx, row) in enumerate(top3.iterrows()):
                col_c = [col_c1, col_c2, col_c3][i]
                with col_c:
                    medalla = ["🥇 Clon #1", "🥈 Clon #2", "🥉 Clon #3"][i]
                    st.markdown(f"<h4 style='text-align: center; color: #FFF;'>{medalla}</h4>", unsafe_allow_html=True)
                    
                    foto_c = get_img_html(obtener_ruta_foto(row['jugador']))
                    sim = row['Similitud']
                    
                    color_c = "#00FF7F" if sim >= 85 else ("#C0C0C0" if sim >= 75 else BARCA_BLUE)
                    
                    tarjeta_html_c = f"""
                    <div style="background: rgba(0,0,0,0.5); border: 2px solid {color_c}; border-radius: 15px; padding: 25px 15px; text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,0.4);">
                        <h3 style="color: {color_c}; margin: 0 0 15px 0; font-size: 1.6rem;">{sim:.1f}% Match</h3>
                        <img src="{foto_c}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 50%; border: 4px solid {color_c}; margin-bottom: 15px; background-color: #fff;">
                        <h3 style="color: white; margin: 0 0 10px 0; font-size: 1.4rem;">{row['jugador']}</h3>
                        <p style="color: #ccc; font-size: 0.9rem; margin: 0;">{row['edad']} años | Score: {row['Indice_Barca_Final']:.1f}</p>
                    </div>
                    """
                    st.markdown(tarjeta_html_c, unsafe_allow_html=True)
        else:
            st.warning("No he encontrado estadísticas suficientes en común entre los Excels para poder compararlos.")
            











