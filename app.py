import streamlit as st
import folium
from streamlit_folium import st_folium
import networkx as nx
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import datetime

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="CyberLogic Tech - Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; color: white;}
    </style>
""", unsafe_allow_html=True)

st.title("🚀 CyberLogic Tech (CLT) - Painel de IA")
st.write("Fábrica de Software em Inteligência Artificial - Protótipos Oficiais")

aba1, aba2 = st.tabs(["🗺️ Logística (Roteamento A*)", "📈 Mercado Financeiro (Predição ML)"])

# =====================================================================
# ABA 1: MAPA LOGÍSTICO (A-ESTRELA)
# =====================================================================
with aba1:
    st.header("Protótipo Google Maps Avançado")
    
    locais = {
        'UAM_Paulista': [-23.555, -46.652], 'MASP': [-23.561, -46.655],
        'Bixiga': [-23.556, -46.645], 'Liberdade_Metro': [-23.556, -46.633],
        'Radial_Leste': [-23.548, -46.620], 'Parque_Aclimacao': [-23.570, -46.635],
        'UAM_Mooca': [-23.553, -46.598]
    }

    G = nx.Graph()
    for local, coord in locais.items():
        G.add_node(local, pos=coord)

    conexoes = [
        ('UAM_Paulista', 'MASP', 1.2), ('MASP', 'Bixiga', 1.5),
        ('Bixiga', 'Liberdade_Metro', 1.8), ('Liberdade_Metro', 'Radial_Leste', 2.5),
        ('Radial_Leste', 'UAM_Mooca', 2.2), ('UAM_Paulista', 'Parque_Aclimacao', 2.0),
        ('Parque_Aclimacao', 'UAM_Mooca', 3.5)
    ]
    G.add_weighted_edges_from(conexoes)

    # --- O SEGREDO 1: MEMÓRIA DA SESSÃO ---
    # Isso garante que o sistema lembre que o botão foi clicado
    if "rota_calculada" not in st.session_state:
        st.session_state.rota_calculada = False
        st.session_state.origem = 'UAM_Paulista'
        st.session_state.destino = 'UAM_Mooca'

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Painel de Controle de Rotas")
        origem = st.selectbox("📍 Origem:", list(locais.keys()), index=0)
        destino = st.selectbox("🏁 Destino:", list(locais.keys()), index=len(locais)-1)
        
        # O botão agora apenas salva o estado na memória
        if st.button("Calcular Rota Otimizada"):
            st.session_state.rota_calculada = True
            st.session_state.origem = origem
            st.session_state.destino = destino

    with col2:
        mapa = folium.Map(location=[-23.555, -46.620], zoom_start=14, tiles="CartoDB dark_matter")
        for nome, coord in locais.items():
            folium.CircleMarker(location=coord, radius=6, color="white").add_to(mapa)

        # Só desenha a linha verde se a memória disser que já foi calculado
        if st.session_state.rota_calculada:
            try:
                rota = nx.astar_path(G, st.session_state.origem, st.session_state.destino, weight='weight')
                dist = nx.path_weight(G, rota, weight='weight')
                
                st.success("Rota otimizada encontrada com sucesso!")
                st.metric("Distância Total Ponderada", f"{dist:.2f} km")
                st.write("**Sequência Algorítmica:** " + " ➔ ".join([p.replace("_", " ") for p in rota]))
                
                coordenadas_rota = [locais[ponto] for ponto in rota]
                folium.PolyLine(coordenadas_rota, color="#00ff00", weight=5).add_to(mapa)
            except nx.NetworkXNoPath:
                st.error("Nenhuma rota encontrada na malha.")
        
        # --- O SEGREDO 2: RETURNED_OBJECTS=[] ---
        # Impede que o mapa atualize a página e suma com os dados!
        st_folium(mapa, width=700, height=400, returned_objects=[])


# =====================================================================
# ABA 2: MERCADO FINANCEIRO (PREDIÇÃO)
# =====================================================================
# =====================================================================
# ABA 2: MERCADO FINANCEIRO (PREDIÇÃO)
# =====================================================================
with aba2:
    st.header("Inteligência Preditiva de Ativos (3 Anos)")
    
    col3, col4 = st.columns([1, 3])
    
    with col3:
        st.subheader("Configuração do Modelo")
        ativo = st.selectbox("Selecione o Ativo:", ["Euro (EURBRL=X)", "Bitcoin (BTC)"])
        # Truque de Mestre: Usar BTC-USD internamente pois o servidor BRL do Yahoo é instável
        simbolo = "EURBRL=X" if ativo == "Euro (EURBRL=X)" else "BTC-USD" 
        anos_historico = st.slider("Anos de Histórico para Treino:", 1, 5, 2)
        submit_pred = st.button("Executar Machine Learning")
        
    with col4:
        if submit_pred:
            with st.spinner('Acessando servidor e treinando a IA...'):
                try:
                    # 1. Puxando dados com a estrutura simples do professor (prepp.py)
                    hoje = datetime.date.today()
                    inicio = hoje - datetime.timedelta(days=anos_historico * 365)
                    df = yf.download(simbolo, start=inicio, end=hoje)
                    
                    if df.empty:
                        st.warning("O servidor do Yahoo Finance não retornou dados no momento. Tente novamente.")
                    else:
                        # 2. Tratamento blindado contra o bug de "MultiIndex" das versões novas do Pandas
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        
                        # 3. Tratamento blindado contra o erro de fuso horário
                        datas_historicas = pd.to_datetime(df.index)
                        if datas_historicas.tz is not None:
                            datas_historicas = datas_historicas.tz_localize(None)
                            
                        y = df['Close'].values.ravel()
                        X = np.array([(d - datas_historicas.min()).days for d in datas_historicas]).reshape(-1, 1)
                        
                        # Treinamento ML
                        modelo = LinearRegression()
                        modelo.fit(X, y)
                        
                        # Predição Matemática para 1095 dias (3 Anos)
                        dias_futuros = 1095
                        ultimo_dia = X[-1][0]
                        X_futuro = np.array([[ultimo_dia + i] for i in range(1, dias_futuros + 1)])
                        datas_futuras = [hoje + datetime.timedelta(days=i) for i in range(1, dias_futuros + 1)]
                        y_futuro = modelo.predict(X_futuro)
                        
                        # Renderização Gráfica Interativa
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=datas_historicas, y=y, mode='lines', name='Histórico Real', line=dict(color='#00ff00')))
                        fig.add_trace(go.Scatter(x=datas_futuras, y=y_futuro, mode='lines', name='Tendência', line=dict(color='#ff0000', dash='dash')))
                        
                        fig.update_layout(title=f"Projeção de Fechamento: {ativo}", xaxis_title="Data", yaxis_title="Valor", template="plotly_dark", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("Sucesso! Modelo preditivo rodando liso sem erros de dados.")
                
                except Exception as e:
                    # Se algo muito absurdo acontecer, o erro exato será escrito na sua tela em formato de código!
                    st.error("O sistema bloqueou uma falha nos dados da bolsa. Detalhe do erro abaixo:")
                    st.code(str(e)) #py -m streamlit run app.py