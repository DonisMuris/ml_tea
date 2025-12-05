import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO (UI PRO) ---
st.set_page_config(page_title="Triagem TEA (AQ-10)", page_icon="⚕️", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2c3e50; font-family: sans-serif; font-weight: 700;}
    .stMetric {background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    div.stButton > button {background-color: #005b96; color: white; width: 100%; border-radius: 5px; padding: 12px;}
    div.stButton > button:hover {background-color: #034066; color: white; border: none;}
    </style>
""", unsafe_allow_html=True)

# --- 2. CARREGAMENTO ---
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_campeao.pkl'), joblib.load('scaler.pkl'), joblib.load('colunas.pkl')
    except: return None, None, None

modelo, scaler, colunas_treino = carregar_modelo()

# --- 3. CABEÇALHO ---
st.title("Sistema de Apoio à Decisão Clínica")
st.caption("Protocolo: AQ-10 (Autism Spectrum Quotient) | Modelo: SVM Linear")

if modelo is None:
    st.error("⚠️ Erro crítico: Arquivos do modelo não encontrados. Faça o upload dos arquivos .pkl.")
    st.stop()

# --- 4. INPUTS ---
with st.sidebar:
    st.header("Perfil do Paciente")
    idade = st.number_input("Idade", 1, 18, 6)
    genero = st.radio("Sexo", ["Masculino", "Feminino"])
    ictericia = st.toggle("Icterícia ao nascer?")
    familia = st.toggle("Histórico familiar de TEA?")

st.subheader("Avaliação Comportamental")
with st.form("form_aq10"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Atenção e Detalhes**")
        q1 = st.radio("1. Percebe pequenos sons quando outros não?", ["Não", "Sim"], horizontal=True, key="q1")
        q2 = st.radio("2. Foca mais no todo do que em detalhes?", ["Não", "Sim"], horizontal=True, key="q2")
        q3 = st.radio("3. Consegue fazer mais de uma coisa ao mesmo tempo?", ["Não", "Sim"], horizontal=True, key="q3")
        q4 = st.radio("4. Se interrompida, volta rápido ao que fazia?", ["Não", "Sim"], horizontal=True, key="q4")
        q5 = st.radio("5. Sabe como manter uma conversa?", ["Não", "Sim"], horizontal=True, key="q5")
    with c2:
        st.markdown("**Social e Comunicação**")
        q6 = st.radio("6. É boa conversadora socialmente?", ["Não", "Sim"], horizontal=True, key="q6")
        q7 = st.radio("7. Entende personagens em histórias?", ["Não", "Sim"], horizontal=True, key="q7")
        q8 = st.radio("8. Gosta de jogos de 'faz de conta'?", ["Não", "Sim"], horizontal=True, key="q8")
        q9 = st.radio("9. Entende sentimentos pelo olhar?", ["Não", "Sim"], horizontal=True, key="q9")
        q10 = st.radio("10. Tem dificuldade em fazer novos amigos?", ["Não", "Sim"], horizontal=True, key="q10")
    
    st.markdown("---")
    submitted = st.form_submit_button("PROCESSAR ANÁLISE")

# --- 5. PROCESSAMENTO BLINDADO ---
if submitted:
    # Mapeamento Lógico (AQ-10 Child)
    # Perguntas diretas (Sim=1): 1, 10
    # Perguntas inversas (Não=1): 2, 3, 4, 5, 6, 7, 8, 9
    def pontuar(resp, tipo): return 1 if resp == tipo else 0
    
    # Vetor de Score
    v = {
        'a1': pontuar(q1, "Sim"), 'a2': pontuar(q2, "Não"), 'a3': pontuar(q3, "Não"),
        'a4': pontuar(q4, "Não"), 'a5': pontuar(q5, "Não"), 'a6': pontuar(q6, "Não"),
        'a7': pontuar(q7, "Não"), 'a8': pontuar(q8, "Não"), 'a9': pontuar(q9, "Não"),
        'a10': pontuar(q10, "Sim")
    }
    
    # Dados Clínicos
    dados = {
        'age': idade,
        'gender_m': 1 if genero == "Masculino" else 0,
        'jaundice_yes': 1 if ictericia else 0,
        'austim_yes': 1 if familia else 0
    }
    
    # Mapeamento para as colunas do modelo (AQUI ESTAVA O ERRO ANTES)
    # Criamos um dicionário robusto que mapeia "a1" para "a1_score"
    entrada_modelo = pd.DataFrame(columns=colunas_treino)
    entrada_modelo.loc[0] = 0 # Inicializa com zeros
    
    # Preenchimento inteligente
    for col_treino in colunas_treino:
        col_lower = col_treino.lower()
        
        # Mapeia Scores
        if 'a1_' in col_lower or 'a1score' in col_lower: entrada_modelo.at[0, col_treino] = v['a1']
        elif 'a2_' in col_lower: entrada_modelo.at[0, col_treino] = v['a2']
        elif 'a3_' in col_lower: entrada_modelo.at[0, col_treino] = v['a3']
        elif 'a4_' in col_lower: entrada_modelo.at[0, col_treino] = v['a4']
        elif 'a5_' in col_lower: entrada_modelo.at[0, col_treino] = v['a5']
        elif 'a6_' in col_lower: entrada_modelo.at[0, col_treino] = v['a6']
        elif 'a7_' in col_lower: entrada_modelo.at[0, col_treino] = v['a7']
        elif 'a8_' in col_lower: entrada_modelo.at[0, col_treino] = v['a8']
        elif 'a9_' in col_lower: entrada_modelo.at[0, col_treino] = v['a9']
        elif 'a10_' in col_lower: entrada_modelo.at[0, col_treino] = v['a10']
        
        # Mapeia Demográficos
        elif 'age' in col_lower: entrada_modelo.at[0, col_treino] = dados['age']
        elif 'gender' in col_lower: entrada_modelo.at[0, col_treino] = dados['gender_m']
        elif 'jaundice' in col_lower: entrada_modelo.at[0, col_treino] = dados['jaundice_yes']
        elif 'austim' in col_lower or 'family' in col_lower: entrada_modelo.at[0, col_treino] = dados['austim_yes']

    # Predição
    X_input = scaler.transform(entrada_modelo)
    prob = modelo.predict_proba(X_input)[0][1]
    classe = modelo.predict(X_input)[0]
    score_total = sum(v.values())

    # --- 6. RESULTADO ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico sugerido: ≥ 6")
    
    with col2:
        cor = "normal" if classe == 0 else "inverse"
        label = "NEGATIVO" if classe == 0 else "POSITIVO"
        st.metric("Rastreamento IA", label, delta_color=cor)
        
    with col3:
        st.metric("Probabilidade TEA", f"{prob:.1%}")

    # Lógica de coerência visual
    if score_total >= 6 and classe == 0:
        st.warning("⚠️ **Nota:** O Score está alto (≥6), mas a IA ponderou outros fatores (ex: idade/histórico) e sugeriu baixo risco. Prevalece a cautela clínica: investigue.")
    elif classe == 1:
        st.error(f"**Indicativo de TEA Detectado.** O padrão de respostas (Score {score_total}) é altamente compatível com o espectro.")
    else:
        st.success("**Baixo Risco Detectado.** O padrão de respostas é compatível com desenvolvimento neurotípico.")