import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO VISUAL (CORRIGIDA) ---
st.set_page_config(page_title="Triagem TEA (AQ-10)", page_icon="⚕️", layout="centered")

st.markdown("""
    <style>
    /* Fundo geral mais clínico (cinza bem claro) */
    .main {background-color: #f4f6f9;}
    
    /* Títulos */
    h1 {color: #1e3a8a; font-family: sans-serif; font-weight: 700;}
    h3 {color: #374151;}
    
    /* --- CORREÇÃO DOS CARDS (METRICS) --- */
    /* Container do card */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Rótulo (ex: "Score AQ-10") - Cinza Escuro */
    div[data-testid="stMetricLabel"] > label {
        color: #4b5563 !important;
        font-size: 14px;
    }
    
    /* Valor (ex: "8/10") - Preto Forte */
    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Botão */
    div.stButton > button {
        background-color: #2563eb; 
        color: white; 
        width: 100%; 
        border-radius: 6px; 
        padding: 12px;
        border: none;
        font-weight: 600;
    }
    div.stButton > button:hover {background-color: #1d4ed8; color: white;}
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
    st.error("⚠️ Erro: Arquivos .pkl não encontrados.")
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
    # A. Pontuação (Regra AQ-10 Child)
    # Diretas (Sim=1): 1, 10
    # Inversas (Não=1): 2, 3, 4, 5, 6, 7, 8, 9
    def p_dir(r): return 1 if r == "Sim" else 0
    def p_inv(r): return 1 if r == "Não" else 0
    
    respostas = {
        'a1': p_dir(q1), 'a2': p_inv(q2), 'a3': p_inv(q3), 'a4': p_inv(q4), 'a5': p_inv(q5),
        'a6': p_inv(q6), 'a7': p_inv(q7), 'a8': p_inv(q8), 'a9': p_inv(q9), 'a10': p_dir(q10)
    }
    
    # B. Preparação para o Modelo (Mapeamento Flexível)
    entrada = pd.DataFrame(columns=colunas_treino)
    entrada.loc[0] = 0 # Inicia zerado
    
    # Normaliza nomes para garantir o match
    colunas_map = {c.lower().strip(): c for c in colunas_treino}
    
    # Preenche Scores
    for key, val in respostas.items():
        # Procura variações: 'a1', 'a1_score', 'A1', etc
        for col_name_lower, col_real in colunas_map.items():
            if key in col_name_lower and 'score' in col_name_lower:
                entrada.at[0, col_real] = val
                break # Achou, para
                
    # Preenche Demográficos
    for col_name_lower, col_real in colunas_map.items():
        if 'age' in col_name_lower: entrada.at[0, col_real] = idade
        if 'gender' in col_name_lower: entrada.at[0, col_real] = 1 if genero == "Masculino" else 0
        if 'jaundice' in col_name_lower: entrada.at[0, col_real] = 1 if ictericia else 0
        if 'austim' in col_name_lower or 'family' in col_name_lower: entrada.at[0, col_real] = 1 if familia else 0

    # C. Predição
    X_input = scaler.transform(entrada)
    prob = modelo.predict_proba(X_input)[0][1]
    classe = modelo.predict(X_input)[0]
    score_total = sum(respostas.values())

    # --- 6. EXIBIÇÃO DE RESULTADOS ---
    st.markdown("### 📊 Resultado da Triagem")
    
    # Colunas para métricas
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico sugerido: ≥ 6")
        
    with c2:
        # Lógica de segurança: Se Score alto, mas IA deu negativo -> Alerta
        if score_total >= 6 and classe == 0:
            lbl = "INCONCLUSIVO"
            cor = "off"
        else:
            lbl = "POSITIVO" if classe == 1 else "NEGATIVO"
            cor = "inverse" if classe == 1 else "normal"
        st.metric("Rastreamento IA", lbl, delta_color=cor)
        
    with c3:
        st.metric("Probabilidade TEA", f"{prob:.1%}")

    # Barra e Feedback
    st.progress(prob)
    
    if classe == 1 or score_total >= 6:
        st.warning(f"""
        **Atenção:** O perfil comportamental (Score {score_total}) indica a necessidade de avaliação especializada.
        \nO algoritmo detectou padrões compatíveis com o espectro autista com **{prob:.1%} de confiança**.
        """)
    else:
        st.success(f"""
        **Baixo Risco:** O perfil atual não sugere traços significativos do espectro.
        \nScore: {score_total}/10 | Probabilidade IA: {prob:.1%}
        """)