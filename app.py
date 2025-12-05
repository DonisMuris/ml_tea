import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="Triagem TEA | AQ-10",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CSS MODERNO & DESIGN SYSTEM ---
st.markdown("""
    <style>
        /* Ajuste fino de fontes e espaçamentos */
        h1 {
            color: #00bcd4;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 700;
            letter-spacing: -1px;
        }
        h2, h3 {
            color: #e0e0e0;
            font-weight: 600;
        }
        
        /* CARDS CUSTOMIZADOS (Container de Resultado) */
        div[data-testid="stMetric"] {
            background-color: #1a1c24; /* Um pouco mais claro que o fundo */
            border: 1px solid #30333d;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        div[data-testid="stMetricLabel"] {
            color: #a0a0a0 !important; /* Cinza para o rótulo */
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important; /* Branco puro para o número */
        }

        /* AJUSTE DE INPUTS (RADIO BUTTONS) */
        /* Garante que o texto das perguntas seja legível */
        .stRadio label p {
            font-size: 16px;
            color: #e0e0e0 !important;
        }
        
        /* BOTÃO PRINCIPAL */
        div.stButton > button {
            background: linear-gradient(90deg, #00bcd4 0%, #008ba3 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            width: 100%;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
            color: white;
        }

        /* LINKS */
        a {
            color: #00bcd4 !important;
            text-decoration: none;
        }
        a:hover { text-decoration: underline; }

        /* RODAPÉ LIMPO */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# --- 3. CARREGAMENTO ---
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_campeao.pkl'), joblib.load('scaler.pkl'), joblib.load('colunas.pkl')
    except: return None, None, None

modelo, scaler, colunas_treino = carregar_modelo()

# --- 4. INTERFACE ---
st.title("Sistema de Apoio à Triagem TEA")
st.markdown("**Protocolo:** AQ-10 (Child/Adolescent) | **Engine:** SVM Linear")

st.info("""
    ℹ️ **AVISO DE USO:** Esta ferramenta é um modelo estatístico de apoio à decisão e **NÃO substitui a avaliação clínica**.
    O objetivo é agilizar a identificação de sinais de risco para encaminhamento precoce.
""")

if modelo is None:
    st.error("⚠️ **Erro de Sistema:** Arquivos do modelo não encontrados.")
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### 📋 Dados Clínicos")
    idade = st.number_input("Idade (anos)", 1, 18, 6)
    genero = st.selectbox("Sexo", ["Masculino", "Feminino"])
    
    st.markdown("---")
    st.markdown("### Histórico")
    ictericia = st.checkbox("Icterícia Neonatal?")
    familia = st.checkbox("Histórico Familiar de TEA?")
    
    st.markdown("---")
    with st.expander("📚 Fundamentação Científica"):
        st.markdown("""
        Modelo baseado em SVM Linear com **100% de sensibilidade** em validação cruzada.
        
        Referência: [Artoni et al. (2022)](https://sol.sbc.org.br/index.php/sbbd/article/view/30682)
        """)

# --- 6. QUESTIONÁRIO ---
st.markdown("### 📝 Avaliação Comportamental")
st.caption("Responda com base na observação do comportamento da criança.")

with st.form("form_aq10"):
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 🧠 Atenção e Padrões")
        q1 = st.radio("1. Percebe pequenos sons quando outros não?", ["Não", "Sim"], horizontal=True, key="q1")
        q2 = st.radio("2. Foca mais no todo do que em detalhes?", ["Não", "Sim"], horizontal=True, key="q2")
        q3 = st.radio("3. Consegue fazer mais de uma coisa ao mesmo tempo?", ["Não", "Sim"], horizontal=True, key="q3")
        q4 = st.radio("4. Se interrompida, volta rápido ao que fazia?", ["Não", "Sim"], horizontal=True, key="q4")
        q5 = st.radio("5. Sabe como manter uma conversa?", ["Não", "Sim"], horizontal=True, key="q5")
        
    with c2:
        st.markdown("##### 🗣️ Social e Comunicação")
        q6 = st.radio("6. É boa conversadora socialmente?", ["Não", "Sim"], horizontal=True, key="q6")
        q7 = st.radio("7. Entende personagens em histórias?", ["Não", "Sim"], horizontal=True, key="q7")
        q8 = st.radio("8. Gosta de jogos de 'faz de conta'?", ["Não", "Sim"], horizontal=True, key="q8")
        q9 = st.radio("9. Entende sentimentos pelo olhar?", ["Não", "Sim"], horizontal=True, key="q9")
        q10 = st.radio("10. Tem dificuldade em fazer novos amigos?", ["Não", "Sim"], horizontal=True, key="q10")
    
    st.markdown("###")
    submitted = st.form_submit_button("🔍 PROCESSAR TRIAGEM")

# --- 7. LÓGICA E RESULTADOS ---
if submitted:
    with st.spinner("Analisando padrões..."):
        time.sleep(0.5)

    # --- LÓGICA DE PONTUAÇÃO ---
    def p_dir(r): return 1 if r == "Sim" else 0
    def p_inv(r): return 1 if r == "Não" else 0
    
    scores = {
        'a1': p_dir(q1), 'a2': p_inv(q2), 'a3': p_inv(q3), 'a4': p_inv(q4), 'a5': p_inv(q5),
        'a6': p_inv(q6), 'a7': p_inv(q7), 'a8': p_inv(q8), 'a9': p_inv(q9), 'a10': p_dir(q10)
    }
    
    # --- PREPARAÇÃO ---
    entrada = pd.DataFrame(columns=colunas_treino)
    entrada.loc[0] = 0
    colunas_map = {c.lower().strip(): c for c in colunas_treino}
    
    for key, val in scores.items():
        for col_lower, col_real in colunas_map.items():
            if key in col_lower and 'score' in col_lower:
                entrada.at[0, col_real] = val
                break
                
    for col_lower, col_real in colunas_map.items():
        if 'age' in col_lower: entrada.at[0, col_real] = idade
        if 'gender' in col_lower: entrada.at[0, col_real] = 1 if genero == "Masculino" else 0
        if 'jaundice' in col_lower: entrada.at[0, col_real] = 1 if ictericia else 0
        if 'austim' in col_lower or 'family' in col_lower: entrada.at[0, col_real] = 1 if familia else 0

    # --- PREDIÇÃO ---
    try:
        X_input = scaler.transform(entrada)
        # prob = modelo.predict_proba(X_input)[0][1] # Opcional
        classe = modelo.predict(X_input)[0]
    except Exception as e:
        st.error(f"Erro técnico: {e}")
        st.stop()

    score_total = sum(scores.values())
    risco_elevado = (classe == 1) or (score_total >= 6)

    # --- EXIBIÇÃO CLÍNICA ---
    st.markdown("---")
    st.markdown("### 📊 Resultado da Análise")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico sugerido: ≥ 6")
    
    with c2:
        if risco_elevado:
            texto_status = "ATENÇÃO NECESSÁRIA"
            cor_status = "#ffca28" # Amarelo
            icone = "⚠️"
        else:
            texto_status = "BAIXA PROBABILIDADE"
            cor_status = "#00bcd4" # Azul
            icone = "🔹"
            
        # Card Customizado (HTML para garantir visual independente do tema)
        st.markdown(f"""
            <div style="
                background-color: #1a1c24; 
                border: 1px solid {cor_status}60; 
                border-radius: 8px; 
                padding: 10px; 
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <span style="color: #a0a0a0; font-size: 13px; font-weight: 500;">Rastreamento IA</span><br>
                <span style="color: {cor_status}; font-size: 22px; font-weight: 700;">{icone} {texto_status}</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("") 

    if risco_elevado:
        st.warning(f"""
        **Sinais de Risco Identificados (Score {score_total})**
        
        O algoritmo e a pontuação indicam correlação com características do Espectro Autista.
        
        **👉 Conduta Sugerida:**
        1. Encaminhamento para **Neuropediatria** ou **Psiquiatria Infantil**.
        2. Aplicação de avaliação diagnóstica completa (ex: ADOS-2, ADI-R).
        """)
    else:
        st.info(f"""
        **Perfil de Baixo Risco (Score {score_total})**
        
        O padrão de respostas é compatível com o desenvolvimento neurotípico esperado.
        
        **👉 Conduta Sugerida:**
        1. Manter acompanhamento pediátrico de rotina.
        2. Orientar pais sobre marcos do desenvolvimento.
        """)