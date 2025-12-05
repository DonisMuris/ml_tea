import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="Triagem TEA | AQ-10",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CSS MODERNO (FORÇAR DARK MODE & UI CLÍNICA) ---
# Aqui definimos variáveis globais para garantir contraste total
st.markdown("""
    <style>
        /* Forçar Variáveis de Cores do Streamlit (Override Global) */
        :root {
            --primary-color: #4f8bf9;
            --background-color: #0e1117;
            --secondary-background-color: #262730;
            --text-color: #fafafa;
            --font: "Source Sans Pro", sans-serif;
        }

        /* Fundo Principal */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        /* Títulos */
        h1 {
            color: #4f8bf9 !important; /* Azul Neon Suave */
            font-weight: 700;
            border-bottom: 1px solid #30333d;
            padding-bottom: 15px;
        }
        h2, h3 {
            color: #e0e0e0 !important;
        }

        /* --- CARDS DE RESULTADOS (MODERNIZADO) --- */
        div[data-testid="stMetric"] {
            background-color: #1f2229; /* Cinza Escuro Profundo */
            border: 1px solid #30333d;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #4f8bf9;
        }
        
        /* Rótulos dos Cards */
        div[data-testid="stMetricLabel"] > label {
            color: #a0a0a0 !important;
            font-size: 14px;
        }
        
        /* Valores dos Cards */
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 700;
        }

        /* --- INPUTS & WIDGETS --- */
        /* Garantir que textos de radio/checkbox sejam visíveis */
        .stRadio label, .stNumberInput label, .stSelectbox label, .stCheckbox label {
            color: #e0e0e0 !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            border-right: 1px solid #30333d;
        }

        /* Botão Principal (Gradiente Moderno) */
        div.stButton > button {
            background: linear-gradient(90deg, #4f8bf9 0%, #2d5cf6 100%);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: none;
            width: 100%;
            box-shadow: 0 4px 12px rgba(79, 139, 249, 0.4);
        }
        div.stButton > button:hover {
            box-shadow: 0 6px 16px rgba(79, 139, 249, 0.6);
            color: white;
        }

        /* Alertas Personalizados */
        .stAlert {
            background-color: #262730;
            border: 1px solid;
            border-radius: 8px;
        }
        
        /* Remove rodapés padrão */
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

# --- 4. CABEÇALHO ---
st.title("Sistema de Triagem TEA")
st.markdown("**Protocolo:** AQ-10 (Child/Adolescent) | **Engine:** SVM Linear")

if modelo is None:
    st.error("⚠️ **Erro de Sistema:** Modelos de IA não carregados. Verifique o repositório.")
    st.stop()

# --- 5. BARRA LATERAL (PERFIL) ---
with st.sidebar:
    st.markdown("### 📋 Perfil do Paciente")
    
    idade = st.number_input("Idade (anos)", min_value=1, max_value=18, value=6)
    genero = st.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
    
    st.markdown("### Histórico Clínico")
    ictericia = st.checkbox("Histórico de Icterícia?")
    familia = st.checkbox("Casos de TEA na família?")
    
    st.markdown("---")
    with st.expander("ℹ️ Sobre a IA", expanded=False):
        st.info("""
        Modelo treinado em base clínica validadas (Artoni et al., 2022).
        **Acurácia em Teste:** ~100% (Separação Linear).
        """)

# --- 6. FORMULÁRIO (AQ-10) ---
st.markdown("### 📝 Avaliação Comportamental")
st.caption("Preencha com base na observação direta do comportamento.")

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
    submitted = st.form_submit_button("PROCESSAR TRIAGEM")

# --- 7. LÓGICA E RESULTADOS ---
if submitted:
    # Barra de Progresso Estilizada
    progress_text = "Processando vetores..."
    my_bar = st.progress(0, text=progress_text)
    for p in range(100):
        time.sleep(0.003)
        my_bar.progress(p + 1, text=progress_text)
    my_bar.empty()

    # --- LÓGICA DE PONTUAÇÃO (AQ-10) ---
    def p_dir(r): return 1 if r == "Sim" else 0
    def p_inv(r): return 1 if r == "Não" else 0
    
    scores = {
        'a1': p_dir(q1), 'a2': p_inv(q2), 'a3': p_inv(q3), 'a4': p_inv(q4), 'a5': p_inv(q5),
        'a6': p_inv(q6), 'a7': p_inv(q7), 'a8': p_inv(q8), 'a9': p_inv(q9), 'a10': p_dir(q10)
    }
    
    # --- PREPARAÇÃO PARA IA ---
    entrada = pd.DataFrame(columns=colunas_treino)
    entrada.loc[0] = 0
    colunas_map = {c.lower().strip(): c for c in colunas_treino}
    
    # Mapeamento Inteligente
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
        prob = modelo.predict_proba(X_input)[0][1]
        classe = modelo.predict(X_input)[0]
    except Exception as e:
        st.error(f"Erro no cálculo vetorial: {e}")
        st.stop()

    score_total = sum(scores.values())
    risco_elevado = (classe == 1) or (score_total >= 6)

    # --- EXIBIÇÃO DO LAUDO (DESIGN ESCURO) ---
    st.markdown("---")
    st.markdown("### 📊 Análise Clínica")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico: ≥ 6")
    
    with col_b:
        lbl_ia = "POSITIVO" if risco_elevado else "NEGATIVO"
        # Usamos CSS inline para garantir a cor no modo escuro
        cor_texto = "#ff4b4b" if risco_elevado else "#00c853"
        
        st.markdown(f"""
            <div style="background-color: #1f2229; border: 1px solid #30333d; border-radius: 12px; padding: 10px; text-align: center;">
                <span style="color: #a0a0a0; font-size: 14px;">Rastreamento</span><br>
                <span style="color: {cor_texto}; font-size: 24px; font-weight: 700;">{lbl_ia}</span>
            </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.metric("Confiança IA", f"{prob:.1%}", help="Probabilidade calculada pelo SVM.")

    st.write("") 

    if risco_elevado:
        st.error(f"""
        #### 🚩 Indicativo de Risco Identificado
        **Interpretação:** O perfil (Score {score_total}) apresenta correlação significativa com o Espectro Autista.
        
        **Conduta Sugerida:**
        1. Encaminhar para **Neuropediatria** ou **Psiquiatria Infantil**.
        2. Aplicar instrumentos complementares (ex: M-CHAT, ADOS-2).
        """)
    else:
        st.success(f"""
        #### ✅ Baixa Probabilidade
        **Interpretação:** O padrão de respostas é compatível com o desenvolvimento neurotípico.
        
        **Conduta Sugerida:**
        1. Manter acompanhamento de rotina.
        2. Orientar responsáveis sobre marcos do desenvolvimento.
        """)