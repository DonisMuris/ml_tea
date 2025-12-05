import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO INICIAL E CSS (DESIGN SYSTEM) ---
st.set_page_config(
    page_title="Triagem TEA | AQ-10",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS PROFISSIONAL (CLINICAL LIGHT THEME)
st.markdown("""
    <style>
        /* Forçar Tema Claro Global */
        .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* Títulos e Cabeçalhos */
        h1 {
            color: #0e4d92; /* Azul Clínico */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 700;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }
        h2, h3 {
            color: #343a40;
            font-family: 'Segoe UI', sans-serif;
        }
        
        /* Textos e Labels */
        .stMarkdown p, .stRadio label, .stNumberInput label, .stSelectbox label {
            color: #212529 !important;
            font-size: 16px;
        }
        
        /* Barra Lateral */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #dee2e6;
        }
        
        /* Cards de Métricas (Resultados) */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        div[data-testid="stMetricLabel"] {
            color: #6c757d !important; /* Cinza médio */
            font-size: 14px;
        }
        div[data-testid="stMetricValue"] {
            color: #212529 !important; /* Quase preto */
            font-size: 26px;
            font-weight: 700;
        }
        
        /* Botão Principal */
        div.stButton > button {
            background-color: #0e4d92;
            color: white;
            font-weight: 600;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            border: none;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #0b3d75; /* Azul mais escuro */
            color: white;
            border: none;
        }
        
        /* Remove rodapés padrão */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. CARREGAMENTO DOS ARQUIVOS ---
@st.cache_resource
def carregar_modelo():
    try:
        return joblib.load('modelo_campeao.pkl'), joblib.load('scaler.pkl'), joblib.load('colunas.pkl')
    except: return None, None, None

modelo, scaler, colunas_treino = carregar_modelo()

# --- 3. CABEÇALHO DA APLICAÇÃO ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
with col_title:
    st.title("Sistema de Apoio à Decisão Clínica")
    st.markdown("**Protocolo:** AQ-10 (Autism Spectrum Quotient) | **Modelo:** SVM Linear")

if modelo is None:
    st.error("⚠️ **Erro Crítico:** Arquivos do modelo (.pkl) não encontrados. Verifique o diretório.")
    st.stop()

# --- 4. BARRA LATERAL (DADOS) ---
with st.sidebar:
    st.header("📋 Dados do Paciente")
    idade = st.number_input("Idade (anos)", min_value=1, max_value=18, value=6)
    genero = st.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
    
    st.subheader("Histórico")
    ictericia = st.checkbox("Nasceu com Icterícia?")
    familia = st.checkbox("Histórico familiar de TEA?")
    
    st.markdown("---")
    with st.expander("Sobre a Ferramenta"):
        st.caption("""
        Esta aplicação utiliza Machine Learning para triagem inicial. 
        Não substitui avaliação médica.
        **Desenvolvido para fins acadêmicos.**
        """)

# --- 5. FORMULÁRIO COMPORTAMENTAL ---
st.markdown("### 📝 Avaliação Comportamental")
st.info("Preencha as questões abaixo com base na observação direta da criança.")

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

# --- 6. LÓGICA E RESULTADOS ---
if submitted:
    # Barra de progresso para UX
    progress_text = "Processando vetores de características..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()

    # --- A. LÓGICA DE PONTUAÇÃO (AQ-10 Child) ---
    # Diretas (Sintoma = Sim): 1, 10
    # Inversas (Habilidade = Não): 2, 3, 4, 5, 6, 7, 8, 9
    def p_dir(r): return 1 if r == "Sim" else 0
    def p_inv(r): return 1 if r == "Não" else 0
    
    scores = {
        'a1': p_dir(q1), 'a2': p_inv(q2), 'a3': p_inv(q3), 'a4': p_inv(q4), 'a5': p_inv(q5),
        'a6': p_inv(q6), 'a7': p_inv(q7), 'a8': p_inv(q8), 'a9': p_inv(q9), 'a10': p_dir(q10)
    }
    
    # --- B. PREPARAÇÃO PARA IA ---
    # Mapeamento normalizado
    entrada = pd.DataFrame(columns=colunas_treino)
    entrada.loc[0] = 0
    colunas_map = {c.lower().strip(): c for c in colunas_treino}
    
    # Preenche Scores
    for key, val in scores.items():
        for col_lower, col_real in colunas_map.items():
            if key in col_lower and 'score' in col_lower:
                entrada.at[0, col_real] = val
                break
                
    # Preenche Demográficos
    for col_lower, col_real in colunas_map.items():
        if 'age' in col_lower: entrada.at[0, col_real] = idade
        if 'gender' in col_lower: entrada.at[0, col_real] = 1 if genero == "Masculino" else 0
        if 'jaundice' in col_lower: entrada.at[0, col_real] = 1 if ictericia else 0
        if 'austim' in col_lower or 'family' in col_lower: entrada.at[0, col_real] = 1 if familia else 0

    # --- C. PREDIÇÃO ---
    try:
        X_input = scaler.transform(entrada)
        prob = modelo.predict_proba(X_input)[0][1]
        classe = modelo.predict(X_input)[0]
    except Exception as e:
        st.error(f"Erro no processamento matemático: {e}")
        st.stop()

    score_total = sum(scores.values())

    # --- 7. EXIBIÇÃO DO LAUDO (DESIGN DE CARDS) ---
    st.markdown("---")
    st.markdown("### 📊 Resultado da Análise")
    
    # Lógica de Coerência Clínica (Safety Check)
    # Se Score >= 6, é considerado positivo pelo protocolo padrão, mesmo se a IA hesitar
    risco_elevado = (classe == 1) or (score_total >= 6)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico: ≥ 6 indica necessidade de investigação.")
    
    with col_b:
        lbl_ia = "POSITIVO" if risco_elevado else "NEGATIVO"
        # Usamos HTML customizado para garantir a cor do texto do status
        cor_status = "#d9534f" if risco_elevado else "#5cb85c" # Vermelho ou Verde
        st.markdown(f"""
            <div style="background-color: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 10px; text-align: center;">
                <label style="color: #6c757d; font-size: 14px;">Rastreamento Clínico</label>
                <div style="color: {cor_status}; font-size: 24px; font-weight: 700;">{lbl_ia}</div>
            </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.metric("Confiança do Modelo", f"{prob:.1%}", help="Probabilidade estatística baseada no treino.")

    st.write("") # Espaçamento

    # Card de Conclusão Final
    if risco_elevado:
        st.warning(f"""
        #### 🚩 Indicativo de Risco Identificado
        **Interpretação:** O perfil de respostas (Score {score_total}) apresenta correlação significativa com o Espectro Autista.
        
        **Conduta Sugerida:**
        1. Encaminhar para avaliação multidisciplinar (Neuropediatria/Psiquiatria Infantil).
        2. Aplicar instrumentos complementares de diagnóstico.
        """)
    else:
        st.success(f"""
        #### ✅ Baixa Probabilidade Identificada
        **Interpretação:** O padrão de respostas (Score {score_total}) é compatível com o desenvolvimento neurotípico.
        
        **Conduta Sugerida:**
        1. Manter acompanhamento de rotina.
        2. Reavaliar em 6 meses caso surjam novos sintomas.
        """)