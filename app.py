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

# --- 2. CSS MODERNO (PALETA ÉTICA & DARK MODE) ---
st.markdown("""
    <style>
        /* Variáveis de Cores Éticas */
        :root {
            --primary-color: #00bcd4;
            --warning-color: #ffc107;
            --background-color: #0e1117;
            --secondary-background-color: #262730;
            --text-color: #fafafa;
        }

        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        h1 {
            color: #00bcd4 !important;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 700;
            border-bottom: 1px solid #30333d;
            padding-bottom: 15px;
        }
        h2, h3 { color: #e0e0e0 !important; }

        /* CARDS */
        div[data-testid="stMetric"] {
            background-color: #1f2229;
            border: 1px solid #30333d;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        
        div[data-testid="stMetricLabel"] > label { color: #a0a0a0 !important; }
        div[data-testid="stMetricValue"] { color: #ffffff !important; }

        /* WIDGETS */
        .stRadio label, .stNumberInput label, .stSelectbox label, .stCheckbox label {
            color: #e0e0e0 !important;
        }
        
        section[data-testid="stSidebar"] {
            background-color: var(--secondary-background-color);
            border-right: 1px solid #30333d;
        }

        /* BOTÃO */
        div.stButton > button {
            background: linear-gradient(90deg, #00bcd4 0%, #00acc1 100%);
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            border: none;
            width: 100%;
            box-shadow: 0 4px 12px rgba(0, 188, 212, 0.3);
        }
        div.stButton > button:hover {
            box-shadow: 0 6px 16px rgba(0, 188, 212, 0.5);
            color: white;
        }

        /* ALERTAS */
        .stAlert {
            background-color: #262730;
            border: 1px solid #ffc107;
            border-radius: 8px;
            color: #ffc107;
        }
        
        /* Links no Texto */
        a {
            color: #00bcd4 !important;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        
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

st.warning("""
    ⚠️ **AVISO LEGAL: ESTA FERRAMENTA NÃO SUBSTITUI A ANÁLISE CLÍNICA.**
    
    Este sistema é um modelo estatístico de apoio à decisão. Resultados positivos indicam apenas 
    a necessidade de investigação aprofundada por um profissional de saúde qualificado.
""")

if modelo is None:
    st.error("⚠️ **Erro de Sistema:** Modelos de IA não carregados.")
    st.stop()

# --- 5. BARRA LATERAL ---
with st.sidebar:
    st.markdown("### 📋 Perfil do Paciente")
    idade = st.number_input("Idade (anos)", 1, 18, 6)
    genero = st.selectbox("Sexo Biológico", ["Masculino", "Feminino"])
    
    st.markdown("### Histórico")
    ictericia = st.checkbox("Histórico de Icterícia?")
    familia = st.checkbox("Casos de TEA na família?")
    
    st.markdown("---")
    with st.expander("ℹ️ Sobre o Modelo"):
        # AQUI ESTÁ A ATUALIZAÇÃO COM O LINK
        st.info("Baseado em SVM Linear com 100% de sensibilidade em testes controlados ([Artoni et al., 2022](https://sol.sbc.org.br/index.php/sbbd/article/view/30682)).")

# --- 6. FORMULÁRIO ---
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
    # Feedback visual rápido
    with st.spinner("Analisando perfil comportamental..."):
        time.sleep(0.6)

    # Mapeamento
    def p_dir(r): return 1 if r == "Sim" else 0
    def p_inv(r): return 1 if r == "Não" else 0
    
    scores = {
        'a1': p_dir(q1), 'a2': p_inv(q2), 'a3': p_inv(q3), 'a4': p_inv(q4), 'a5': p_inv(q5),
        'a6': p_inv(q6), 'a7': p_inv(q7), 'a8': p_inv(q8), 'a9': p_inv(q9), 'a10': p_dir(q10)
    }
    
    # DataFrame
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

    # Predição (Apenas Classe)
    try:
        X_input = scaler.transform(entrada)
        classe = modelo.predict(X_input)[0]
    except Exception as e:
        st.error(f"Erro de processamento: {e}")
        st.stop()

    score_total = sum(scores.values())
    
    # Critério Clínico de Segurança
    risco_elevado = (classe == 1) or (score_total >= 6)

    # --- EXIBIÇÃO CLÍNICA (2 COLUNAS) ---
    st.markdown("---")
    st.markdown("### 📊 Análise Clínica")
    
    # Layout centralizado de 2 colunas
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("Score AQ-10", f"{score_total}/10", help="Corte clínico sugerido: ≥ 6")
    
    with col_b:
        # Lógica de Cores Éticas
        if risco_elevado:
            lbl = "ATENÇÃO NECESSÁRIA"
            cor_texto = "#ffca28" # Amarelo Ouro
            icone = "⚠️"
        else:
            lbl = "BAIXA PROBABILIDADE"
            cor_texto = "#00bcd4" # Azul Ciano
            icone = "🔹"
            
        # Card Customizado HTML/CSS
        st.markdown(f"""
            <div style="background-color: #1f2229; border: 1px solid {cor_texto}40; border-radius: 12px; padding: 10px; text-align: center;">
                <span style="color: #a0a0a0; font-size: 13px;">Resultado da Triagem</span><br>
                <span style="color: {cor_texto}; font-size: 20px; font-weight: 700;">{icone} {lbl}</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("") 

    if risco_elevado:
        # Card Amarelo (Atenção)
        st.markdown(f"""
        <div style="background-color: #262730; border-left: 5px solid #ffca28; padding: 15px; border-radius: 5px;">
            <h4 style="color: #ffca28; margin:0;">⚠️ Rastreamento Positivo</h4>
            <p style="color: #e0e0e0; margin-top: 10px;">
                O perfil comportamental (Score {score_total}) apresenta correlação significativa com características do espectro.
            </p>
            <p style="color: #e0e0e0;"><strong>Recomendação:</strong> Encaminhar para avaliação com neuropediatra ou especialista para diagnóstico diferencial.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Card Azul (Informativo)
        st.markdown(f"""
        <div style="background-color: #262730; border-left: 5px solid #00bcd4; padding: 15px; border-radius: 5px;">
            <h4 style="color: #00bcd4; margin:0;">🔹 Rastreamento Negativo</h4>
            <p style="color: #e0e0e0; margin-top: 10px;">
                O padrão de respostas não sugere risco elevado no momento.
            </p>
            <p style="color: #e0e0e0;"><strong>Recomendação:</strong> Manter acompanhamento de rotina e observar marcos do desenvolvimento.</p>
        </div>
        """, unsafe_allow_html=True)