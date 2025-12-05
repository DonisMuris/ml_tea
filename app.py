import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO DE PÁGINA E ESTILO (UI PRO) ---
st.set_page_config(
    page_title="AQ-10 Screening Tool",
    page_icon="⚕️",
    layout="centered", # Foco central para leitura tipo formulário médico
    initial_sidebar_state="expanded"
)

# CSS Customizado para remover "cara de Streamlit" e dar tom clínico
st.markdown("""
    <style>
    /* Fonte e Cores */
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
    }
    /* Cards de Resultado */
    .stAlert {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    /* Botões */
    div.stButton > button {
        background-color: #005b96;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #034066;
        border: none;
        color: white;
    }
    /* Remover elementos padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2. CARREGAMENTO ROBUSTO ---
@st.cache_resource
def carregar_modelo():
    try:
        modelo = joblib.load('modelo_campeao.pkl')
        scaler = joblib.load('scaler.pkl')
        colunas = joblib.load('colunas.pkl')
        return modelo, scaler, colunas
    except FileNotFoundError:
        return None, None, None

modelo, scaler, colunas_treino = carregar_modelo()

# --- 3. CABEÇALHO E DISCLAIMER (UX ÉTICO) ---
st.title("Sistema de Apoio à Decisão Clínica")
st.caption("Protocolo: AQ-10 (Autism Spectrum Quotient) | Público: Pediátrico/Adolescente")

with st.expander("ℹ️  Aviso Legal e Termos de Uso", expanded=False):
    st.info("""
    **Este sistema é uma ferramenta de triagem baseada em estatística e NÃO substitui o diagnóstico médico.**
    
    A Inteligência Artificial utiliza padrões aprendidos de casos prévios para estimar probabilidades. 
    Resultados positivos indicam apenas a necessidade de investigação aprofundada por equipe multidisciplinar.
    """)

# Se os arquivos não existirem, para tudo de forma elegante
if modelo is None:
    st.error("⚠️ **Erro de Configuração:** Arquivos do modelo não encontrados no servidor.")
    st.stop()

# --- 4. BARRA LATERAL (DADOS DEMOGRÁFICOS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=50) # Ícone médico genérico
    st.markdown("### Perfil do Paciente")
    st.markdown("---")
    
    idade = st.number_input("Idade (anos)", min_value=1, max_value=18, value=6, help="Idade cronológica do paciente.")
    genero = st.radio("Sexo Biológico", ["Masculino", "Feminino"], horizontal=True)
    
    st.markdown("### Histórico Clínico")
    ictericia = st.toggle("Nasceu com Icterícia?")
    familia = st.toggle("Histórico familiar de TEA?")
    
    st.markdown("---")
    st.caption("v.1.0.2 | Modelo: SVM Linear")

# --- 5. FORMULÁRIO PRINCIPAL (UX: EVITA RECARREGAR A PÁGINA) ---
st.subheader("Avaliação Comportamental")
st.write("Preencha com base na observação direta ou relato dos responsáveis.")

# Uso de st.form para agrupar inputs e só processar no final
with st.form("formulario_triagem"):
    
    # Layout em duas colunas para não ficar uma lista infinita
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Domínio: Atenção e Detalhes**")
        q1 = st.radio("1. Percebe pequenos sons quando outros não?", ["Não", "Sim"], horizontal=True, key="q1")
        q2 = st.radio("2. Foca mais no todo do que em detalhes?", ["Não", "Sim"], horizontal=True, key="q2")
        q3 = st.radio("3. Consegue fazer mais de uma coisa ao mesmo tempo?", ["Não", "Sim"], horizontal=True, key="q3")
        q4 = st.radio("4. Se interrompida, consegue voltar rápido ao que fazia?", ["Não", "Sim"], horizontal=True, key="q4")
        q5 = st.radio("5. Sabe como manter uma conversa com seus pares?", ["Não", "Sim"], horizontal=True, key="q5")

    with col2:
        st.markdown("**Domínio: Social e Comunicação**")
        q6 = st.radio("6. É boa conversadora socialmente?", ["Não", "Sim"], horizontal=True, key="q6")
        q7 = st.radio("7. Entende personagens ao ler uma história?", ["Não", "Sim"], horizontal=True, key="q7")
        q8 = st.radio("8. Gosta de jogos de 'faz de conta'?", ["Não", "Sim"], horizontal=True, key="q8")
        q9 = st.radio("9. Entende o que alguém sente olhando para o rosto?", ["Não", "Sim"], horizontal=True, key="q9")
        q10 = st.radio("10. Tem dificuldade em fazer novos amigos?", ["Não", "Sim"], horizontal=True, key="q10")

    st.markdown("---")
    submitted = st.form_submit_button("PROCESSAR ANÁLISE CLÍNICA")

# --- 6. PROCESSAMENTO E RESULTADOS (VISUALIZAÇÃO DE DADOS) ---
if submitted:
    with st.spinner("Computando vetores de características..."):
        time.sleep(0.8) # Pequeno delay proposital para sensação de processamento
        
        # 1. Preparação dos Dados
        mapa = {"Sim": 1, "Não": 0}
        
        # Mapeamento Booleano -> Binário
        ictericia_bin = 1 if ictericia else 0
        familia_bin = 1 if familia else 0
        genero_bin = 1 if genero == "Masculino" else 0
        
        dados_entrada = {
            'A1_Score': mapa[q1], 'A2_Score': mapa[q2], 'A3_Score': mapa[q3], 
            'A4_Score': mapa[q4], 'A5_Score': mapa[q5], 'A6_Score': mapa[q6], 
            'A7_Score': mapa[q7], 'A8_Score': mapa[q8], 'A9_Score': mapa[q9], 
            'A10_Score': mapa[q10],
            'age': idade,
            'gender_m': genero_bin,
            'jaundice_yes': ictericia_bin,
            'austim_yes': familia_bin
        }
        
        # Criação do DataFrame alinhado com o treino
        df_input = pd.DataFrame(columns=colunas_treino)
        df_input.loc[0] = 0 # Zera tudo inicial
        
        for col, valor in dados_entrada.items():
            # Busca fuzzy simples para garantir match das colunas
            cols_match = [c for c in colunas_treino if col.lower() in c.lower()]
            if cols_match:
                df_input.at[0, cols_match[0]] = valor
        
        # Normalização e Predição
        try:
            X_input = scaler.transform(df_input)
            probabilidade = modelo.predict_proba(X_input)[0][1]
            classe = modelo.predict(X_input)[0]
        except Exception as e:
            st.error(f"Erro técnico no processamento: {e}")
            st.stop()

    # --- 7. EXIBIÇÃO DO LAUDO TÉCNICO ---
    st.markdown("### 📊 Resultado da Triagem")
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    
    # Definição de cores e status baseados no limiar
    if classe == 1:
        status_cor = "inverse" # Vermelho/Destaque no st.metric
        msg_resultado = "RASTREAMENTO POSITIVO"
        icone = "🚩"
        cor_barra = ":red"
    else:
        status_cor = "normal"
        msg_resultado = "RASTREAMENTO NEGATIVO"
        icone = "✅"
        cor_barra = ":blue"

    with col_metric1:
        st.metric(label="Classificação do Algoritmo", value=msg_resultado)
    
    with col_metric2:
        st.metric(label="Probabilidade de TEA", value=f"{probabilidade:.1%}", delta_color=status_cor)
        
    with col_metric3:
        # Score AQ-10 simples (soma dos 1s) para referência
        score_aq = sum([mapa[q1], mapa[q2], mapa[q3], mapa[q4], mapa[q5], 
                        mapa[q6], mapa[q7], mapa[q8], mapa[q9], mapa[q10]])
        st.metric(label="Score Bruto (AQ-10)", value=f"{score_aq}/10")

    # Barra de Progresso Visual
    st.markdown(f"**Escala de Confiança do Modelo:**")
    st.progress(probabilidade)
    
    # Card de Conclusão
    if classe == 1:
        st.warning("""
        **Interpretação:**
        O padrão de respostas apresenta correlação significativa com características do Espectro Autista segundo o modelo preditivo.
        
        **Recomendação:**
        - Encaminhar para avaliação neuropsicológica.
        - Aplicar instrumentos complementares (M-CHAT, ADOS-2).
        """)
    else:
        st.success("""
        **Interpretação:**
        O padrão de respostas não indica, neste momento, prevalência significativa de traços do espectro.
        
        **Recomendação:**
        - Manter acompanhamento pediátrico regular.
        - Reavaliar se surgirem novos sintomas comportamentais.
        """)