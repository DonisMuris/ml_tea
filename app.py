import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- 1. CONFIGURAÇÃO DE PÁGINA E ESTILO (UI PRO) ---
st.set_page_config(
    page_title="Triagem TEA (AQ-10)",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS Customizado para aparência clínica
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    h1 {color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700;}
    h2, h3 {color: #34495e;}
    .stAlert {border-radius: 8px; border: 1px solid #e0e0e0;}
    div.stButton > button {
        background-color: #005b96; color: white; border-radius: 5px; border: none;
        padding: 10px 24px; font-size: 16px; width: 100%; transition: all 0.3s;
    }
    div.stButton > button:hover {background-color: #034066; color: white; border: none;}
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
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

# --- 3. CABEÇALHO ---
st.title("Sistema de Apoio à Decisão Clínica")
st.caption("Protocolo: AQ-10 (Autism Spectrum Quotient) | Público: Pediátrico/Adolescente")

with st.expander("ℹ️  Aviso Legal e Termos de Uso", expanded=False):
    st.info("""
    **Este sistema é uma ferramenta de triagem baseada em estatística e NÃO substitui o diagnóstico médico.**
    A IA utiliza padrões aprendidos para estimar probabilidades. Resultados positivos indicam apenas a necessidade de investigação clínica.
    """)

if modelo is None:
    st.error("⚠️ **Erro de Configuração:** Arquivos .pkl não encontrados. Verifique o upload no GitHub.")
    st.stop()

# --- 4. DADOS DEMOGRÁFICOS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=50)
    st.markdown("### Perfil do Paciente")
    idade = st.number_input("Idade (anos)", min_value=1, max_value=18, value=6)
    genero = st.radio("Sexo Biológico", ["Masculino", "Feminino"], horizontal=True)
    st.markdown("### Histórico Clínico")
    ictericia = st.toggle("Nasceu com Icterícia?")
    familia = st.toggle("Histórico familiar de TEA?")
    st.markdown("---")
    st.caption("v.1.1.0 (Logic Fix) | Modelo: SVM Linear")

# --- 5. FORMULÁRIO (AQ-10) ---
st.subheader("Avaliação Comportamental")
st.write("Preencha com base na observação direta.")

with st.form("formulario_triagem"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Domínio: Atenção e Detalhes**")
        # Perguntas que pontuam no SIM (Traço Autista)
        q1 = st.radio("1. Percebe pequenos sons quando outros não?", ["Não", "Sim"], horizontal=True, key="q1")
        # Perguntas que pontuam no NÃO (Habilidade Típica invertida)
        q2 = st.radio("2. Foca mais no todo do que em detalhes?", ["Não", "Sim"], horizontal=True, key="q2")
        q3 = st.radio("3. Consegue fazer mais de uma coisa ao mesmo tempo?", ["Não", "Sim"], horizontal=True, key="q3")
        q4 = st.radio("4. Se interrompida, consegue voltar rápido ao que fazia?", ["Não", "Sim"], horizontal=True, key="q4")
        # Pontua no SIM
        q5 = st.radio("5. Sabe como manter uma conversa com seus pares?", ["Não", "Sim"], horizontal=True, key="q5") 
        # Nota: Dependendo da tradução do dataset, a A5 pode ser inversa. 
        # Assumindo padrão "Dificuldade na conversa" = Sim (1 ponto). 
        # Se o texto for "Sabe manter conversa" = Não (1 ponto). 
        # Vou ajustar na lógica abaixo para garantir coerência.

    with col2:
        st.markdown("**Domínio: Social e Comunicação**")
        # Pontuam no NÃO (Habilidades Sociais Típicas)
        q6 = st.radio("6. É boa conversadora socialmente?", ["Não", "Sim"], horizontal=True, key="q6")
        q7 = st.radio("7. Entende personagens ao ler uma história?", ["Não", "Sim"], horizontal=True, key="q7")
        q8 = st.radio("8. Gosta de jogos de 'faz de conta'?", ["Não", "Sim"], horizontal=True, key="q8")
        q9 = st.radio("9. Entende o que alguém sente olhando para o rosto?", ["Não", "Sim"], horizontal=True, key="q9")
        # Pontua no SIM
        q10 = st.radio("10. Tem dificuldade em fazer novos amigos?", ["Não", "Sim"], horizontal=True, key="q10")

    st.markdown("---")
    submitted = st.form_submit_button("PROCESSAR ANÁLISE CLÍNICA")

# --- 6. PROCESSAMENTO COM LÓGICA CORRIGIDA ---
if submitted:
    with st.spinner("Analisando padrões vetoriais..."):
        time.sleep(0.5)
        
        # --- LÓGICA DE PONTUAÇÃO (CORREÇÃO AQUI) ---
        # 1 = Traço Autista Presente | 0 = Traço Ausente (Neurotípico)
        
        # Grupo A: Perguntas onde o comportamento autista é o "Sim" (Sintomas diretos)
        # A1 (Sons), A10 (Dificuldade Amigos)
        # A5 (ATENÇÃO: Se o texto é "Sabe manter conversa", o traço autista é NÃO. Se o texto fosse "Não sabe...", seria SIM.
        # Vamos assumir o texto da tela: "Sabe manter conversa?" -> Sim = 0, Não = 1 (Traço)
        
        # Mapeamento para perguntas de HABILIDADE (Sim=0, Não=1)
        # A2, A3, A4, A5, A6, A7, A8, A9
        def map_inverso(resposta): return 1 if resposta == "Não" else 0
        
        # Mapeamento para perguntas de SINTOMA (Sim=1, Não=0)
        # A1, A10
        def map_direto(resposta): return 1 if resposta == "Sim" else 0

        # Aplicação da Lógica (Baseado no padrão AQ-10 Child)
        val_a1 = map_direto(q1)
        val_a2 = map_inverso(q2)
        val_a3 = map_inverso(q3)
        val_a4 = map_inverso(q4)
        val_a5 = map_inverso(q5) # "Sabe manter conversa?" -> Não = Ponto
        val_a6 = map_inverso(q6) # "É boa conversadora?" -> Não = Ponto
        val_a7 = map_inverso(q7) # "Entende personagens?" -> Não = Ponto
        val_a8 = map_inverso(q8)
        val_a9 = map_inverso(q9) # "Entende sentimentos?" -> Não = Ponto
        val_a10 = map_direto(q10) # "Tem dificuldade?" -> Sim = Ponto

        # Dados Demográficos
        genero_bin = 1 if genero == "Masculino" else 0
        ictericia_bin = 1 if ictericia else 0
        familia_bin = 1 if familia else 0

        # Montagem do Dicionário (Nomes devem bater com o treino)
        dados_entrada = {
            'A1_Score': val_a1, 'A2_Score': val_a2, 'A3_Score': val_a3, 
            'A4_Score': val_a4, 'A5_Score': val_a5, 'A6_Score': val_a6, 
            'A7_Score': val_a7, 'A8_Score': val_a8, 'A9_Score': val_a9, 
            'A10_Score': val_a10,
            'age': idade,
            'gender_m': genero_bin,
            'jaundice_yes': ictericia_bin,
            'austim_yes': familia_bin
        }

        # Criação do DataFrame
        df_input = pd.DataFrame(columns=colunas_treino)
        df_input.loc[0] = 0 
        
        for col, valor in dados_entrada.items():
            cols_match = [c for c in colunas_treino if col.lower() in c.lower()]
            if cols_match:
                df_input.at[0, cols_match[0]] = valor
        
        # Predição
        try:
            X_input = scaler.transform(df_input)
            probabilidade = modelo.predict_proba(X_input)[0][1]
            classe = modelo.predict(X_input)[0]
        except Exception as e:
            st.error(f"Erro técnico: {e}")
            st.stop()

    # --- 7. EXIBIÇÃO ---
    st.markdown("### 📊 Resultado da Triagem")
    col1, col2, col3 = st.columns(3)
    
    # Cálculo do Score Visual (Soma dos 1s)
    score_visual = sum([val_a1, val_a2, val_a3, val_a4, val_a5, val_a6, val_a7, val_a8, val_a9, val_a10])

    if classe == 1:
        msg = "RASTREAMENTO POSITIVO"
        cor = "inverse"
    else:
        msg = "RASTREAMENTO NEGATIVO"
        cor = "normal"

    with col1: st.metric("Classificação IA", msg)
    with col2: st.metric("Probabilidade TEA", f"{probabilidade:.1%}", delta_color=cor)
    with col3: st.metric("Score AQ-10", f"{score_visual}/10", help="Ponto de corte sugerido: >= 6")

    st.progress(probabilidade)

    if classe == 1:
        st.warning("""
        **Interpretação:** O algoritmo identificou padrões comportamentais compatíveis com o Espectro Autista.
        \n**Recomendação:** Encaminhamento prioritário para avaliação diagnóstica especializada.
        """)
    else:
        st.success("""
        **Interpretação:** Não foram identificados padrões significativos de risco no momento.
        \n**Recomendação:** Manter acompanhamento de rotina.
        """)