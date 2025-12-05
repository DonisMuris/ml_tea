import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Triagem TEA (AQ-10)", page_icon="🧩")

# --- CARREGAMENTO DOS ARQUIVOS ---
@st.cache_resource
def carregar_arquivos():
    modelo = joblib.load('modelo_campeao.pkl')
    scaler = joblib.load('scaler.pkl')
    colunas = joblib.load('colunas.pkl')
    return modelo, scaler, colunas

try:
    modelo, scaler, colunas_modelo = carregar_arquivos()
except FileNotFoundError:
    st.error("Erro: Arquivos .pkl não encontrados. Verifique se subiu modelo_campeao.pkl, scaler.pkl e colunas.pkl.")
    st.stop()

# --- INTERFACE ---
st.title("🧩 Sistema de Apoio à Triagem de TEA")
st.markdown("---")
st.write("""
Este sistema utiliza Inteligência Artificial (**SVM Linear**) para auxiliar na identificação 
de traços do Espectro Autista em crianças, baseado no protocolo **AQ-10**.
""")

with st.sidebar:
    st.header("Dados do Paciente")
    idade = st.number_input("Idade (Meses/Anos conforme base)", min_value=1, max_value=120, value=36)
    genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
    ictericia = st.selectbox("Nasceu com Icterícia?", ["Não", "Sim"])
    familia = st.selectbox("Casos de TEA na família?", ["Não", "Sim"])

st.subheader("Questionário Comportamental (AQ-10)")
st.info("Responda com base na observação do comportamento da criança.")

# Perguntas mapeadas (Ajuste os textos se quiser)
q1 = st.radio("A1. Percebe pequenos sons quando outros não?", ["Não", "Sim"], horizontal=True)
q2 = st.radio("A2. Foca mais no todo do que em detalhes?", ["Não", "Sim"], horizontal=True)
q3 = st.radio("A3. Consegue fazer mais de uma coisa ao mesmo tempo?", ["Não", "Sim"], horizontal=True)
q4 = st.radio("A4. Se interrompida, consegue voltar ao que estava fazendo?", ["Não", "Sim"], horizontal=True)
q5 = st.radio("A5. Sabe como manter uma conversa com seus pares?", ["Não", "Sim"], horizontal=True)
q6 = st.radio("A6. É boa conversadora socialmente?", ["Não", "Sim"], horizontal=True)
q7 = st.radio("A7. Entende personagens ao ler uma história?", ["Não", "Sim"], horizontal=True)
q8 = st.radio("A8. Gosta de jogos de 'faz de conta'?", ["Não", "Sim"], horizontal=True)
q9 = st.radio("A9. Entende o que alguém sente olhando para o rosto?", ["Não", "Sim"], horizontal=True)
q10 = st.radio("A10. Tem dificuldade em fazer novos amigos?", ["Não", "Sim"], horizontal=True)

# --- BOTÃO DE ANÁLISE ---
if st.button("🔍 Processar Triagem", type="primary"):
    
    # 1. Converter Respostas para Dados (0 ou 1)
    # ATENÇÃO: O AQ-10 tem pontuação invertida em algumas perguntas. 
    # Aqui assumimos a codificação direta do Dataset (Sim=1, Não=0).
    mapa = {"Sim": 1, "Não": 0}
    
    dados_entrada = {
        'A1_Score': mapa[q1], 'A2_Score': mapa[q2], 'A3_Score': mapa[q3], 
        'A4_Score': mapa[q4], 'A5_Score': mapa[q5], 'A6_Score': mapa[q6], 
        'A7_Score': mapa[q7], 'A8_Score': mapa[q8], 'A9_Score': mapa[q9], 
        'A10_Score': mapa[q10],
        'age': idade,
        # Variáveis dummy (precisam bater com o treino)
        'gender_m': 1 if genero == "Masculino" else 0,
        'jaundice_yes': 1 if ictericia == "Sim" else 0,
        'austim_yes': 1 if familia == "Sim" else 0
    }
    
    # 2. Criar DataFrame com TODAS as colunas do treino
    # (Preenche com 0 o que não foi perguntado para não quebrar o modelo)
    df_input = pd.DataFrame(columns=colunas_modelo)
    df_input.loc[0] = 0 # Inicializa com zeros
    
    # Preenche os valores coletados
    for col, valor in dados_entrada.items():
        # Tenta achar a coluna correspondente (ignora case sensitive)
        cols_match = [c for c in colunas_modelo if col.lower() in c.lower()]
        if cols_match:
            df_input.at[0, cols_match[0]] = valor

    # 3. Normalizar (Usando o scaler salvo)
    # Se o modelo foi treinado com dados escalados, a entrada também deve ser.
    try:
        dados_finais = scaler.transform(df_input)
    except:
        # Fallback se der erro de coluna, usa direto
        dados_finais = df_input

    # 4. Predição
    predicao = modelo.predict(dados_finais)[0]
    probabilidade = modelo.predict_proba(dados_finais)[0][1]

    # 5. Exibição do Resultado
    st.markdown("---")
    if predicao == 1:
        st.error(f"### 🚩 Resultado: INDICATIVO DE TEA")
        st.write(f"**Confiança do Modelo:** {probabilidade:.1%}")
        st.warning("⚠️ **Atenção:** Este resultado é uma triagem baseada em dados estatísticos e **NÃO** substitui o diagnóstico médico. Recomenda-se encaminhamento para avaliação multidisciplinar.")
    else:
        st.success(f"### ✅ Resultado: BAIXA PROBABILIDADE")
        st.write(f"**Confiança do Modelo:** {1-probabilidade:.1%}")
        st.write("O padrão de respostas não indica traços fortes no momento. Continue o acompanhamento padrão.")