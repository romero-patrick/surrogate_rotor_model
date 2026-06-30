# Surrogate Model Neural para Análise de Confiabilidade Paramétrica em Turbomáquinas (ROSS + PyTorch)

Este repositório contém o desenvolvimento de um **Surrogate Model (Modelo Substituto) Baseado em Redes Neurais Artificiais (MLP + Fourier Feature Mapping)** para prever curvas de Resposta em Frequência (FRF) de sistemas rotativos. O objetivo principal é acelerar simulações estocásticas de Monte Carlo para análise de confiabilidade de acordo com os critérios normativos da **API 617**.

## 🚀 O Problema e a Solução
Análises de Confiabilidade via Monte Carlo exigem milhares de avaliações do modelo físico (ROSS). Quando expandimos o espaço de busca dos mancais (rigidez e amortecimento) para **±70% de incerteza**, o custo computacional de resolver os autovalores e equações diferenciais a cada iteração torna-se proibitivo.

**A Solução:** Treinamos uma rede neural profunda capaz de mapear instantaneamente os parâmetros operacionais e geométricos diretamente na curva FRF, eliminando o solucionador físico do loop de simulação e reduzindo o tempo de processamento de minutos para poucos segundos.

---

## 🛠️ Arquitetura do Modelo Neural
Para capturar os picos de ressonância agudos e a alta não-linearidade do rotor no espectro de frequências, o pipeline utiliza:
* **Fourier Feature Mapping:** Uma camada de projeção randômica baseada em funções seno e cosseno para mapear as entradas de baixa dimensão em altas frequências, mitigando o viés espectral das MLPs tradicionais (*Spectral Bias*).
* **Deep MLP:** Rede totalmente conectada com normalização por lote (*BatchNorm*) e conexões densas para predição vetorial paralela de 1000 pontos de frequência de forma simultânea diretamente na GPU.
* **Escalonamento Logarítmico:** Tratamento robusto de dados para coeficientes dinâmicos ($10^3$ a $10^6$), evitando explosão de gradientes.

---

## 📊 Resultados e Validação Cruzada (ROSS vs IA)

O modelo foi treinado utilizando **10.000 amostras assimétricas** geradas via Distribuição Uniforme em um intervalo amplo de ±70% sobre as propriedades nominais dos mancais.

### Desempenho e Ganho de Velocidade
Substituir o resolvedor físico clássico pelo modelo surrogate em lote gerou uma aceleração monumental mantendo a precisão das curvas de convergência de Probabilidade de Falha ($P_f$).

* **Tempo de Simulação (ROSS - 10k Amostras):** 19h
* **Tempo de Predição (IA - 10k Amostras):** 3.35s
* **Ganho de Velocidade Real:** `X` 20419

## 🔧 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o ambiente virtual Python configurado com as bibliotecas necessárias:
```bash
pip install numpy scipy matplotlib torch ross-rotordynamics notebook