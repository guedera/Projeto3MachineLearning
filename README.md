# APS3 — Classificação: Previsão de Renda Anual

Projeto de Machine Learning desenvolvido para a disciplina de ML do Insper.  
O objetivo é prever se um indivíduo ganha **mais de $50 mil por ano** com base em características demográficas, educacionais e ocupacionais — usando o dataset [Adult Census Income (UCI)](https://archive.ics.uci.edu/ml/datasets/Adult).

---

## Como executar

```bash
# 1. Instalar dependências (gerenciadas com uv)
uv sync

# 2. Abrir o notebook
uv run jupyter notebook APS3.ipynb
```

---

## Estrutura do projeto

```
APS3.ipynb               # Notebook principal com todo o pipeline
Projeto_Enunciado.ipynb  # Enunciado original da APS
pyproject.toml           # Dependências do projeto
```

---

## O que o notebook faz (visão geral)

O notebook está dividido em **6 sprints**, cada um responsável por uma etapa do pipeline de Machine Learning.

---

### Sprint 1 — Setup, Dados e Pré-processamento

O ponto de partida de qualquer projeto de ML é entender os dados que você tem.

**O que foi feito:**
- Carregamento do dataset diretamente do repositório UCI usando `ucimlrepo`
- Inspeção inicial: quantas linhas, quais colunas, quais tipos de dados
- Tratamento de valores ausentes: no Adult Dataset, valores faltantes aparecem como `'?'` — eles são convertidos para `NaN` e depois preenchidos (imputação)
- Análise do **desbalanceamento de classes**: apenas ~24% das pessoas ganham >50K. Isso é um problema porque um modelo ingênuo aprende a dizer "<=50K" para todo mundo e ainda acerta 76% das vezes — mas é inútil na prática

**Conceitos:**
- **Imputação:** preencher valores ausentes com a mediana (para números) ou a moda (para categorias), em vez de simplesmente apagar as linhas
- **Desbalanceamento de classes:** quando uma categoria aparece muito mais do que a outra. A solução adotada foi usar `class_weight='balanced'`, que faz o modelo penalizar mais o erro na classe rara
- **Pipeline de pré-processamento:** um objeto que encadeia os passos de transformação (imputação → normalização → encoding) de forma organizada, evitando vazamento de informação entre treino e teste (data leakage)
- **ColumnTransformer:** aplica transformações diferentes para colunas numéricas e categóricas ao mesmo tempo
- **StandardScaler:** normaliza os números para terem média 0 e desvio padrão 1 — necessário para modelos como regressão logística que são sensíveis à escala
- **OneHotEncoder:** converte categorias textuais (ex: "Male", "Female") em colunas binárias de 0 e 1 para que o modelo consiga processar
- **Split estratificado:** divide os dados em treino e teste mantendo a mesma proporção de classes nos dois conjuntos

---

### Sprint 2 — Feature Engineering

Features (variáveis) bem construídas melhoram muito o desempenho dos modelos. Aqui foram criadas 4 novas features a partir das originais.

**Features criadas:**

| Feature | Fórmula | Por que faz sentido |
|---|---|---|
| `capital_net` | `capital-gain` − `capital-loss` | O saldo líquido de capital é mais informativo do que ganho e perda separados |
| `work_experience_proxy` | `age` − `education-num` − 6 | Estima quantos anos de experiência profissional a pessoa tem |
| `hours_education_ratio` | `hours-per-week` ÷ (`education-num` + 1) | Relaciona esforço semanal com nível educacional |
| `age_group` | Faixa etária (<=25, 26-35, ...) | Captura efeitos não-lineares da idade de forma discreta |

**Conceitos:**
- **Feature Engineering:** criação manual de novas variáveis que o modelo sozinho não conseguiria descobrir facilmente
- **Análise de separabilidade:** verificar visualmente se uma feature consegue distinguir as duas classes — se as distribuições de >50K e <=50K forem muito sobrepostas, a feature tem pouco poder preditivo
- **Binning:** agrupar valores contínuos em categorias (ex: idades em faixas) para capturar relações não-lineares

---

### Sprint 3 — Modelagem e Validação Cruzada

Três modelos foram treinados e comparados, do mais simples ao mais complexo.

**Modelos:**

| Modelo | Tipo | Ideia central |
|---|---|---|
| Regressão Logística | Baseline linear | Aprende uma fronteira linear entre as classes usando probabilidades |
| Random Forest | Ensemble de árvores | Combina centenas de árvores de decisão treinadas em subamostras aleatórias |
| XGBoost | Gradient Boosting | Constrói árvores sequencialmente, cada uma corrigindo os erros da anterior |

**Métricas reportadas (todas obrigatórias):**

| Métrica | O que mede |
|---|---|
| **Accuracy** | Proporção de acertos no total — enganosa com classes desbalanceadas |
| **Precision** | Dos que o modelo disse ">50K", quantos realmente são? |
| **Recall** | Dos que realmente são ">50K", quantos o modelo encontrou? |
| **F1-score** | Média harmônica entre Precision e Recall — melhor métrica com desbalanceamento |
| **ROC-AUC** | Área sob a curva ROC — mede a capacidade geral de separar as classes (0.5 = aleatório, 1.0 = perfeito) |
| **PR-AUC** | Área sob a curva Precision-Recall — mais informativa do que ROC-AUC quando há desbalanceamento forte |
| **Confusion Matrix** | Tabela que mostra Verdadeiros Positivos, Falsos Positivos, Verdadeiros Negativos e Falsos Negativos |

**Conceitos:**
- **Validação cruzada (RepeatedStratifiedKFold):** divide os dados em 5 partes, treina em 4 e testa em 1, repetindo 3 vezes com embaralhamentos diferentes. Resulta em 15 medições independentes de desempenho, o que dá uma estimativa muito mais confiável do que um único split treino/teste
- **Falso Positivo (FP):** o modelo previu ">50K" mas a pessoa ganha menos — pode levar a benefícios indevidos
- **Falso Negativo (FN):** o modelo previu "<=50K" mas a pessoa ganha mais — pode levar a exclusão injusta

---

### Sprint 4 — Otimização e Seleção de Modelo

Cada modelo tem hiperparâmetros (configurações que não são aprendidas dos dados) que precisam ser ajustados.

**O que foi feito:**
- `RandomizedSearchCV` para Random Forest e XGBoost: testa combinações aleatórias de hiperparâmetros e escolhe a melhor usando validação cruzada
- Teste de Wilcoxon entre os modelos: verifica estatisticamente se a diferença de desempenho entre eles é real ou apenas variação aleatória
- Avaliação final no **holdout de teste** (dados que nenhum modelo viu durante o treino e tuning)

**Conceitos:**
- **Hiperparâmetros:** configurações do modelo definidas antes do treinamento, como profundidade máxima das árvores, número de árvores, taxa de aprendizado — em oposição aos parâmetros que o modelo aprende sozinho dos dados
- **RandomizedSearchCV:** em vez de testar todas as combinações possíveis (GridSearch), testa um número fixo de combinações aleatórias — muito mais rápido com resultado praticamente igual
- **Teste de Wilcoxon:** teste estatístico não-paramétrico que verifica se dois conjuntos de medições vêm de distribuições diferentes. Um p-value < 0.05 indica que a diferença entre dois modelos é estatisticamente significativa
- **Holdout de teste:** conjunto de dados separado no início e usado apenas uma vez, no final, para dar uma estimativa honesta do desempenho real do modelo

---

### Sprint 5 — Interpretabilidade e Análise de Fairness

Saber que o modelo acerta 90% das vezes não é suficiente — precisamos entender **por que** ele toma cada decisão.

**O que foi feito:**
- **SHAP Summary Plot:** mostra quais features têm mais impacto nas previsões e em qual direção (valores altos de `capital_net` empurram para ">50K", por exemplo)
- **SHAP Bar Plot:** importância global de cada feature (média dos impactos absolutos)
- **SHAP Waterfall:** explica a previsão para um indivíduo específico — mostra quanto cada feature contribuiu para empurrar a previsão para cima ou para baixo
- **Análise de Fairness:** compara Precision, Recall e F1 separadamente para homens/mulheres e para diferentes grupos raciais

**Conceitos:**
- **SHAP (SHapley Additive exPlanations):** método matemático baseado em teoria dos jogos que atribui a cada feature uma "fatia" da previsão do modelo. É considerado o padrão ouro de interpretabilidade porque é consistente e localmente fiel
- **Interpretabilidade local vs. global:** local = explica uma previsão específica; global = explica o comportamento médio do modelo
- **Fairness / Equidade algorítmica:** verificar se o modelo tem desempenho semelhante para diferentes grupos demográficos. Um modelo com alto F1 global pode ainda assim ser muito pior para grupos minoritários — o que é um problema ético grave em aplicações reais

---

### Sprint 6 — Discussão Crítica e Conclusões

A última etapa consolida os resultados e discute os limites do que foi feito.

**Pontos discutidos:**
- O impacto do desbalanceamento de classes e como foi mitigado
- A diferença entre correlação e causalidade: o modelo aprende que `marital-status` está correlacionado com renda alta, mas isso não significa que casar aumenta a renda — reflete desigualdades estruturais históricas
- As limitações do dataset (dados de 1994, viés histórico, sub-representação de minorias)
- Sugestões de melhoria: ajuste de threshold, SMOTE, fairness-aware learning com `fairlearn`

---

## Dependências

Gerenciadas com `uv`. As principais bibliotecas utilizadas:

| Biblioteca | Para quê |
|---|---|
| `pandas` / `numpy` | Manipulação e operações numéricas |
| `scikit-learn` | Modelos, pipelines, métricas, validação cruzada |
| `xgboost` | Gradient Boosting |
| `imbalanced-learn` | SMOTE e técnicas de balanceamento |
| `shap` | Interpretabilidade dos modelos |
| `matplotlib` / `seaborn` | Visualizações |
| `ucimlrepo` | Download automático do dataset UCI |
| `scipy` | Teste estatístico de Wilcoxon |
