# Plano do Trabalho – Comparação de Modelos de Previsão T(t+1)

> Trabalho da disciplina **Data Science no Mercado Financeiro**
> Bases: `dados/WINN_M1/WIN$N_M1.csv` (mini índice) e `dados/WDON_M1/WDO$N_M1.csv` (mini dólar)
> Documento de referência teórica: [`MODELO_Tt+1.md`](MODELO_Tt+1.md) · EDA: `eda/analise_exploratoria_WIN_WDO.ipynb`

**O que mudou:** o trabalho deixa de ser "aplicar um SARIMAX" e passa a ser **comparar, sob o mesmo protocolo, várias famílias de modelos** de previsão um passo à frente. O SARIMAX vira um dos competidores. O foco do trabalho passa a ser a **comparação justa**: mesmos dados, mesmo alvo, mesmas janelas de teste, mesmas métricas e testes estatísticos.

---

## Sumário

1. [Objetivo e perguntas de pesquisa](#1-objetivo-e-perguntas-de-pesquisa)
2. [Protocolo comum (decisões fixas)](#2-protocolo-comum-decisões-fixas)
3. [Modelos que serão comparados](#3-modelos-que-serão-comparados)
4. [Conjuntos de features (ablação)](#4-conjuntos-de-features-ablação)
5. [Avaliação: métricas, testes e backtest](#5-avaliação-métricas-testes-e-backtest)
6. [Fluxo do trabalho](#6-fluxo-do-trabalho)
7. [Passo a passo](#7-passo-a-passo)
8. [Código base do comparador](#8-código-base-do-comparador)
9. [Armadilhas específicas da comparação](#9-armadilhas-específicas-da-comparação)
10. [Prioridades e divisão do trabalho](#10-prioridades-e-divisão-do-trabalho)
11. [Estrutura do relatório final](#11-estrutura-do-relatório-final)

---

## 1. Objetivo e perguntas de pesquisa

**Objetivo:** comparar modelos estatísticos clássicos, de machine learning e (opcionalmente) redes neurais na previsão do **retorno do próximo candle** (T(t+1)) do WIN e do WDO, verificando se algum supera o modelo ingênuo **fora da amostra**, com significância estatística e após custos.

| # | Pergunta | Como é respondida |
|---|---|---|
| P1 | Algum modelo supera o naive fora da amostra? | Theil U < 1, R²_oos > 0, teste de Clark-West / Diebold-Mariano |
| P2 | Modelos não lineares (árvores, redes) superam os lineares? | Comparação entre famílias com o mesmo conjunto de features |
| P3 | O outro ativo (WDO ↔ WIN) acrescenta informação? | Ablação: mesmo modelo com e sem features cruzadas |
| P4 | O ganho estatístico vira dinheiro? | Backtest com custos |
| P5 | O resultado depende do timeframe? | Repetir a comparação em 15 e 60 min (e 5 min, se der tempo) |
| P6 *(trilha B, recomendada)* | A **volatilidade** do próximo candle é mais previsível que a direção? | Mesma comparação com alvo = \|r\| do próximo candle |

**Hipóteses esperadas** (com base na EDA e no teste preliminar da Seção 3.7 do `MODELO_Tt+1.md`):
- H1: no retorno, nenhum modelo bate o naive de forma significativa (mercado eficiente na forma fraca).
- H2: a correlação WIN × WDO é contemporânea (−0,50) e não preditiva (≈ 0,01 com defasagem), então features cruzadas não ajudam.
- H3: na volatilidade, modelos simples (HAR, GARCH) batem o naive com folga, porque \|r\| tem autocorrelação forte e padrão intradiário.

> Confirmar ou refutar essas hipóteses com rigor **é** o resultado do trabalho. Um "nenhum modelo bate o naive", bem demonstrado, é uma conclusão válida.

---

## 2. Protocolo comum (decisões fixas)

Tudo o que está nesta seção é **igual para todos os modelos**. É isso que torna a comparação justa.

### 2.1 Alvo

| Item | Decisão |
|---|---|
| Alvo principal | Log-retorno do próximo candle, em %: $y_t = r_{t+1} = 100 \cdot \ln(P_{t+1}/P_t)$ |
| Alvo de direção | Derivado do principal: $\text{sinal}(r_{t+1})$. Candles com $r_{t+1} = 0$ ficam fora das métricas de direção |
| Indexação | A linha *t* contém as features conhecidas em *t* e o alvo $r_{t+1}$ |
| Fronteira entre pregões | O último candle de cada pregão **não tem alvo** (o próximo retorno incluiria o gap noturno) |
| 1º candle do pregão | Retorno = abertura → fechamento do próprio candle (sem o gap noturno) |
| Alvo da trilha B | $\lvert r_{t+1} \rvert$ (ou $r_{t+1}^2$) |

### 2.2 Ativos e timeframes

| Item | Decisão |
|---|---|
| Ativo principal | **WIN** (exógena: WDO) |
| Replicação | **WDO** (exógena: WIN), com o mesmo pipeline |
| Timeframe principal | **15 min** (s ≈ 38 candles por pregão) |
| Robustez | **60 min** (s ≈ 10); 5 min opcional, sem SARIMAX |
| Período comum | Interseção das bases: 16/10/2020 → 01/04/2026 (o WDO termina em 01/04/2026) |

### 2.3 Divisão temporal

```
 2020-10 ────────────────────────── 2023 (buraco jun–out) ──────── 2024-01 ─────── 2025-01 ─────────── 2026-04
 |<──────────────────── TREINO INICIAL ────────────────────────>|<─ VALIDAÇÃO ─>|<──── TESTE (holdout) ───>|
                                                                  ajuste de        rodado UMA vez, no final,
                                                                  hiperparâmetros  com tudo congelado
```

| Bloco | Período | Uso |
|---|---|---|
| Treino inicial | 16/10/2020 → 31/12/2023 | Ajuste dos modelos |
| Validação | 01/01/2024 → 31/12/2024 | Escolha de hiperparâmetros, ordens (p, q), limiar do backtest. Walk-forward trimestral |
| **Teste** | 01/01/2025 → 31/03/2026 | Comparação final. **Só é rodado depois que todas as escolhas estiverem congeladas** |

### 2.4 Esquema walk-forward (janela expansiva)

```
Trimestre 1 do teste: [=========== treino até 31/12/2024 ===========][T1]
Trimestre 2 do teste: [============= treino até 31/03/2025 =============][T2]
Trimestre 3 do teste: [=============== treino até 30/06/2025 ===============][T3]
...                                        → re-treino a cada trimestre
```

- Re-treino **trimestral** para todos os modelos (mensal se o tempo de execução permitir; o mesmo para todos).
- Dentro de cada trimestre a previsão é T(t+1) de verdade: para prever *t+1* o modelo usa apenas dados até *t*, sem re-estimar os parâmetros.
- **Embargo** de 1 candle entre treino e teste (o alvo da última linha do treino é o primeiro retorno do teste).
- Todos os modelos preveem **exatamente os mesmos timestamps**. Isso é pré-requisito para os testes pareados (DM, Clark-West).

### 2.5 Formato padrão de saída

Cada rodada (modelo × feature set × ativo × timeframe) salva **um arquivo de previsões** no mesmo formato:

```
resultados/previsoes/{ativo}_{timeframe}_{modelo}_{features}.csv

datetime, y_true, y_pred, p_alta (opcional, só classificadores)
```

A avaliação (Seção 5) lê só esses arquivos. Assim cada pessoa do grupo pode trabalhar num modelo diferente, e a comparação continua consistente.

---

## 3. Modelos que serão comparados

### 3.1 Catálogo

| Código | Modelo | Família | Biblioteca | Hiperparâmetros a ajustar na validação | Prioridade |
|---|---|---|---|---|---|
| **B0** | Naive (retorno previsto = 0 ⇔ $\hat P_{t+1} = P_t$) | Baseline | – | – | **Essencial** |
| B1 | Média histórica do retorno | Baseline | – | – | Essencial |
| B2 | Média por horário (retorno médio daquele candle do dia no treino) | Baseline sazonal | pandas | – | Recomendado |
| **E1** | AR(p) | Estatístico linear | `statsmodels` / OLS | p ∈ {1, …, 10} por BIC | **Essencial** |
| **E2** | ARIMA(p, 0, q) | Estatístico linear | `statsmodels` SARIMAX | p, q ∈ {0, 1, 2} por AIC/BIC | **Essencial** |
| **E3** | ARIMAX | Estatístico linear + exógenas | `statsmodels` SARIMAX | ordem de E2 + escolha das exógenas | **Essencial** |
| **E4** | SARIMAX | E3 + sazonalidade do pregão | `statsmodels` SARIMAX | P ∈ {0, 1}, s = candles/pregão | **Essencial** |
| E5 | ARMA-GARCH(1,1) | Estatístico (média + volatilidade) | `arch` | ordem da média, distribuição (normal × t) | Recomendado |
| **M1** | Regressão Ridge / Lasso | ML linear | `scikit-learn` | α | **Essencial** |
| M1c | Regressão logística (direção) | ML linear (classificação) | `scikit-learn` | C | Recomendado |
| M2 | Random Forest | Árvores (bagging) | `scikit-learn` | nº de árvores, profundidade, min_samples_leaf | Recomendado |
| **M3** | LightGBM (ou XGBoost) | Árvores (boosting) | `lightgbm` / `xgboost` | learning_rate, num_leaves, n_estimators, regularização | **Essencial** |
| M4 | MLP (rede densa) | Rede neural | `scikit-learn` | camadas, α, early stopping | Opcional |
| M5 | LSTM / GRU | Rede neural recorrente | `tensorflow` / `pytorch` | tamanho da janela, unidades, dropout | Opcional |
| C1 | Combinação (média simples das previsões de E2, M1 e M3) | Ensemble | – | – | Recomendado |

> **Por que a combinação (C1)?** Na literatura de previsão, a média simples de vários modelos costuma ser mais estável que qualquer modelo individual (Timmermann, 2006). É barata: usa os arquivos de previsão que já existem.

### 3.2 Trilha B – previsão da volatilidade (recomendada)

Mesmo protocolo, alvo $\lvert r_{t+1} \rvert$:

| Código | Modelo |
|---|---|
| V0 | Naive de volatilidade: $\lvert \hat r_{t+1} \rvert = \lvert r_t \rvert$ |
| V1 | Média móvel de \|r\| (janela de 1 pregão) |
| V2 | Média de \|r\| por horário do dia |
| V3 | HAR (regressão em \|r\| médio do último candle, último pregão e últimos 5 pregões) |
| V4 | GARCH(1,1) |
| V5 | LightGBM com as features do conjunto F3 |

É aqui que a comparação provavelmente mostra **diferenças reais** entre modelos, o que enriquece a discussão.

---

## 4. Conjuntos de features (ablação)

Os conjuntos são **aninhados**: cada um contém o anterior. Rodar o mesmo modelo em F0 → F3 isola a contribuição de cada grupo.

| Conjunto | Conteúdo (tudo conhecido em *t*) | Pergunta que responde |
|---|---|---|
| **F0** | Lags do próprio retorno: $r_t, \dots, r_{t-9}$ | O passado do ativo ajuda? |
| **F1** | F0 + estrutura do candle (amplitude, corpo), volume relativo, volatilidade recente, gap de abertura do dia | Informação de volume/volatilidade ajuda? |
| **F2** | F1 + lags do retorno e do volume relativo do **outro ativo** | WDO ↔ WIN acrescenta algo? (P3) |
| **F3** | F2 + calendário (candle do dia, hora, dia da semana, dia de vencimento) | O padrão intradiário/semanal ajuda? |

Como rodar:
- **Modelos de ML (M1–M5):** em F0 e F3 obrigatoriamente; no melhor modelo, em F0, F1, F2 e F3 (ablação completa).
- **Modelos estatísticos:** E1–E2 usam só o próprio retorno (equivale a F0). E3–E4 usam exógenas reduzidas (2–4 variáveis de F2), para manter o ajuste rápido e os coeficientes interpretáveis.

Regras:
- Nada calculado com dados de *t+1* ou depois (conferir cada feature).
- Normalização (`StandardScaler`) **dentro** do pipeline, ajustada só na janela de treino de cada bloco.
- Coluna SPREAD **não entra** (muda de escala no WDO em 2023, ver EDA).
- Janelas móveis com `min_periods`, porque há buracos e fronteiras de pregão (sem isso, a coluna fica inteira NaN).

---

## 5. Avaliação: métricas, testes e backtest

### 5.1 Métricas estatísticas (todas contra o naive B0)

| Tipo | Métrica | Leitura |
|---|---|---|
| Regressão | RMSE, MAE | Erro absoluto |
| Regressão | **Theil U** = RMSE_modelo / RMSE_naive | < 1 → melhor que o naive |
| Regressão | **R²_oos** = 1 − Σe²_modelo / Σe²_naive (Campbell & Thompson, 2008) | > 0 → melhor que o naive |
| Direção | Acurácia direcional (sem os candles com r = 0) | Referência: 50% e a proporção de altas do teste |
| Direção | Acurácia balanceada, MCC, AUC (classificadores) | Robustas a desbalanceamento |
| Volatilidade (trilha B) | RMSE, MAE, QLIKE | QLIKE é a função de perda padrão para volatilidade |

### 5.2 Testes de significância

| Teste | Quando usar | Observação |
|---|---|---|
| **Clark-West** | Modelo vs. naive (modelos aninhados: o naive é o modelo com todos os coeficientes = 0) | Teste correto para "bate o passeio aleatório?" |
| **Diebold-Mariano** (erro-padrão HAC/Newey-West) | Pares de modelos não aninhados (ex.: ARIMA × LightGBM) | Com correção de Holm para múltiplas comparações |
| **Pesaran-Timmermann** | Acurácia direcional | A acurácia é maior que a do acaso? |
| *Model Confidence Set* (opcional) | Conjunto dos "melhores" modelos | `arch.bootstrap.MCS` |

### 5.3 Backtest (P4)

```
sinal(t)   = +1 se ŷ(t) > +limiar ; −1 se ŷ(t) < −limiar ; 0 caso contrário
resultado  = Σ sinal(t) · r(t+1) − custo · |Δ sinal|
```

- O **limiar** é escolhido na validação (2024), nunca no teste.
- Custo por operação parametrizado em pontos (ex.: WIN 0 / 5 / 10 pontos; WDO 0 / 0,5 / 1 ponto) → análise de sensibilidade. Confirmar com o professor valores realistas de corretagem, emolumentos e slippage.
- Métricas: retorno acumulado, Sharpe anualizado, drawdown máximo, nº de operações, taxa de acerto. Comparar com "comprar e segurar" e com sinais aleatórios.
- Posições zeradas no fim do pregão (sem carregar gap noturno).

### 5.4 Tabela final (modelo do resultado principal)

| Modelo | Features | RMSE | Theil U | R²_oos | CW p-valor | Acurácia | MCC | Sharpe (c/ custo) | Tempo de treino |
|---|---|---|---|---|---|---|---|---|---|
| B0 Naive | – | … | 1,000 | 0 | – | – | – | – | – |
| E2 ARIMA | F0 | … | … | … | … | … | … | … | … |
| … | … | … | … | … | … | … | … | … | … |

Uma tabela por ativo × timeframe, mais um gráfico de barras de Theil U com intervalos de confiança e o gráfico do **erro quadrático acumulado relativo ao naive** ao longo do teste (mostra *quando* um modelo ganha ou perde).

---

## 6. Fluxo do trabalho

```
┌────────────────────┐
│ CSVs M1 (WIN, WDO) │
└─────────┬──────────┘
          ▼
┌────────────────────────────┐   EDA já feita (analise_exploratoria_WIN_WDO.ipynb):
│ 01 Preparação da base      │   buracos 2023, rolagem, SPREAD, fronteiras de pregão
│ reamostrar 15/60 min,      │
│ alvo r(t+1), features F0–F3│──► dados/base_{ativo}_{tf}.parquet  (ou .csv)
└─────────┬──────────────────┘
          ▼
┌────────────────────────────┐
│ tt1.py (módulo comum)      │   carregar · montar_base · walk_forward · metricas
│ mesmas janelas p/ todos    │   (Seção 8)
└─────────┬──────────────────┘
          ▼
┌──────────────┬───────────────┬─────────────────┬──────────────┐
│ 02 Baselines │ 03 Estatíst.  │ 04 Machine      │ 05 Redes     │   ajuste de hiperparâmetros
│ B0 B1 B2     │ E1–E5         │ Learning M1–M3  │ M4 M5 (opc.) │   SÓ na validação (2024)
└──────┬───────┴───────┬───────┴────────┬────────┴──────┬───────┘
       └───────────────┴───────┬────────┴───────────────┘
                               ▼
              resultados/previsoes/*.csv  (formato único)
                               ▼
┌────────────────────────────────────────────────┐
│ 06 Avaliação e comparação                      │
│ métricas · Clark-West / DM · ablação F0–F3 ·   │
│ combinação C1 · gráficos                       │
└─────────┬──────────────────────────────────────┘
          ▼
┌────────────────────┐     ┌──────────────────────────┐
│ 07 Backtest c/ custo│ ──► │ 08 Relatório e conclusões│
└────────────────────┘     └──────────────────────────┘
```

**Regra do congelamento:** as caixas 02–05 rodam primeiro com teste = 2024 (validação). Só depois de fixar todas as escolhas (num arquivo `config_final.json` ou numa célula de configuração) é que o teste 2025–2026 é rodado, uma única vez.

---

## 7. Passo a passo

Cada etapa tem um **entregável** e um **critério de pronto**.

### Etapa 0 – Alinhamento
- [ ] Confirmar com o professor: lista mínima de modelos, alvo (retorno × direção), timeframe e se o backtest é exigido
- [ ] Confirmar custos operacionais realistas para o backtest
- [ ] Dividir os modelos entre os membros do grupo (Seção 10)

**Entregável:** este plano aprovado. **Pronto quando:** alvo, timeframes, divisão temporal e lista de modelos estiverem fechados.

### Etapa 1 – Ambiente e estrutura de pastas
- [ ] Criar a estrutura:

```
data_science/
├── WINN_M1/, WDON_M1/                 (dados originais, não alterar)
├── analise_exploratoria_WIN_WDO.ipynb (já existe)
├── tt1.py                             (módulo comum – Seção 8)
├── 01_preparacao_base.ipynb
├── 02_baselines.ipynb
├── 03_modelos_estatisticos.ipynb
├── 04_machine_learning.ipynb
├── 05_redes_neurais.ipynb             (opcional)
├── 06_avaliacao_comparacao.ipynb
├── 07_backtest.ipynb
├── dados/                             (bases reamostradas com features)
└── resultados/
    ├── previsoes/                     (um CSV por rodada)
    ├── tabelas/
    └── figuras/
```

- [ ] Instalar bibliotecas (no Colab, só as que faltarem):

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn lightgbm xgboost arch
# opcional: pip install tensorflow   (ou torch) para LSTM/GRU
```

- [ ] No Colab: copiar `tt1.py` para a pasta do Drive e fazer `sys.path.append(PASTA)`
- [ ] Fixar sementes (`random_state=42`) em todos os modelos estocásticos

**Pronto quando:** `import tt1` funciona no notebook.

### Etapa 2 – Preparação da base (`01_preparacao_base.ipynb`)
- [ ] Carregar WIN e WDO, remover SPREAD, reamostrar para 15 e 60 min
- [ ] Juntar os dois ativos pelo `datetime` (join interno)
- [ ] Calcular retornos só dentro do pregão; 1º candle = abertura → fechamento
- [ ] Criar o alvo $r_{t+1}$ e anular o alvo do último candle de cada pregão
- [ ] Criar as features F0–F3 (Seção 4) com `min_periods` nas janelas móveis
- [ ] Marcar dias de vencimento (WIN: quarta mais próxima do dia 15 dos meses pares; WDO: 1º dia útil do mês) como feature e para checagem
- [ ] Salvar `dados/base_{ativo}_{tf}`
- [ ] **Checagem anti-vazamento:** para 3 timestamps aleatórios, conferir à mão que nenhuma feature usa preço posterior a *t* e que `alvo(t) = ret(t+1)`

**Entregável:** 4 bases (WIN/WDO × 15/60 min). **Pronto quando:** sem NaN, contagem de linhas por ano coerente (2023 menor por causa do buraco) e checagem anti-vazamento ok.

### Etapa 3 – Baselines (`02_baselines.ipynb`)
- [ ] Rodar B0, B1, B2 no walk-forward da validação (2024)
- [ ] Salvar previsões no formato padrão

**Pronto quando:** a tabela de validação tem B0–B2. O RMSE do B0 é a régua de todas as outras tabelas.

### Etapa 4 – Modelos estatísticos (`03_modelos_estatisticos.ipynb`)
- [ ] ACF/PACF dos retornos no treino → faixa de ordens candidatas
- [ ] E1 AR(p): escolher p por BIC no treino
- [ ] E2 ARIMA(p,0,q): grade pequena p, q ∈ {0, 1, 2}, escolha por BIC; checar resíduos (Ljung-Box)
- [ ] E3 ARIMAX: E2 + exógenas defasadas (retorno do outro ativo em *t*, volume relativo em *t*); reportar coeficientes e p-valores
- [ ] E4 SARIMAX: E3 + (1,0,0,s) com s = candles/pregão (só 15 e 60 min)
- [ ] E5 ARMA-GARCH(1,1) com distribuição t (`arch`)
- [ ] Rodar todos no walk-forward da validação e salvar previsões

**Pronto quando:** ordens escolhidas e justificadas, previsões da validação salvas, tabela de coeficientes do E3/E4 pronta para o relatório.

### Etapa 5 – Machine learning (`04_machine_learning.ipynb`)
- [ ] M1 Ridge/Lasso: α por grade na validação (`Pipeline(StandardScaler, Ridge)`)
- [ ] M1c Regressão logística na direção
- [ ] M2 Random Forest: grade pequena (≤ 12 combinações)
- [ ] M3 LightGBM: grade pequena ou busca aleatória limitada (≤ 30 combinações); `n_estimators` com early stopping em um trecho final do treino
- [ ] Cada modelo em F0 e F3; o melhor em F0, F1, F2, F3
- [ ] Importância das features (gain do LightGBM + permutation importance na validação)
- [ ] Registrar **todas** as configurações testadas e o resultado de cada uma

**Pronto quando:** melhor configuração de cada modelo escolhida **só pela validação** e registrada.

### Etapa 6 – Redes neurais (`05_redes_neurais.ipynb`, opcional)
- [ ] M4 MLP com early stopping
- [ ] M5 LSTM/GRU: sequências de 10–40 candles que **não atravessam** fronteiras de pregão nem os buracos da base
- [ ] Mesmo walk-forward; re-treino trimestral (custo alto: considerar re-treino semestral **para todos** se necessário, e documentar)

**Pronto quando:** previsões no formato padrão. Se não superar M3 na validação, basta reportar.

### Etapa 7 – Congelamento e rodada no teste
- [ ] Registrar em `config_final.json`: ordens, hiperparâmetros, feature sets, limiar do backtest
- [ ] Rodar **todos** os modelos no walk-forward do teste (01/01/2025 → 31/03/2026), WIN e WDO, 15 e 60 min
- [ ] Gerar C1 (média das previsões de E2, M1 e M3)
- [ ] (Trilha B) rodar V0–V5

**Pronto quando:** existe um CSV de previsões por rodada, todos com os mesmos timestamps (conferir com `assert`).

### Etapa 8 – Avaliação e comparação (`06_avaliacao_comparacao.ipynb`)
- [ ] Tabela final (Seção 5.4) por ativo × timeframe
- [ ] Clark-West de cada modelo vs. naive; DM entre os melhores pares, com correção de Holm
- [ ] Pesaran-Timmermann para a acurácia direcional
- [ ] Ablação F0 → F3 do melhor modelo (responde P3)
- [ ] Gráficos: Theil U por modelo; erro quadrático acumulado relativo ao naive; importância das features
- [ ] Desempenho por subperíodo (trimestres do teste) e por horário do dia (abertura × meio × fechamento)

**Pronto quando:** P1, P2, P3 e P5 respondidas com número e teste.

### Etapa 9 – Backtest (`07_backtest.ipynb`)
- [ ] Estratégia da Seção 5.3 para os 2–3 melhores modelos e para o naive/aleatório
- [ ] Curvas de capital sem custo e com custos (3 níveis)
- [ ] Tabela: retorno, Sharpe, drawdown, nº de operações, taxa de acerto

**Pronto quando:** P4 respondida.

### Etapa 10 – Relatório e apresentação
- [ ] Seguir a estrutura da Seção 11
- [ ] Discutir os resultados à luz da eficiência de mercado
- [ ] Listar limitações (buraco de 2023, série contínua com rolagem, poucos regimes no teste, custos aproximados)

---

## 8. Código base do comparador

Esqueleto do `tt1.py`. **Testado com as duas bases** (60 e 15 min, teste 2025-01 → 2026-03, re-treino trimestral): `montar_base`, `walk_forward`, `metricas`, `Naive`, `MediaHistorica` e `SarimaxT1`. O `ComoClassificador` e os modelos do scikit-learn/LightGBM ainda não foram rodados (bibliotecas ausentes na máquina local; no Colab já vêm instaladas). Todo modelo entra no comparador desde que tenha `fit(X, y)` e `predict(X)`; qualquer estimador do scikit-learn, LightGBM ou XGBoost já serve.

```python
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def carregar(caminho):
    df = pd.read_csv(caminho, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    return df.set_index("datetime").drop(columns=["date", "time", "spread"])

def reamostrar(df, regra):
    return (df.resample(regra)
              .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                    "tickvol": "sum", "vol": "sum"})
              .dropna())

def montar_base(win, wdo, regra):
    """Uma linha por candle t: features conhecidas em t + alvo = retorno de t+1 (em %)."""
    win_r, wdo_r = reamostrar(win, regra), reamostrar(wdo, regra)
    df = win_r.join(wdo_r[["close", "vol"]].add_suffix("_wdo"), how="inner")
    dia = pd.Series(df.index.date, index=df.index)
    novo_dia = dia != dia.shift()
    df["ret"] = np.log(df["close"]).diff() * 100
    df["ret_wdo"] = np.log(df["close_wdo"]).diff() * 100
    # 1º candle do dia: retorno intracandle (abertura -> fechamento), sem o gap noturno
    df.loc[novo_dia, "ret"] = np.log(df["close"] / df["open"])[novo_dia] * 100
    df.loc[novo_dia, "ret_wdo"] = 0.0
    df["alvo"] = df["ret"].shift(-1)
    df.loc[novo_dia.shift(-1, fill_value=True), "alvo"] = np.nan   # último candle do dia: sem alvo
    s = int(dia.groupby(dia).size().median())                       # candles por pregão
    for k in range(1, 6):
        df[f"ret_lag{k}"] = df["ret"].shift(k - 1)
        df[f"ret_wdo_lag{k}"] = df["ret_wdo"].shift(k - 1)
    df["amplitude"] = (df["high"] - df["low"]) / df["close"] * 100
    df["corpo"] = (df["close"] - df["open"]) / df["close"] * 100
    df["vol_rel"] = df["vol"] / df["vol"].rolling(s).mean()
    df["volat"] = df["ret"].rolling(s, min_periods=s // 2).std()
    df["hora"] = df.index.hour + df.index.minute / 60
    df["candle_do_dia"] = dia.groupby(dia).cumcount()
    return df.dropna(), s

# ---------- modelos com a interface fit / predict ----------
class Naive:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))

class MediaHistorica:
    def fit(self, X, y): self.m = y.mean(); return self
    def predict(self, X): return np.full(len(X), self.m)

class SarimaxT1:
    """ARIMA/ARIMAX/SARIMAX: estima no treino e, no teste, só incorpora as novas observações
    (extend), prevendo sempre um passo à frente com dados até t."""
    usa_y_teste = True
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), usa_x=False):
        self.kw = dict(order=order, seasonal_order=seasonal_order); self.usa_x = usa_x
    def fit(self, X, y):
        self.res = SARIMAX(y.values, exog=X.values if self.usa_x else None, **self.kw).fit(disp=False)
        return self
    def predict(self, X, y):
        return self.res.extend(y.values, exog=X.values if self.usa_x else None).predict()

class ComoClassificador:
    """Treina um classificador na direção; devolve P(alta) − 0,5 (o sinal é a direção prevista)."""
    def __init__(self, modelo): self.modelo = modelo
    def fit(self, X, y):
        m = y != 0
        self.modelo.fit(X[m], (y[m] > 0).astype(int)); return self
    def predict(self, X): return self.modelo.predict_proba(X)[:, 1] - 0.5

# ---------- walk-forward comum ----------
def walk_forward(df, features, fabrica, inicio, fim, freq="QS", embargo=1):
    """Re-treina no início de cada bloco (QS = trimestral) com todo o passado (janela expansiva)."""
    cortes = list(pd.date_range(inicio, fim, freq=freq)) + [pd.Timestamp(fim)]
    blocos = []
    for ini, fim_bloco in zip(cortes[:-1], cortes[1:]):
        tr = df[df.index < ini].iloc[:-embargo]
        te = df[(df.index >= ini) & (df.index < fim_bloco)]
        if te.empty: continue
        m = fabrica().fit(tr[features], tr["alvo"])
        p = m.predict(te[features], te["alvo"]) if getattr(m, "usa_y_teste", False) else m.predict(te[features])
        blocos.append(pd.DataFrame({"y_true": te["alvo"], "y_pred": np.asarray(p)}, index=te.index))
    return pd.concat(blocos)

def metricas(prev):
    e = prev.y_true - prev.y_pred
    rmse, rmse_naive = np.sqrt((e ** 2).mean()), np.sqrt((prev.y_true ** 2).mean())
    nz = prev.y_true != 0
    return dict(n=len(prev), RMSE=rmse, MAE=e.abs().mean(), TheilU=rmse / rmse_naive,
                R2_oos=1 - (e ** 2).sum() / (prev.y_true ** 2).sum(),
                acc_dir=(np.sign(prev.y_pred[nz]) == np.sign(prev.y_true[nz])).mean())
```

**Uso nos notebooks:**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from lightgbm import LGBMRegressor
import tt1

win = tt1.carregar("dados/WINN_M1/WIN$N_M1.csv")
wdo = tt1.carregar("dados/WDON_M1/WDO$N_M1.csv")
df, s = tt1.montar_base(win, wdo, "15min")

F0 = [f"ret_lag{k}" for k in range(1, 6)]
F3 = F0 + ["amplitude", "corpo", "vol_rel", "volat", "hora", "candle_do_dia"] \
        + [f"ret_wdo_lag{k}" for k in range(1, 6)]

MODELOS = {
    "B0_naive":   (tt1.Naive, F0),
    "E2_arima":   (lambda: tt1.SarimaxT1(order=(1, 0, 1)), F0),
    "E3_arimax":  (lambda: tt1.SarimaxT1(order=(1, 0, 1), usa_x=True), ["ret_wdo_lag1", "vol_rel"]),
    "E4_sarimax": (lambda: tt1.SarimaxT1(order=(1, 0, 1), seasonal_order=(1, 0, 0, s), usa_x=True),
                   ["ret_wdo_lag1", "vol_rel"]),
    "M1_ridge":   (lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), F3),
    "M1c_logit":  (lambda: tt1.ComoClassificador(make_pipeline(StandardScaler(), LogisticRegression())), F3),
    "M3_lgbm":    (lambda: LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=15,
                                         min_child_samples=200, random_state=42, verbose=-1), F3),
}

# validação: inicio="2024-01-01", fim="2025-01-01" | teste: inicio="2025-01-01", fim="2026-04-01"
for nome, (fabrica, feats) in MODELOS.items():
    prev = tt1.walk_forward(df, feats, fabrica, inicio="2024-01-01", fim="2025-01-01")
    prev.to_csv(f"resultados/previsoes/WIN_15min_{nome}.csv")
    print(nome, {k: round(float(v), 4) for k, v in tt1.metricas(prev).items()})
```

> Os hiperparâmetros acima são pontos de partida, não as escolhas finais (essas saem da Etapa 5). Para o WDO como alvo, basta inverter os papéis na `montar_base`.

**Resultado do teste do esqueleto** (WIN, teste 01/2025 → 03/2026, re-treino trimestral, feature set reduzido; serve só para validar o código):

| Modelo | 60 min: Theil U | 60 min: acurácia | 15 min: Theil U | 15 min: acurácia | Tempo (15 min) |
|---|---|---|---|---|---|
| Naive | 1,0000 | – | 1,0000 | – | 0 s |
| Média histórica | 0,9999 | 50,6% | 1,0000 | 50,7% | 0 s |
| AR(5) por OLS | 1,0007 | 50,5% | 1,0000 | 49,8% | 0 s |
| OLS com F2 | 1,0037 | 49,4% | 1,0006 | 50,1% | 0 s |
| ARIMA(1,0,1) | 1,0003 | 48,8% | 1,0000 | 50,2% | 15 s |
| ARIMAX(1,0,1) | 1,0008 | 50,5% | 1,0001 | 50,2% | 41 s |

Coerente com o teste preliminar do `MODELO_Tt+1.md`: os modelos lineares ficam colados no naive. Os tempos mostram que o walk-forward trimestral é viável mesmo para o ARIMAX em 15 min.

---

## 9. Armadilhas específicas da comparação

| Armadilha | Como evitar |
|---|---|
| **Ajustar hiperparâmetros olhando o teste** | Toda escolha é feita em 2024; o teste roda uma vez, com `config_final.json` congelado |
| Modelos avaliados em timestamps diferentes | Mesma base, mesmas janelas; `assert` de índices iguais antes de comparar |
| **Comparação injusta de esforço** | Orçamento de ajuste parecido para cada família (ex.: ≤ 30 configurações); reportar quantas foram testadas |
| *Data snooping* (testar muitos modelos e escolher o melhor) | Correção de Holm nos testes; reportar **todos** os modelos, não só o vencedor; MCS opcional |
| Usar DM para modelo aninhado vs. naive | Usar Clark-West nesse caso (o DM é conservador demais para aninhados) |
| Scaler ajustado com todos os dados | `Pipeline` do scikit-learn dentro do walk-forward |
| LSTM com janelas atravessando pregões ou buracos | Gerar sequências por pregão; descartar as que cruzam a fronteira |
| Exógena contemporânea (WDO em t+1) | Só lags; a correlação contemporânea de −0,50 viraria um modelo "perfeito" e falso |
| Acurácia inflada pela classe "parado" | Excluir r = 0 das métricas de direção (15% dos minutos do WDO em 1 min) |
| Rolagem e buracos | Retornos só dentro do pregão; feature de dia de vencimento; checar resíduos nesses dias |
| Rolling sem `min_periods` | Com NaN na fronteira de cada pregão, a janela vira NaN inteira e o `dropna` apaga a base |
| Concluir "modelo X é melhor" por diferença minúscula | Exigir teste significativo **e** ganho econômico; caso contrário, "empate com o naive" |

---

## 10. Prioridades e divisão do trabalho

### 10.1 Escopo mínimo × completo

| Nível | Inclui |
|---|---|
| **Mínimo (obrigatório)** | WIN 15 min · B0, B1, E1, E2, E3, E4, M1, M3 · F0 e F3 · Theil U, R²_oos, acurácia, Clark-West · tabela comparativa |
| **Recomendado** | + WDO e 60 min · B2, E5, M1c, M2, C1 · ablação F0–F3 · DM com Holm · backtest com custos · trilha B (volatilidade) |
| **Opcional** | + 5 min · M4, M5 (redes) · MCS · análise por horário e por regime |

### 10.2 Sugestão de divisão (grupo de 3–4 pessoas)

| Frente | Etapas | Responsável |
|---|---|---|
| Dados e infraestrutura | 1, 2, `tt1.py`, baselines (3) | … |
| Modelos estatísticos | 4 (E1–E5) e trilha B (V3, V4) | … |
| Machine learning | 5 e 6 (M1–M5) e trilha B (V5) | … |
| Avaliação e backtest | 7, 8, 9 e consolidação do relatório | … |

A frente de dados entrega `tt1.py` e as bases **primeiro**; as outras frentes dependem dela. A partir daí, as frentes de modelos trabalham em paralelo e só trocam arquivos de previsão.

### 10.3 Ordem de execução

1. Etapas 0–2 (bloqueiam todo o resto)
2. Etapa 3 + versões mais simples de E2 e M1 rodando de ponta a ponta até a tabela (prova que o pipeline funciona)
3. Etapas 4, 5 e 6 em paralelo, só na validação
4. Etapa 7 (congelamento + teste)
5. Etapas 8 e 9
6. Etapa 10

---

## 11. Estrutura do relatório final

1. **Introdução** – previsão T(t+1), motivação, perguntas P1–P6
2. **Fundamentação teórica** – random walk e eficiência de mercado; modelos ARIMA/SARIMAX/GARCH; regressão regularizada; árvores e boosting; redes recorrentes; combinação de previsões
3. **Dados** – WIN e WDO, período, tratamento (rolagem, buracos, fronteiras de pregão, SPREAD); principais achados da EDA
4. **Metodologia** – protocolo comum (Seção 2), modelos (Seção 3), features (Seção 4), validação walk-forward, métricas e testes (Seção 5)
5. **Resultados**
   - 5.1 Comparação principal (WIN 15 min)
   - 5.2 Robustez: WDO e 60 min
   - 5.3 Ablação das features e contribuição do outro ativo
   - 5.4 Backtest com custos
   - 5.5 Trilha B: volatilidade
6. **Discussão** – algum modelo agrega valor? Linear × não linear; por que (ou por que não) o WDO ajuda; o ganho estatístico sobrevive aos custos?
7. **Conclusão, limitações e trabalhos futuros**

---

## Referências adicionais (além das do `MODELO_Tt+1.md`)

- Campbell, J. Y.; Thompson, S. B. (2008). *Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?* Review of Financial Studies.
- Clark, T. E.; West, K. D. (2007). *Approximately Normal Tests for Equal Predictive Accuracy in Nested Models*. Journal of Econometrics.
- Pesaran, M. H.; Timmermann, A. (1992). *A Simple Nonparametric Test of Predictive Performance*. Journal of Business & Economic Statistics.
- Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). *The Model Confidence Set*. Econometrica.
- Timmermann, A. (2006). *Forecast Combinations*. Handbook of Economic Forecasting.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility* (modelo HAR). Journal of Financial Econometrics.
- Gu, S.; Kelly, B.; Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies.
- Makridakis, S.; Spiliotis, E.; Assimakopoulos, V. (2018). *Statistical and Machine Learning forecasting methods: Concerns and ways forward*. PLoS ONE.
