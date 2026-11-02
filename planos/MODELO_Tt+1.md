# Modelo T(t+1) – Previsão de Séries Temporais no Mercado Financeiro

> Trabalho da disciplina **Data Science no Mercado Financeiro**
> Bases: `dados/WDON_M1/WDO$N_M1.csv` (mini dólar) e `dados/WINN_M1/WIN$N_M1.csv` (mini índice)

---

## Sumário

1. [O que é o modelo T(t+1)](#1-o-que-é-o-modelo-tt1)
2. [O cenário: os dados disponibilizados](#2-o-cenário-os-dados-disponibilizados)
3. [Como o modelo se aplica a esses dados](#3-como-o-modelo-se-aplica-a-esses-dados)
4. [Armadilhas e cuidados](#4-armadilhas-e-cuidados)
5. [Próximos passos de aplicação](#5-próximos-passos-de-aplicação)
6. [Estrutura sugerida do relatório final](#6-estrutura-sugerida-do-relatório-final)
7. [Referências](#7-referências)

---

## 1. O que é o modelo T(t+1)

### 1.1 Ideia central

O modelo **T(t+1)** é a **previsão um passo à frente** (*one-step-ahead forecast*). Com as informações disponíveis até o instante **t**, ele estima o valor da série no instante seguinte, **t+1**:

$$
\hat{y}_{t+1} = f\big(y_t,\ y_{t-1},\ \dots,\ y_{t-p},\ X_t\big)
$$

| Símbolo | Significado |
|---|---|
| $\hat{y}_{t+1}$ | Valor previsto para o próximo período |
| $y_t, y_{t-1}, \dots, y_{t-p}$ | Valores passados da própria série (*lags*) |
| $p$ | Quantidade de lags usados (tamanho da "memória" do modelo) |
| $X_t$ | Variáveis explicativas conhecidas em *t* (volume, volatilidade, outro ativo, horário…) |
| $f$ | O modelo (ingênuo, linear, ARIMA, árvore, rede neural…) |

A cada novo candle a janela avança um passo (*rolling*) e uma nova previsão é feita:

```
 tempo →   t-3    t-2    t-1     t   |  t+1
           [====== informação ======]|  [?]  ← previsão
                  [====== informação ======]|  [?]   (próximo passo)
```

### 1.2 O baseline obrigatório: modelo ingênuo (*naive*)

$$
\hat{y}_{t+1} = y_t
$$

O modelo ingênuo diz que **o próximo valor será igual ao atual**. Em preços de ativos financeiros ele é difícil de bater, porque os preços se comportam de forma muito próxima de um **passeio aleatório** (*random walk*):

$$
P_{t+1} = P_t + \varepsilon_{t+1}, \qquad \varepsilon \sim \text{ruído}
$$

> **Regra de ouro:** um modelo T(t+1) só tem valor se superar o naive **fora da amostra**.

### 1.3 O que prever: preço, retorno ou direção

| Alvo | Fórmula | Tipo de problema | Observação |
|---|---|---|---|
| Preço | $P_{t+1}$ | Regressão | **Evitar.** Série não estacionária; R² altíssimo e enganoso (o modelo só "copia" o último preço) |
| Log-retorno | $r_{t+1} = \ln(P_{t+1}/P_t)$ | Regressão | Série (aproximadamente) estacionária; é o alvo estatisticamente correto |
| Direção | $\text{sinal}(r_{t+1}) \in \{\text{sobe}, \text{desce}\}$ | Classificação | Mais intuitivo para operação; avaliado por acurácia direcional |

### 1.4 Famílias de modelos para o T(t+1)

1. **Naive** – $\hat{y}_{t+1} = y_t$ (retorno previsto = 0).
2. **Autorregressivo AR(p)** – $r_{t+1} = c + \phi_1 r_t + \phi_2 r_{t-1} + \dots + \phi_p r_{t-p+1} + \varepsilon$
3. **ARIMA(p, d, q)** – AR + diferenciação (*d*) + média móvel dos erros (*q*). Clássico de séries temporais.
4. **ARIMAX / SARIMAX** – ARIMA + variáveis exógenas (X) + componente sazonal (S). Ver [Seção 1.5](#15-sarimax-arima-com-sazonalidade-e-variáveis-exógenas).
5. **Regressão linear / logística com features** – lags + indicadores técnicos + variáveis exógenas.
6. **Árvores (Random Forest, XGBoost, LightGBM)** – capturam relações não lineares entre as features.
7. **Redes neurais recorrentes (LSTM/GRU)** – opcionais; exigem mais dados e cuidado com overfitting.

### 1.5 SARIMAX: ARIMA com sazonalidade e variáveis exógenas

O **SARIMAX** é o modelo estatístico mais completo da família ARIMA e o elo entre os modelos clássicos e os de machine learning:

```
SARIMAX(p, d, q)(P, D, Q, s)  =  S (sazonal)  +  ARIMA  +  X (exógenas)
```

| Parâmetro | Significado | No trabalho |
|---|---|---|
| p | Lags do próprio retorno (AR) | 1–3, escolhido por AIC/BIC e PACF |
| d | Diferenciações | **0**, porque o log-retorno já é estacionário |
| q | Lags dos erros de previsão (MA) | 0–2, escolhido por AIC/BIC e ACF |
| P, D, Q | Equivalentes sazonais de p, d, q | P = 1 para testar o padrão do pregão |
| s | Período sazonal (nº de candles que se repetem) | Candles por pregão (ver tabela abaixo) |
| X | Variáveis exógenas | Outro ativo, volume, horário (sempre **defasadas**) |

#### A parte "X": variáveis exógenas

É a parte que mais interessa ao trabalho, porque permite usar **as duas bases juntas**. Para prever o WIN:

$$
r^{WIN}_{t+1} = c + \underbrace{\phi_1 r^{WIN}_t + \dots}_{\text{AR}} + \underbrace{\theta_1 \varepsilon_t + \dots}_{\text{MA}} + \underbrace{\beta_1\, r^{WDO}_t + \beta_2\, \text{vol\_rel}_t + \dots}_{\text{X (exógenas)}} + \varepsilon_{t+1}
$$

Exógenas candidatas:
- **Retorno defasado do WDO** (ou do WIN, se o alvo for o WDO). Dólar e Ibovespa tendem a ter correlação negativa.
- Volume relativo e volatilidade recente.
- Dummies de horário (abertura, almoço, fechamento) ou termos de Fourier.

O coeficiente **β** e seu p-valor mostram se o outro ativo ajuda a explicar o próximo candle. Essa interpretabilidade é uma vantagem sobre XGBoost/LSTM.

> ⚠️ **Cuidado no T(t+1):** para prever *t+1*, o SARIMAX precisa do valor de X **em t+1**, que ainda não é conhecido. Por isso as exógenas entram **defasadas** (`shift(1)`): usa-se o retorno do WDO em *t* para prever o WIN em *t+1*. Usar o valor contemporâneo é **vazamento de dados** e produz um resultado artificialmente bom.

#### A parte "S": sazonalidade intradiária

No intradiário, a sazonalidade é o **padrão que se repete a cada pregão**: a abertura é mais agitada, o almoço mais calmo e o fechamento volta a acelerar. O período *s* é o número de candles em um pregão:

| Timeframe | s (≈ candles por pregão) | SARIMA viável? |
|---|---|---|
| 1 min | ~540 | Não, pesado demais |
| 5 min | ~108 | Pesado |
| 15 min | ~36 | Sim (ajuste lento) |
| 60 min | ~9 | Sim, leve |

Observações:
- O padrão intradiário aparece mais na **volatilidade e no volume** do que na **média do retorno**. O termo sazonal pode sair não significativo, e isso também é um resultado a reportar. Para modelar a volatilidade, o complemento natural é o **GARCH** (pacote `arch`).
- Uma alternativa mais leve ao "S" é colocar **dummies de horário** dentro do X, mantendo um ARIMAX simples.

#### Escada de modelos

Avaliar os modelos nesta ordem isola a contribuição de cada componente:

| Passo | Modelo | Pergunta que responde |
|---|---|---|
| 1 | Naive | Qual é o piso de comparação? |
| 2 | ARIMA | O passado do próprio ativo ajuda? |
| 3 | **ARIMAX** | O outro ativo e o volume acrescentam informação? |
| 4 | **SARIMAX** | O padrão do pregão acrescenta algo? |
| 5 | XGBoost / LSTM | Relações não lineares melhoram a previsão? |

---

## 2. O cenário: os dados disponibilizados

### 2.1 Os ativos

| Arquivo | Ativo | Descrição |
|---|---|---|
| `WDO$N_M1.csv` | **WDO** – Mini Dólar (B3) | Contrato futuro de dólar comercial, 10.000 USD por contrato; vencimento **mensal** |
| `WIN$N_M1.csv` | **WIN** – Mini Índice (B3) | Contrato futuro do Ibovespa, R$ 0,20 por ponto; vencimento nos **meses pares** |

O sufixo **`$N`** indica uma **série contínua**: os contratos que vão vencendo são "emendados" para formar um histórico único.

O sufixo **`M1`** indica candles de **1 minuto**.

### 2.2 Formato dos arquivos

Exportação no padrão MetaTrader: separador **TAB**, cabeçalhos entre `< >`.

```
<DATE>      <TIME>     <OPEN>    <HIGH>    <LOW>     <CLOSE>   <TICKVOL> <VOL>  <SPREAD>
2020.09.17  14:08:00   5248.500  5249.000  5244.500  5247.000  3244      10451  500
```

| Coluna | Significado |
|---|---|
| DATE / TIME | Data e hora de abertura do candle |
| OPEN / HIGH / LOW / CLOSE | Preço de abertura, máxima, mínima e fechamento do minuto |
| TICKVOL | Número de negócios (ticks) no minuto |
| VOL | Volume (contratos) negociado no minuto |
| SPREAD | Spread registrado pela plataforma |

### 2.3 Características observadas (análise preliminar)

| Métrica | WDO$N | WIN$N |
|---|---|---|
| Nº de candles | 717.232 | 757.716 |
| Início | 17/09/2020 14:08 | 15/10/2020 11:49 |
| Fim | 01/04/2026 16:07 | 25/09/2026 18:24 |
| Pregões | 1.288 | 1.368 |
| % minutos de alta | 42,5% | 47,3% |
| % minutos de baixa | 42,2% | 47,2% |
| % minutos sem variação | 15,3% | 5,5% |
| Autocorrelação do retorno (lag 1) | **−0,034** | **−0,009** |

*Retornos calculados apenas dentro do mesmo pregão.*

### 2.4 O que esses números dizem

- **Altas e baixas estão praticamente equilibradas.** Não há um viés direcional óbvio no minuto a minuto.
- **A autocorrelação está próxima de zero.** O retorno do minuto anterior quase não explica o seguinte, o que é coerente com a **Hipótese de Mercados Eficientes** (forma fraca).
- **O WDO tem leve autocorrelação negativa (−0,034).** É um indício fraco de **reversão à média** no curtíssimo prazo, em parte explicado pelo *bid-ask bounce* (o preço alterna entre a melhor compra e a melhor venda).
- **15% dos minutos do WDO não têm variação.** Se o alvo for direção, é preciso decidir como tratar a classe "estável".
- **Expectativa realista:** num problema de direção, acurácia de **52–55%** fora da amostra já é um resultado relevante. Números muito acima disso costumam indicar **vazamento de dados**.

### 2.5 Particularidades que exigem tratamento

1. **Rolagem de contratos.** Na troca de vencimento a série contínua pode apresentar **saltos artificiais** de preço (diferença entre o contrato velho e o novo). Esses retornos não são reais e precisam ser removidos ou ajustados.
2. **Gaps entre pregões.** O retorno entre o último minuto de um dia e o primeiro do dia seguinte inclui o *overnight* e não é comparável aos retornos intradiários.
3. **Horários distintos.** Leilão de abertura, horário de almoço e fechamento têm volatilidade e volume diferentes.
4. **Coluna SPREAD.** Pode variar conforme a fonte ou o período (por exemplo, valores 0 no final do WDO). Deve ser verificada antes de ser usada como feature.

---

## 3. Como o modelo se aplica a esses dados

### 3.1 Formulação proposta

**Opção A – Regressão do retorno**

$$
\hat{r}_{t+1} = f(r_t, r_{t-1}, \dots, r_{t-p},\ X_t)
$$

**Opção B – Classificação da direção (recomendada para o trabalho)**

$$
\hat{d}_{t+1} = P\big(r_{t+1} > 0 \mid r_t, \dots, r_{t-p},\ X_t\big)
$$

### 3.2 Escolha do timeframe

O passo "t → t+1" pode ser de 1, 5 ou 15 minutos (via reamostragem com `resample`). Os candles de 1 minuto são muito ruidosos. Agregar em 5 ou 15 minutos reduz o ruído e o peso relativo dos custos operacionais.

### 3.3 Features (variáveis de entrada)

Todas calculadas **somente com informação disponível até t**:

| Grupo | Exemplos |
|---|---|
| Lags do retorno | $r_t, r_{t-1}, \dots, r_{t-10}$ |
| Estrutura do candle | Amplitude (High − Low), corpo (Close − Open), posição do fechamento no candle |
| Volume | VOL, TICKVOL, variação do volume, volume relativo à média |
| Volatilidade | Desvio-padrão móvel dos retornos, ATR |
| Indicadores técnicos | Médias móveis (e distância do preço a elas), RSI, Bandas de Bollinger |
| Calendário | Hora/minuto do dia, dia da semana, minutos desde a abertura |
| **Cruzamento entre ativos** | Retornos defasados do **WDO** para prever o **WIN** e vice-versa |

> **Diferencial do trabalho:** dólar e Ibovespa costumam ter **correlação negativa**. Usar um ativo como variável exógena do outro aproveita as duas bases fornecidas e transforma o T(t+1) em um modelo **multivariado**.

### 3.4 Validação temporal

Em séries temporais **não se usa split aleatório**, porque isso mistura futuro com passado. As alternativas são:

**Split cronológico simples**

```
|------------ Treino ------------|--- Validação ---|--- Teste ---|
 2020 ........................ 2024      2025          2026
```

**Walk-forward (recomendado)**

```
Janela 1: [ Treino ][Teste]
Janela 2:    [ Treino ][Teste]
Janela 3:       [ Treino ][Teste]
...        → o modelo é re-treinado conforme o tempo avança
```

### 3.5 Métricas de avaliação

| Tipo | Métrica | Interpretação |
|---|---|---|
| Regressão | RMSE, MAE | Erro médio da previsão |
| Regressão | **Theil's U** = RMSE_modelo / RMSE_naive | U < 1 → melhor que o naive |
| Classificação | Acurácia direcional | % de acertos de sobe/desce |
| Classificação | Precisão, Recall, F1, Matriz de confusão | Qualidade por classe |
| Estatística | Teste de Diebold-Mariano | O ganho sobre o naive é significativo? |
| **Financeira** | Retorno acumulado, Sharpe, drawdown máximo, taxa de acerto | O modelo gera dinheiro **após custos**? |

### 3.6 Backtest simplificado

```
se previsão(t+1) = alta  → comprado no candle t+1
se previsão(t+1) = baixa → vendido no candle t+1
resultado = Σ (posição × retorno real) − custos (corretagem + emolumentos + slippage)
```

Uma conclusão perfeitamente válida e madura para o trabalho é mostrar que **um modelo com 53% de acerto pode não ser lucrativo depois dos custos**.

### 3.7 Teste preliminar: ARIMA × ARIMAX × SARIMAX

Rodada exploratória com o código da [Etapa 6](#etapa-6--modelagem-do-mais-simples-ao-mais-complexo).

**Configuração:**
- Alvo: log-retorno do **WIN** no próximo candle, em %.
- Exógenas: retorno do **WDO** em *t* e volume relativo do WIN em *t*.
- Treino até 31/12/2024, teste de 2025 em diante.
- Previsão T(t+1) feita com `res.apply()`, sem re-treinar.
- O baseline é prever retorno = 0 (equivalente ao naive no preço).

**Correlação entre os ativos:**

| | 15 min | 60 min |
|---|---|---|
| Correlação **contemporânea** WIN(t) × WDO(t) | **−0,50** | **−0,50** |
| Correlação **defasada** WIN(t+1) × WDO(t) | 0,004 | 0,012 |

> Este é o resultado mais importante do teste. O WDO explica muito bem o WIN **no mesmo candle**, mas quase nada **no candle seguinte**. Usar o WDO contemporâneo como exógena produziria um modelo "excelente" e falso. Isso confirma, com os dados, o alerta de vazamento da Seção 1.5.

**Desempenho fora da amostra (teste a partir de 2025):**

| Timeframe | Modelo | AIC | Theil U (vs. retorno = 0) | Acurácia direcional | Exógenas significativas? |
|---|---|---|---|---|---|
| 60 min | ARIMA(1,0,1) | 4145 | 1,0006 | 49,3% | – |
| 60 min | ARIMAX(1,0,1) | 4146 | 1,0017 | 48,8% | Não (p = 0,29 e 0,16) |
| 60 min | SARIMAX(1,0,1)(1,0,0,10) | 4146 | 1,0024 | 49,1% | Não; termo sazonal no limite (p = 0,047) |
| 15 min | ARIMA(1,0,1) | −19586 | 1,0000 | 50,9% | – |
| 15 min | ARIMAX(1,0,1) | −19584 | 1,0002 | 49,8% | Não (p = 0,23 e 0,32) |

**Leitura dos resultados:**
- **Nenhum modelo linear superou o baseline** (Theil U ≥ 1), e a acurácia ficou em torno de 50%. É coerente com a autocorrelação próxima de zero vista na Seção 2.3 e com a eficiência de mercado na forma fraca.
- No 60 min, os coeficientes AR (−0,95) e MA (+0,93) quase se cancelam. É um sinal clássico de que a série se comporta como **ruído branco** e o ARMA está apenas "ajustando ruído". Uma ordem menor, como (0,0,0) ou (1,0,0), deve ter desempenho equivalente.
- O componente sazonal (lag de 10 candles = 1 pregão) saiu no limite da significância, mas **não melhorou a previsão**. O padrão do pregão está na volatilidade, não na média.
- **Isso não é um fracasso do trabalho; é um resultado.** Os próximos passos naturais são: (a) testar o GARCH para a volatilidade, (b) testar modelos não lineares (XGBoost) com mais features, e (c) avaliar se há janelas específicas (abertura, divulgação de dados macro) onde a previsibilidade é maior.

> Rodada única e exploratória, sem walk-forward nem busca de ordens (p, q). Os números servem para orientar o trabalho, não como resultado final.

---

## 4. Armadilhas e cuidados

| Armadilha | Como evitar |
|---|---|
| **Vazamento de dados (look-ahead bias)** | Nenhuma feature pode usar dados de t+1. Usar `shift(1)` nas features e conferir cada cálculo |
| Exógenas contemporâneas no SARIMAX | Usar X defasado (`shift(1)`); o valor de X em t+1 não é conhecido no momento da previsão |
| SARIMAX lento em base grande | Reamostrar (15/60 min), ajustar só no treino e usar `res.apply()` no teste |
| Normalização com dados futuros | Ajustar `StandardScaler` / `MinMaxScaler` **apenas no treino** |
| Prever preço em vez de retorno | Trabalhar com log-retornos ou direção |
| Split aleatório | Usar split cronológico ou walk-forward |
| Saltos de rolagem | Remover retornos nos dias de troca de contrato |
| Retornos entre pregões | Calcular retornos só dentro do mesmo dia |
| Overfitting | Poucos hiperparâmetros, validação fora da amostra, comparar sempre com o naive |
| Ignorar custos | Incluir corretagem/emolumentos/slippage no backtest |
| Classe "estável" (retorno = 0) | Excluir, agrupar ou criar uma 3ª classe, e documentar a escolha |

---

## 5. Próximos passos de aplicação

### Etapa 0 – Alinhamento
- [ ] Confirmar com o professor a definição exata de "T(t+1)": alvo (fechamento, retorno, direção) e timeframe esperado
- [ ] Definir o ativo principal (WIN, WDO ou ambos)

### Etapa 1 – Ambiente
- [ ] Criar o notebook (`modelo_tt1.ipynb`)
- [ ] Instalar as bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `xgboost` / `lightgbm`

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost lightgbm
```

### Etapa 2 – Leitura e limpeza

```python
import pandas as pd, numpy as np

def carregar(caminho):
    df = pd.read_csv(caminho, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M:%S")
    return df.set_index("datetime").drop(columns=["date", "time"])

wdo = carregar("dados/WDON_M1/WDO$N_M1.csv")
win = carregar("dados/WINN_M1/WIN$N_M1.csv")
```

- [ ] Verificar duplicatas, valores nulos e candles inconsistentes (High < Low etc.)
- [ ] Identificar e tratar os saltos de rolagem
- [ ] (Opcional) Reamostrar para 5 ou 15 minutos

```python
def reamostrar(df, regra="5min"):
    return df.resample(regra).agg({"open": "first", "high": "max", "low": "min",
                                   "close": "last", "tickvol": "sum", "vol": "sum"}).dropna()
```

### Etapa 3 – Análise exploratória (EDA)
- [ ] Gráficos de preço e retornos ao longo do tempo
- [ ] Histograma dos retornos (caudas pesadas × normal)
- [ ] ACF e PACF dos retornos (`statsmodels.graphics.tsaplots`)
- [ ] Teste ADF: preço (não estacionário) × retorno (estacionário)
- [ ] Volatilidade e volume por horário do dia
- [ ] Correlação entre retornos do WDO e do WIN

### Etapa 4 – Construção do alvo e das features

```python
df = win.copy()
df["ret"] = np.log(df["close"]).diff()
df.loc[df.index.date != pd.Series(df.index.date, index=df.index).shift().values, "ret"] = np.nan  # remove gap entre dias

# ALVO: retorno / direção do PRÓXIMO candle
df["alvo_ret"] = df["ret"].shift(-1)
df["alvo_dir"] = (df["alvo_ret"] > 0).astype(int)

# FEATURES: apenas informação até t
for lag in range(1, 11):
    df[f"ret_lag{lag}"] = df["ret"].shift(lag - 1)
df["amplitude"] = (df["high"] - df["low"]) / df["close"]
df["vol_rel"]   = df["vol"] / df["vol"].rolling(30).mean()
df["volat_30"]  = df["ret"].rolling(30).std()
df["hora"]      = df.index.hour
df = df.dropna()
```

- [ ] Incluir features do outro ativo (alinhar índices de WDO e WIN por datetime)
- [ ] Conferir que **nenhuma** feature usa informação de t+1

### Etapa 5 – Divisão temporal
- [ ] Treino: 2020–2024 · Validação: 2025 · Teste: 2026 (ou walk-forward)

### Etapa 6 – Modelagem (do mais simples ao mais complexo)
- [ ] **Naive** (baseline)
- [ ] **AR(p) / ARIMA**, com escolha de p e q via AIC/BIC
- [ ] **ARIMAX**: ARIMA + retorno defasado do outro ativo + volume relativo
- [ ] **SARIMAX**: ARIMAX + componente sazonal do pregão (15 ou 60 min)
- [ ] (Opcional) **GARCH** para a volatilidade (`pip install arch`)
- [ ] **Regressão logística** (direção) ou **linear** (retorno)
- [ ] **Random Forest / XGBoost / LightGBM**
- [ ] (Opcional) **LSTM**

**Código base do ARIMA / ARIMAX / SARIMAX** (testado com as duas bases; resultados na [Seção 3.7](#37-teste-preliminar-arima--arimax--sarimax)):

```python
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

REGRA = "60min"   # timeframe do passo t → t+1 ("15min" também funciona)

def preparar(df, janela):
    out = pd.DataFrame(index=df.index)
    out["ret"] = np.log(df["close"]).diff() * 100           # log-retorno em %
    dia = pd.Series(df.index.date, index=df.index)
    out.loc[dia != dia.shift(), "ret"] = np.nan              # remove gap entre pregões
    out["vol_rel"] = df["vol"] / df["vol"].rolling(janela).mean()
    return out

# carregar() e reamostrar() definidas na Etapa 2
win_r = reamostrar(win, REGRA)
wdo_r = reamostrar(wdo, REGRA)
s = int(win_r.groupby(win_r.index.date).size().median())   # candles por pregão (60min ≈ 10, 15min ≈ 38)

df = (preparar(win_r, s).add_suffix("_win")
      .join(preparar(wdo_r, s).add_suffix("_wdo"), how="inner"))
df["ret_wdo_lag1"] = df["ret_wdo"].shift(1)                  # exógenas DEFASADAS
df["vol_rel_lag1"] = df["vol_rel_win"].shift(1)
df = df.dropna()

treino = df.index < "2025-01-01"
y = df["ret_win"].reset_index(drop=True)
X = df[["ret_wdo_lag1", "vol_rel_lag1"]].reset_index(drop=True)
y_tr, y_te = y[treino], y[~treino]
X_tr, X_te = X[treino], X[~treino]

modelos = {
    "ARIMA":   dict(order=(1, 0, 1), exog=None),
    "ARIMAX":  dict(order=(1, 0, 1), exog="X"),
    "SARIMAX": dict(order=(1, 0, 1), exog="X", seasonal_order=(1, 0, 0, s)),
}

for nome, cfg in modelos.items():
    usa_x = cfg.pop("exog") == "X"
    res = SARIMAX(y_tr, exog=X_tr if usa_x else None, **cfg).fit(disp=False)

    # T(t+1) no teste: incorpora cada nova observação sem re-treinar
    prev = res.apply(y_te, exog=X_te if usa_x else None).predict()

    theil = np.sqrt(((prev - y_te) ** 2).mean()) / np.sqrt((y_te ** 2).mean())
    acc   = (np.sign(prev) == np.sign(y_te))[y_te != 0].mean()
    print(f"{nome:8s} AIC={res.aic:.0f}  Theil U={theil:.4f}  acurácia={acc:.2%}")
    print(res.pvalues.round(4).to_dict())
```

> O SARIMAX com `s` alto (por exemplo 1 min, s ≈ 540) é inviável. Use 15 ou 60 minutos para o componente sazonal.

### Etapa 7 – Avaliação
- [ ] Tabela comparativa de todos os modelos contra o naive (RMSE, Theil's U, acurácia, F1)
- [ ] Teste de Diebold-Mariano
- [ ] Importância das features (árvores) / coeficientes (modelos lineares)

### Etapa 8 – Backtest
- [ ] Estratégia comprado/vendido baseada na previsão T(t+1)
- [ ] Curva de capital com e sem custos
- [ ] Sharpe, drawdown máximo, nº de operações

### Etapa 9 – Conclusões e relatório
- [ ] Discutir resultados à luz da eficiência de mercado
- [ ] Listar limitações e possíveis melhorias

---

## 6. Estrutura sugerida do relatório final

1. **Introdução** – objetivo e motivação da previsão T(t+1)
2. **Fundamentação teórica** – séries temporais, estacionariedade, random walk, eficiência de mercado
3. **Dados** – descrição do WDO e do WIN, período, tratamento (rolagem, gaps)
4. **Análise exploratória** – gráficos, ACF/PACF, ADF, correlação entre ativos
5. **Metodologia** – alvo, features, validação walk-forward, modelos
6. **Resultados** – comparação com o naive, métricas estatísticas e financeiras
7. **Discussão** – o modelo agrega valor? É lucrativo após custos?
8. **Conclusão e trabalhos futuros**

---

## 7. Referências

- Box, G. E. P.; Jenkins, G. M.; Reinsel, G. C. *Time Series Analysis: Forecasting and Control*. Wiley.
- Hyndman, R. J.; Athanasopoulos, G. *Forecasting: Principles and Practice*. OTexts (disponível online gratuitamente).
- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work*. Journal of Finance.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Documentação `statsmodels` – `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Diebold, F. X.; Mariano, R. S. (1995). *Comparing Predictive Accuracy*. Journal of Business & Economic Statistics.
- B3 – Especificações dos contratos Mini Índice (WIN) e Mini Dólar (WDO).
