# Plano das Frentes – Comparação de Modelos T(t+1)

> Trabalho da disciplina **Data Science no Mercado Financeiro**
> Origem: reunião de 02/11/2026 (Ana Beatriz e Itamar), transcrição em `reunioes/Reuniao_data_science_02.11.txt`
> Prazo: **tudo pronto hoje, 02/11, à noite**. Apresentação: **amanhã, 03/11**.
> Código comum: [`tt1_comum.py`](../tt1_comum.py) · Plano de referência anterior: [`PLANO_COMPARACAO_MODELOS_Tt+1.md`](PLANO_COMPARACAO_MODELOS_Tt+1.md)

**Como usar este arquivo:** cada pessoa entrega este MD ao Claude junto com a frase da Seção 9. O Claude segue as regras comuns (Seção 2) e o passo a passo (Seção 4) só com o modelo da sua frente. Assim as três frentes entregam os mesmos arquivos, com os mesmos nomes, as mesmas métricas e os mesmos gráficos, e a comparação no final é direta.

---

## Sumário

1. [Objetivo de hoje](#1-objetivo-de-hoje)
2. [Regras comuns (ninguém altera)](#2-regras-comuns-ninguém-altera)
3. [As três frentes](#3-as-três-frentes)
4. [Passo a passo (igual para as três frentes)](#4-passo-a-passo-igual-para-as-três-frentes)
5. [Entregáveis (checklist idêntico)](#5-entregáveis-checklist-idêntico)
6. [Consolidação e decisão da regra de operação](#6-consolidação-e-decisão-da-regra-de-operação)
7. [Roteiro da apresentação de amanhã](#7-roteiro-da-apresentação-de-amanhã)
8. [Armadilhas](#8-armadilhas)
9. [Instrução para colar no Claude](#9-instrução-para-colar-no-claude)

---

## 1. Objetivo de hoje

Entregar amanhã o mesmo nível do grupo mais avançado da turma: **modelo fazendo previsões e uma simulação de operações** mostrando quanto ele ganharia ou perderia.

Cada frente responde às mesmas quatro perguntas, com o seu modelo:

| # | Pergunta | Como responder |
|---|---|---|
| Q1 | Qual **tamanho de candle** e qual **janela de treino** dão o melhor resultado? | Grade candle × janela na validação (2024), heatmaps |
| Q2 | O modelo bate o **naive** fora da amostra? | Theil U < 1 e R²_oos > 0 |
| Q3 | Acertar mais é ganhar mais? | Acurácia direcional × esperança matemática na simulação |
| Q4 | Com uma regra de compra/venda (limiar, take profit, stop loss), a **esperança matemática** fica positiva depois dos custos? | Simulação no teste (01/2025 → 03/2026) |

> O professor cobra mais **interpretação** (o que o resultado quer dizer, por que se decidiu X) do que a teoria do modelo. Todo gráfico precisa de uma frase de conclusão e não pode contradizer a tabela.

---

## 2. Regras comuns (ninguém altera)

Tudo nesta seção está implementado em `tt1_comum.py`. **Importar o módulo, não copiar e editar.** Se algo precisar mudar, muda para as três frentes ao mesmo tempo, avisando no grupo.

### 2.1 Dados e alvo

| Item | Decisão |
|---|---|
| Ativo | **WIN** (mini índice). WDO só como exógena na variante do estatístico |
| Alvo | Log-retorno do próximo candle, em %: `alvo = 100·ln(P(t+1)/P(t))` → prevê **quanto** sobe ou desce, não só a direção |
| Fronteira de pregão | O último candle do dia não tem alvo; o 1º candle usa abertura → fechamento (sem gap noturno) |
| Features | `LAGS` = retornos defasados 1–10 · `COMPLETO` = LAGS + amplitude, corpo, volume relativo, volatilidade recente, hora, nº do candle no dia |

### 2.2 Grade de experimentos (decisão da reunião)

| Dimensão | Valores | Observação |
|---|---|---|
| Tamanho do candle | **1 min, 15 min, 60 min** | 2 e 3 min ficam como extra, se sobrar tempo |
| Janela de treino (deslizante) | **1, 3, 6 e 12 meses** para trás | O modelo nunca vê dado futuro |
| Re-treino | **Trimestral** | Treina com os últimos N meses e prevê os 3 meses seguintes |
| Período de escolha (validação) | **01/01/2024 → 31/12/2024** | Escolhe candle, janela e regra de operação |
| Período de teste | **01/01/2025 → 31/03/2026** | Roda **uma vez**, com as escolhas congeladas |

São **12 combinações** (3 candles × 4 janelas) por modelo. Todas as frentes rodam as 12.

```
Validação 2024, janela de 6 meses, re-treino trimestral:
  treino jul–dez/2023 → prevê jan–mar/2024
  treino out/23–mar/24 → prevê abr–jun/2024
  treino jan–jun/2024 → prevê jul–set/2024
  treino abr–set/2024 → prevê out–dez/2024
```

### 2.3 Simulação de operações (comum)

Função `simular` do `tt1_comum.py`:

- **Compra** se previsão > +limiar; **vende** se previsão < −limiar; senão fica de fora.
- Entra no fechamento do candle *t* e sai no fechamento de *t+1*, ou antes se tocar o **take profit** ou o **stop loss** dentro do candle *t+1*.
- TP e SL em múltiplos da volatilidade recente do ativo. Se os dois forem tocados no mesmo candle, conta o **stop** (hipótese conservadora).
- Custo padrão: **5 pontos** por operação (ida + volta). Sensibilidade: 0, 5 e 10 pontos. Valor do ponto: R$ 0,20 por contrato.

Grade da regra (escolhida **só na validação**):

| Parâmetro | Valores |
|---|---|
| Limiar (quantil de \|previsão\|) | 0 (opera sempre), 50%, 75%, 90% |
| Take profit | sem, 1×, 2× a volatilidade |
| Stop loss | sem, 1×, 2× a volatilidade |

**Critério de escolha:** maior **esperança matemática por operação** (em pontos, já descontado o custo). A regra só é considerada lucrativa se a esperança for **> 0** e o **fator de lucro** (soma dos ganhos ÷ soma das perdas) for **> 1**.

> Lição do outro grupo: o modelo acertava 90% das vezes e mesmo assim perdia dinheiro, porque ganhava pouco quando acertava e perdia muito quando errava. Por isso o critério é a esperança, não a acurácia. Pode acertar menos e ganhar mais.

### 2.4 Referências que todos reportam

| Referência | Para quê |
|---|---|
| **Naive** (retorno previsto = 0) | Régua das métricas de previsão (Theil U = 1, R²_oos = 0) |
| **Sinais aleatórios** (`sinais_aleatorios`) | Régua da simulação: o modelo precisa ganhar de jogar moeda |

---

## 3. As três frentes

Cada frente usa **um modelo base fixo** na grade (mesmos hiperparâmetros nas 12 combinações) e **uma variante** só na melhor combinação. O trabalho é o mesmo para as três; muda só o modelo.

| Frente | Responsável | Modelo base (grade) | Features | Variante (só na melhor combinação) |
|---|---|---|---|---|
| Estatístico | **Itamar** | ARIMA(1, 0, 1) | `LAGS` (o ARIMA usa só o próprio retorno) | **ARIMAX** com o retorno defasado do WDO; **SARIMAX** sazonal (só 15 e 60 min) |
| Machine learning | **Ana Beatriz** | Random Forest | `COMPLETO` | Grade pequena de hiperparâmetros (≤ 6 combinações) + importância das features |
| Redes neurais | **Gabriel Modesto** | MLP (rede densa) | `COMPLETO` | **GRU/LSTM** com sequência dos últimos 10 retornos |

O naive é a base comum: o `tt1_comum.metricas` já calcula tudo contra ele, então ninguém precisa rodá-lo à parte.

> **LightGBM saiu do escopo:** o professor disse que não faz sentido no projeto, e não vale testar vários modelos da mesma família (árvores).

### 3.1 Estatístico – Itamar

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ArimaT1:
    """ARIMA/ARIMAX/SARIMAX: estima no treino; no teste só incorpora as novas observações (extend),
    então a previsão de t+1 usa só dados até t, sem re-estimar parâmetros."""
    usa_y_teste = True
    def __init__(self, order=(1, 0, 1), seasonal_order=(0, 0, 0, 0), usa_x=False):
        self.kw = dict(order=order, seasonal_order=seasonal_order); self.usa_x = usa_x
    def fit(self, X, y):
        self.res = SARIMAX(y.values, exog=X.values if self.usa_x else None, **self.kw).fit(disp=False)
        return self
    def predict(self, X, y):
        return self.res.extend(y.values, exog=X.values if self.usa_x else None).predict()

MODELO, FABRICA, FEATS = "arima", ArimaT1, "LAGS"
```

Variante, na melhor combinação candle × janela:
- **ARIMAX:** `lambda: ArimaT1(usa_x=True)` com features `["ret_wdo_lag1"]` (retorno do WDO em *t*, nunca em *t+1*).
- **SARIMAX:** `lambda: ArimaT1(seasonal_order=(1, 0, 0, s), usa_x=True)`, com `s` = candles por pregão devolvido pelo `montar_base` (só 15 e 60 min; em 1 min é pesado demais).
- **Gráfico extra que embasa a conclusão:** correlação WIN × WDO contemporânea (≈ −0,50) contra a correlação com 1 candle de defasagem (≈ 0). Mostra por que o X não melhora a previsão: a reação entre os dois é instantânea. Bate com o teste exploratório que a Ana Beatriz fez com o Claude (rodou o SARIMAX e o X não fez diferença).

### 3.2 Machine learning – Ana Beatriz

```python
from sklearn.ensemble import RandomForestRegressor

FABRICA = lambda: RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=100,
                                        max_features=0.5, n_jobs=-1, random_state=42)
MODELO, FEATS = "rf", "COMPLETO"
```

Variante, na melhor combinação:
- Grade de até 6 combinações: `max_depth ∈ {4, 8, 12}` × `min_samples_leaf ∈ {50, 200}`, escolhida pela esperança na validação.
- Importância das features (`feature_importances_`) em gráfico de barras.
- Comparar o tempo de treino com o da árvore simples (mencionado na reunião).

### 3.3 Redes neurais – Gabriel Modesto

```python
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FABRICA = lambda: make_pipeline(StandardScaler(),
                                MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True,
                                             max_iter=200, random_state=42))
MODELO, FEATS = "mlp", "COMPLETO"
```

Variante, na melhor combinação: GRU (Keras, no Colab). Esqueleto **não testado localmente** (TensorFlow não está instalado na máquina); conferir no Colab:

```python
import numpy as np, tensorflow as tf

class GruT1:
    """GRU sobre a sequência dos últimos 10 retornos (ret_lag10 … ret_lag1). Usar com FEATS = "LAGS"."""
    def fit(self, X, y):
        Xs = X[X.columns[::-1]].values[..., None]            # (n, 10, 1), do mais antigo ao mais recente
        self.esc = y.std()
        tf.keras.utils.set_random_seed(42)
        self.m = tf.keras.Sequential([tf.keras.layers.Input((Xs.shape[1], 1)),
                                      tf.keras.layers.GRU(16), tf.keras.layers.Dense(1)])
        self.m.compile("adam", "mse")
        self.m.fit(Xs / self.esc, y.values / self.esc, epochs=10, batch_size=512, validation_split=0.1,
                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=0)
        return self
    def predict(self, X):
        Xs = X[X.columns[::-1]].values[..., None]
        return self.m.predict(Xs / self.esc, verbose=0).ravel() * self.esc
```

Observação: a sequência de lags atravessa a virada do dia (o 1º candle do dia não carrega o gap noturno). É uma simplificação aceita para caber no prazo; registrar como limitação.

---

## 4. Passo a passo (igual para as três frentes)

Rodar no Colab (ou local) num notebook dentro da pasta da frente: `frentes/estatistico/frente_estatistico.ipynb`, `frentes/ml/frente_ml.ipynb`, `frentes/redes/frente_redes.ipynb`. Saídas em `resultados/{frente}/`, com `{frente}` ∈ `estatistico`, `ml`, `redes`.

### Etapa A – Preparação (≈ 10 min)

```python
import sys; sys.path.append(PASTA)          # PASTA = raiz do repositório (onde está o tt1_comum.py)
import tt1_comum as t
win = t.carregar(f"{PASTA}/dados/WINN_M1/WIN$N_M1.csv")
wdo = t.carregar(f"{PASTA}/dados/WDON_M1/WDO$N_M1.csv")
FRENTE = "estatistico"                      # ou "ml" / "redes"
SAIDA = f"{PASTA}/resultados/{FRENTE}"
```

**Pronto quando:** `t.montar_base(win, wdo, "15min")` devolve ~45 mil linhas e `s = 38`.

### Etapa B – Grade candle × janela na validação (Q1, Q2)

```python
tab = t.rodar_grade(win, wdo, MODELO, FABRICA, FEATS, SAIDA)          # 12 combinações, período 2024
t.heatmap(tab, "TheilU", f"{MODELO}: Theil U (validação 2024)", f"{SAIDA}/heatmap_theilU.png", menor_melhor=True)
t.heatmap(tab, "acc_dir", f"{MODELO}: acurácia direcional", f"{SAIDA}/heatmap_acuracia.png", fmt=".3f")
t.heatmap(tab, "sim_esperanca_pts", f"{MODELO}: esperança por operação (pts)", f"{SAIDA}/heatmap_esperanca.png", fmt=".2f")
```

Escolher a **melhor combinação** pela maior `sim_esperanca_pts` (empate: menor Theil U). Escrever uma frase do tipo "com candle de X e janela de Y meses o modelo teve o melhor resultado porque…".

**Pronto quando:** `grade_val.csv` e os 3 heatmaps estão salvos.

### Etapa C – Regra de operação na validação (Q3, Q4)

```python
import pandas as pd
melhor = tab.sort_values(["sim_esperanca_pts", "TheilU"], ascending=[False, True]).iloc[0]
CANDLE, JANELA = melhor.candle, int(melhor.janela_meses)
prev_val = pd.read_csv(f"{SAIDA}/previsoes/WIN_{CANDLE}_{JANELA}m_{MODELO}_val.csv", index_col=0, parse_dates=True)
regras = t.grade_regras(prev_val, custo_pts=5)
regras.to_csv(f"{SAIDA}/regras_val.csv", index=False)
regra = regras.iloc[0]                       # maior esperança por operação
```

**Pronto quando:** `regras_val.csv` salvo e a regra escolhida anotada (limiar, TP, SL).

### Etapa D – Variante da frente (Seção 3)

Rodar a variante só em `CANDLE` × `JANELA`, na validação, com `t.walk_forward` + `t.metricas` + `t.resumo_operacoes(t.simular(...))`. Salvar em `{SAIDA}/variante_val.csv` (uma linha por variante, mesmas colunas da grade). Se a variante for melhor que o modelo base, ela passa a ser o modelo da frente no teste.

### Etapa E – Congelar e rodar o teste (uma vez)

```python
import json
config = dict(frente=FRENTE, modelo=MODELO, candle=CANDLE, janela_meses=JANELA,
              limiar=float(regra.limiar), tp_mult=regra.tp_mult, sl_mult=regra.sl_mult)
json.dump(config, open(f"{SAIDA}/config_final.json", "w"), indent=2, default=str)

df, s = t.montar_base(win, wdo, CANDLE)
prev_teste = t.walk_forward(df, t.features()[FEATS], FABRICA, *t.TESTE, janela_meses=JANELA)
prev_teste.to_csv(f"{SAIDA}/previsoes/WIN_{CANDLE}_{JANELA}m_{MODELO}_teste.csv")

tp = None if pd.isna(regra.tp_mult) else regra.tp_mult
sl = None if pd.isna(regra.sl_mult) else regra.sl_mult
linhas = []
for custo in t.CUSTOS_PTS:
    linhas.append(dict(serie="modelo", custo_pts=custo, **t.metricas(prev_teste),
                       **t.resumo_operacoes(t.simular(prev_teste, regra.limiar, tp, sl, custo))))
    linhas.append(dict(serie="aleatorio", custo_pts=custo,
                       **t.resumo_operacoes(t.simular(t.sinais_aleatorios(prev_teste), 0, tp, sl, custo))))
pd.DataFrame(linhas).to_csv(f"{SAIDA}/teste_final.csv", index=False)
```

**Não voltar para ajustar nada depois de ver o teste.** Se o resultado for ruim, ele é o resultado.

### Etapa F – Gráficos finais e resumo

1. `curva_capital_teste.png`: resultado acumulado em R$ ao longo do teste, modelo × aleatório, com custo de 5 pontos (`ops.pontos.cumsum() * t.PONTO_WIN`).
2. `previsto_vs_real_teste.png`: dispersão `y_pred` × `y_true` (mostra se o modelo prevê a magnitude ou só fica perto de zero).
3. `resumo.md`, com exatamente estas 5 linhas:
   - Melhor candle e janela, e por quê (Q1)
   - Theil U e R²_oos no teste: bate ou não o naive (Q2)
   - Acurácia × esperança: acertar mais deu mais lucro? (Q3)
   - Regra escolhida e resultado no teste com custo 0/5/10 pts (Q4)
   - Uma limitação ou surpresa

---

## 5. Entregáveis (checklist idêntico)

Cada frente entrega, em `resultados/{frente}/`:

- [ ] `previsoes/WIN_{candle}_{janela}m_{modelo}_val.csv` (12 arquivos) e `..._teste.csv` (1 arquivo)
- [ ] `grade_val.csv` (12 linhas: candle, janela, tempo, métricas, simulação)
- [ ] `heatmap_theilU.png`, `heatmap_acuracia.png`, `heatmap_esperanca.png`
- [ ] `regras_val.csv` (36 linhas: limiar × TP × SL)
- [ ] `variante_val.csv`
- [ ] `config_final.json`
- [ ] `teste_final.csv` (modelo e aleatório, custos 0/5/10)
- [ ] `curva_capital_teste.png`, `previsto_vs_real_teste.png`
- [ ] `resumo.md` (5 linhas)
- [ ] 2 slides: (1) heatmaps + escolha de candle/janela; (2) curva de capital + tabela do teste

Formato das previsões (igual para todos): `datetime, y_true, y_pred, close, high_prox, low_prox, close_prox, volat`.

---

## 6. Consolidação e decisão da regra de operação

Quando as três frentes terminarem (hoje à noite), juntar as tabelas num único notebook `resultados/comparacao/comparacao_frentes.ipynb`:

```python
import pandas as pd
frentes = ["estatistico", "ml", "redes"]
grade = pd.concat([pd.read_csv(f"{PASTA}/resultados/{f}/grade_val.csv").assign(frente=f) for f in frentes])
teste = pd.concat([pd.read_csv(f"{PASTA}/resultados/{f}/teste_final.csv").assign(frente=f) for f in frentes])
```

Na sincronização, o grupo decide junto:
1. Qual combinação candle × janela foi a melhor **no geral** (se as três frentes concordam, a conclusão fica forte).
2. Qual frente teve a maior esperança no teste.
3. Se vale fixar **uma regra única** de compra/venda para os três modelos e comparar só o modelo (rodar `t.simular` com a mesma regra nos três arquivos de teste).

Tabela final para a apresentação:

| Frente | Modelo | Candle | Janela | Theil U | R²_oos | Acurácia | Nº ops | Taxa de acerto | Esperança (pts) | Fator de lucro | Resultado R$ (5 pts) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | Naive / aleatório | … | … | 1,000 | 0 | … | … | … | … | … | … |
| Estatístico | ARIMA / ARIMAX | … | … | … | … | … | … | … | … | … | … |
| ML | Random Forest | … | … | … | … | … | … | … | … | … | … |
| Redes | MLP / GRU | … | … | … | … | … | … | … | … | … | … |

---

## 7. Roteiro da apresentação de amanhã

1. **Problema e dados** – T(t+1) no WIN; alvo = retorno do próximo candle (quanto sobe/desce)
2. **Achado da EDA** – correlação WIN × WDO forte e instantânea; resultado do ARIMAX mostrando que o X não ajuda a prever
3. **Protocolo** – janela deslizante, re-treino trimestral, validação 2024 e teste 2025–2026, sem dado futuro
4. **Escolha de candle e janela** – heatmaps das três frentes lado a lado
5. **Comparação dos modelos** – tabela da Seção 6
6. **Simulação** – curvas de capital, efeito dos custos, acurácia × lucro
7. **Conclusões e próximos passos** – refinar a regra (limiar de certeza, stop), testar mais janelas, WDO como ativo

---

## 8. Armadilhas

| Armadilha | Como evitar |
|---|---|
| Escolher candle, janela ou regra olhando o teste | Tudo é escolhido em 2024; o teste roda uma vez |
| Usar o WDO em *t+1* como exógena | Só `ret_wdo_lag1` (WDO em *t*). O valor contemporâneo daria um modelo "perfeito" e falso |
| Cada frente alterar o `tt1_comum.py` | Importar sem editar; mudança só em conjunto |
| Avaliar pela acurácia | Critério é a esperança por operação após custos |
| Gráfico contraditório com a tabela | Conferir os números do gráfico com o CSV antes de montar o slide |
| Rede neural com previsão gigante ou constante | Conferir `y_pred.std()`; o MLP de teste ficou com Theil U ≈ 1,15 (pior que o naive), então vale reportar e, se der, aumentar `alpha` |
| 1 min com 12 meses lento demais | Se passar de ~10 min por combinação, registrar o tempo e pular (o tempo também é resultado) |

**Teste rápido do código comum** (WIN, 15 min, janela de 6 meses, validação 2024, custo 5 pts), só para validar o pipeline:

| Modelo | Theil U | Acurácia | Melhor regra na validação | Esperança (pts/op) | Fator de lucro |
|---|---|---|---|---|---|
| Naive | 1,000 | – | – | – | – |
| ARIMA(1,0,1) | 1,0005 | 49,8% | limiar 75%, SL 1× | −2,57 | 0,96 |
| Random Forest (100 árvores) | 1,0030 | 50,4% | limiar 90%, TP 1× | −2,29 | 0,97 |
| MLP (32, 16) | 1,1551 | 49,8% | limiar 90%, SL 1× | −4,19 | 0,94 |
| Sinais aleatórios | – | 47,1% | – | −7,18 | 0,88 |

Tempos: ARIMA 15 min ≈ 1 s por combinação; ARIMA 1 min com janela de 12 meses ≈ 4 s por trimestre. Esses números são ponto de partida e não substituem a grade completa.

---

## 9. Instrução para colar no Claude

> Leia `planos/PLANO_FRENTES_02.11.md` e `tt1_comum.py`. Sou da frente **[estatístico / ML / redes neurais]**. Monte o notebook `frentes/[estatistico / ml / redes]/frente_[estatistico / ml / redes].ipynb` seguindo as Etapas A a F da Seção 4 com o modelo da minha frente (Seção 3), sem alterar o `tt1_comum.py` e sem mudar a grade, os períodos, a simulação ou os nomes dos arquivos de saída. No final, confira o checklist da Seção 5.
