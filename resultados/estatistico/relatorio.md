# Relatório da frente estatística – ARIMA / ARIMAX / SARIMAX

> Responsável: Itamar da Silva Rocha Filho · Notebook: [`frentes/estatistico/frente_estatistico.ipynb`](../../frentes/estatistico/frente_estatistico.ipynb) · Resumo de 5 linhas: [`resumo.md`](resumo.md) · Plano: [`planos/PLANO_FRENTES_02.11.md`](../../planos/PLANO_FRENTES_02.11.md)

> **Atualização (nov/2026).** Este relatório descreve o protocolo de 02/11 e não foi reescrito. A reabordagem da frente ([`NOTAS.md`](../../frentes/estatistico/NOTAS.md) e [estudos](../../frentes/estatistico/estudos/README.md)) corrigiu ou completou quatro pontos:
> 1. **Comparação com o aleatório (Seção 5.5):** o aleatório operava todos os candles. Nos mesmos 214 candles do modelo, ele dá −7,9 pts/op (faixa de 90%: −68 a +55); o modelo (−41) fica no percentil 19, **indistinguível do acaso**. A frase "por operação ele perde mais" não vale.
> 2. **Custo de 5 pts:** era um exemplo do plano, nunca derivado. Ver o modelo de custo em [`custos.py`](../../frentes/estatistico/custos.py): 2,5 / 7,5 / 12,5 pts por cenário de execução, e o **custo de equilíbrio** de cada estratégia.
> 3. **Bandas por horário (Seção 6):** resolvidas no [Estudo 07](../../frentes/estatistico/estudos/07_h6_volatilidade.ipynb). Com perfil intradiário recente × nível do dia (HAR), a cobertura das 10h passa de 74% para ~94%.
> 4. **Cone diário (Seção 6.5):** usa fechamento → fechamento da série contínua sem ajuste de rolagem e inclui um salto falso de ~+1,7% a cada 2 meses. Correção pendente.

Este relatório explica **o que foi feito, por quê e o que os resultados querem dizer**. O `resumo.md` segue o formato fixo de 5 linhas do plano; aqui está o raciocínio completo, para estudar antes da apresentação.

---

## Sumário

1. [Em uma frase](#1-em-uma-frase)
2. [O modelo prevê retorno ou preço?](#2-o-modelo-prevê-retorno-ou-preço)
3. [O que é cada modelo, no nosso problema](#3-o-que-é-cada-modelo-no-nosso-problema)
4. [O pipeline passo a passo e as decisões](#4-o-pipeline-passo-a-passo-e-as-decisões)
5. [Resultados em detalhe](#5-resultados-em-detalhe)
6. [As bandas de previsão do SARIMAX](#6-as-bandas-de-previsão-do-sarimax)
7. [Por que a ARIMAX foi escolhida e não funcionou no teste](#7-por-que-a-arimax-foi-escolhida-e-não-funcionou-no-teste)
8. [Limitações](#8-limitações)
9. [Para a apresentação](#9-para-a-apresentação)
10. [Arquivos gerados](#10-arquivos-gerados)

---

## 1. Em uma frase

A família ARIMA (com ou sem o dólar como exógena, com ou sem sazonalidade) **não consegue prever a direção nem a magnitude do retorno do próximo candle do WIN**: o erro é igual ao de prever "zero" (naive) e a simulação de operações perde dinheiro. O que o modelo **consegue** fornecer é uma **banda de risco** (até onde o preço pode ir), e mesmo essa banda erra por não acompanhar a volatilidade, que muda ao longo do pregão.

---

## 2. O modelo prevê retorno ou preço?

**Prevê o retorno**, e do retorno sai o preço.

O alvo, definido no protocolo comum (`tt1_comum.montar_base`), é o **log-retorno do próximo candle, em %**:

```
alvo(t) = 100 · ln( P(t+1) / P(t) )
```

O modelo devolve `r̂ = alvo previsto`, e o preço previsto sai direto dele:

```
P̂(t+1) = P(t) · exp( r̂ / 100 )
```

Exemplo: WIN em 141.000 pontos e previsão de +0,02% → P̂(t+1) = 141.000 · e^0,0002 ≈ 141.028 pontos (+28 pts).

**Por que modelar o retorno e não o preço direto:**

| Motivo | Explicação |
|---|---|
| **Estacionariedade** | O preço tem tendência e nível que mudam (o WIN foi de ~129 mil em 2024 para ~141 mil em 2025). O ARIMA precisa de uma série estacionária, com média e variância estáveis. O retorno é estacionário; o preço não. |
| **É o mesmo modelo** | Um ARIMA(1, **0**, 1) no log-retorno é exatamente um ARIMA(1, **1**, 1) no log-preço. O "I" (integração, a diferença) do ARIMA é justamente transformar preço em retorno. Fizemos a diferença "à mão" (no `montar_base`) e usamos d = 0. |
| **Comparação justa** | Uma previsão de preço parece ótima mesmo quando é inútil: prever "o preço de amanhã = o de hoje" erra pouquíssimo em %, porque o preço muda pouco de um candle para o outro. No retorno não há esse truque: o naive (retorno = 0) é a régua e fica explícito se o modelo acrescenta algo. |
| **Comparabilidade entre frentes** | ML e redes neurais preveem o mesmo alvo, com as mesmas métricas. |

Então o modelo **prevê "quanto sobe ou desce", não só a direção**: a previsão é um número contínuo em %. O problema, visto na Seção 5, é que esse número sai sempre minúsculo, perto de zero.

---

## 3. O que é cada modelo, no nosso problema

Notação: `r(t)` = retorno do candle *t*; `ε(t)` = choque (a parte imprevisível); `x(t)` = retorno do WDO no candle *t*.

| Modelo | Equação (simplificada) | Em palavras |
|---|---|---|
| **ARIMA(1,0,1)** | r(t+1) = φ·r(t) + ε(t+1) + θ·ε(t) | O próximo retorno depende do retorno atual (parte **AR**, φ) e do "susto" do candle atual (parte **MA**, θ). |
| **ARIMAX** | ARIMA + β·x(t) | Acrescenta uma variável **eX**ógena: o retorno do dólar no candle atual. |
| **SARIMA** | ARIMA + Φ·r(t+1−s) | Acrescenta um termo **S**azonal: o retorno do "mesmo horário do pregão anterior" (s candles atrás). |
| **SARIMAX** | SARIMA + β·x(t) | As duas coisas juntas. |

Detalhes de implementação (classe `ArimaT1`, da Seção 3.1 do plano):

- **Sem constante** (`trend` padrão do `SARIMAX` = nenhum): assume retorno médio zero. É razoável, porque a deriva de 1 hora é desprezível perto da volatilidade.
- **Estimação por máxima verossimilhança** com filtro de Kalman (`statsmodels`). Além dos coeficientes, o modelo estima **σ²**, a variância do choque. É dela que saem as bandas da Seção 6.
- **`extend`, não re-estimação**: no início de cada trimestre os parâmetros são estimados com a janela de treino. Durante o trimestre eles ficam **fixos**, e o modelo só "lê" cada novo retorno observado para atualizar seu estado. A previsão de *t+1* usa apenas dados até *t*. É como o modelo funcionaria em produção, e sem vazamento de dado futuro.

---

## 4. O pipeline passo a passo e as decisões

Legenda da coluna **Origem**: *Reunião/plano* = decidido pelo grupo em 02/11 e implementado no `tt1_comum.py`; *Frente* = decisão tomada nesta implementação da frente estatística.

### Etapa A – Dados

| Decisão | Origem | Por quê |
|---|---|---|
| Ativo previsto: WIN; WDO só como exógena | Reunião/plano | Foco num ativo só; o WDO entra para testar se o dólar ajuda a prever o índice. |
| Candles de 1 min reamostrados para 15 e 60 min | Reunião/plano | O tamanho do candle muda a relação sinal/ruído e o peso do custo. |
| 1º candle do dia usa abertura → fechamento | Reunião/plano | Não misturar o **gap noturno** (notícias da noite) com o movimento intradiário. |
| Último candle do dia não tem alvo | Reunião/plano | Seu "próximo candle" seria no dia seguinte, atravessando a noite. |

### Etapa B – Grade candle × janela (validação 2024)

| Decisão | Origem | Por quê |
|---|---|---|
| ARIMA(1,0,1) fixo nas 12 combinações | Reunião/plano | A pergunta é "qual candle/janela", não "qual ordem do ARIMA". Variar os dois ao mesmo tempo multiplicaria as comparações e o risco de achar algo por acaso. |
| Janela deslizante de 1, 3, 6, 12 meses | Reunião/plano | Testa a troca entre **mais dados** (estimativa estável) e **dados recentes** (mercado muda). |
| Re-treino trimestral | Reunião/plano | Simula o uso real: re-estima a cada 3 meses e prevê os 3 meses seguintes. |
| Embargo de 1 linha no fim do treino | Reunião/plano | O alvo da última linha do treino é o retorno do 1º candle do teste. Sem o embargo, o modelo veria um dado do período de teste. |
| Validação = 2024, teste = 01/2025 → 03/2026 | Reunião/plano | Toda escolha é feita em 2024. O teste roda **uma vez**, com tudo congelado, para medir o desempenho fora da amostra sem viés. |
| Melhor combinação = maior esperança por operação (empate: menor Theil U) | Reunião/plano | O objetivo final é ganhar dinheiro, não acertar a direção (lição do outro grupo). |

### Etapa C – Regra de operação

| Decisão | Origem | Por quê |
|---|---|---|
| Compra se previsão > +limiar, vende se < −limiar | Reunião/plano | Operar só quando o modelo está "mais convicto". |
| Limiar = quantil (0, 50, 75, 90%) do \|previsão\| medido **na validação** | Reunião/plano | A escala da previsão muda de modelo para modelo; o quantil torna a regra comparável. Medir na validação e congelar o valor absoluto evita olhar o teste. |
| Entra no fechamento de *t*, sai no de *t+1*, ou no TP/SL dentro de *t+1* | Reunião/plano | Operação de 1 candle, coerente com a previsão T+1. |
| TP/SL = 1× ou 2× a volatilidade recente (desvio do retorno do último pregão) | Reunião/plano | O alvo se adapta ao momento do mercado. Em 60 min, 1× ≈ 0,25% ≈ 360 pts num candle típico do teste, mas ≈ **607 pts** (mediana; de ~320 a ~1.080) nos candles em que o modelo de fato operou, porque ele tende a operar quando a volatilidade está alta. |
| TP e SL no mesmo candle → conta o stop | Reunião/plano | Com candle de 60 min não se sabe o que veio primeiro; supor o pior evita superestimar o lucro. |
| Custo de 5 pts por operação (sensibilidade 0/5/10) | Reunião/plano | Corretagem, emolumentos e slippage. |
| Referência: sinais aleatórios (direção sorteada, mesma regra de TP/SL) | Reunião/plano | O modelo precisa ganhar de "jogar moeda". |

### Etapa D – Variantes

| Decisão | Origem | Por quê |
|---|---|---|
| ARIMAX com `ret_wdo_lag1` = WDO em *t* (nunca em *t+1*) | Reunião/plano | Usar o WDO de *t+1* seria usar informação futura: o modelo pareceria ótimo e seria inútil na prática. |
| SARIMAX só em 15 e 60 min | Reunião/plano | Em 1 min, s ≈ 565 candles por pregão torna a estimação inviável. |
| **SARIMA sem exógena** (linha a mais) | **Frente** | Separa os dois efeitos: se o SARIMAX melhorasse, saberíamos se foi pela sazonalidade ou pelo WDO. |
| **Período sazonal efetivo s − 1** (além do `s` do plano) | **Frente** | O `montar_base` calcula `s` (10 candles/pregão em 60 min) **antes** de descartar o último candle do dia. Na série que o modelo vê, cada pregão tem **9** linhas, então "mesmo horário de ontem" fica 9 posições atrás, não 10. A autocorrelação do \|retorno\| confirma: 0,149 na defasagem 9 contra 0,044 na defasagem 10. Rodamos os dois valores, por transparência. |
| **Regra para escolher o modelo final, fixada antes do teste:** maior esperança com a melhor regra na validação (empate → ARIMA) | **Frente** | O plano diz "se a variante for melhor, ela passa a ser o modelo da frente", mas não diz por qual métrica. Usamos o critério oficial (esperança) e o fixamos no código antes de rodar o teste. |
| **Erro-padrão da esperança** em cada linha | **Frente** | Mostra se a diferença entre modelos é real ou ruído (ver Seção 7). |

### Etapas E, F, G

| Decisão | Origem | Por quê |
|---|---|---|
| Teste rodado uma vez, sem ajustes depois | Reunião/plano | "Se o resultado for ruim, ele é o resultado." |
| `config_final.json` gravado antes do teste | Reunião/plano | Registro auditável do que estava congelado. |
| **Etapa G: parâmetros estimados e bandas** | **Frente** | Explica *por que* a previsão é ≈ 0 e responde à observação do professor sobre as bandas. É descritiva e não altera nenhuma escolha. |
| **Previsões fora do git** | **Frente** | Os CSVs de 1 min têm ~16 MB cada. Ficam só locais; o notebook os regenera em ~1,5 min. |

---

## 5. Resultados em detalhe

### 5.1 Grade candle × janela (validação 2024) – Q1 e Q2

| Candle | Janela | Theil U | R²_oos | Acurácia | Esperança (pts/op, operando sempre, custo 5) | Fator de lucro |
|---|---|---|---|---|---|---|
| 1 min | 1–12 m | 1,0000–1,0003 | ≈ −0,0003 | 51–52% | −4,5 a −4,7 | 0,73 |
| 15 min | 1–12 m | 1,0001–1,0014 | −0,003 a −0,0002 | 49–51% | −3,3 a −7,1 | 0,88–0,94 |
| 60 min | 1 m | 1,0257 | −0,052 | 49,6% | −10,9 | 0,90 |
| 60 min | 3 m | 1,0068 | −0,014 | 47,5% | −18,7 | 0,84 |
| **60 min** | **6 m** | **1,0016** | **−0,003** | **50,2%** | **−1,44** | **0,986** |
| 60 min | 12 m | 1,0022 | −0,005 | 47,8% | −16,8 | 0,85 |

**Como ler:**

- **Theil U** = erro do modelo ÷ erro do naive. Abaixo de 1 = melhor que prever zero. **Nenhuma das 12 combinações ficou abaixo de 1.** O **R²_oos** (quanto da variância do retorno o modelo explica fora da amostra) é negativo em todas.
- **Janela curta piora:** com 1 mês há poucos dados, os coeficientes estimados são ruidosos e o modelo "inventa" padrões (60 min × 1m: Theil U 1,026). Com mais dados, o modelo aprende que não há padrão e encolhe para zero, e o Theil U vai a 1.
- **Em 1 min o custo domina:** a esperança de ≈ −4,6 pts é quase exatamente o custo de −5. Ou seja, o ganho bruto do modelo é ≈ 0, e quem decide o resultado é o custo.
- **60 min × 6 meses** foi a escolhida (critério: esperança). Mas os vizinhos (3 m: −18,7; 12 m: −16,8) mostram que em 60 min o resultado oscila muito: são só 2.254 candles no ano. A escolha segue o protocolo, mas a diferença entre combinações é em boa parte ruído.

### 5.2 Por que a previsão é ≈ 0: os parâmetros estimados

Arquivo `parametros_estimados.csv` (modelo final, ARIMAX 60 min × 6 m, um ajuste por trimestre):

| Trimestre | φ (AR) | θ (MA) | β (WDO) | p-valor de β |
|---|---|---|---|---|
| 2024-01 | 0,29 | −0,26 | −0,12 | 0,12 |
| 2024-04 | 0,89 | −0,89 | −0,14 | **0,003** |
| 2024-07 | −0,14 | 0,09 | −0,06 | 0,32 |
| 2024-10 | 0,65 | −0,62 | 0,06 | 0,12 |
| 2025-01 | −0,65 | 0,62 | 0,01 | 0,76 |
| 2025-04 | −0,96 | 0,95 | 0,01 | 0,80 |
| 2025-07 | 0,00 | −0,04 | −0,01 | 0,87 |
| 2025-10 | −0,14 | 0,04 | −0,07 | 0,10 |
| 2026-01 | 0,56 | −0,50 | 0,02 | 0,64 |

- **φ ≈ −θ em quase todos os trimestres.** Quando isso acontece, a parte AR e a parte MA se cancelam e o ARMA(1,1) vira **ruído branco**: a melhor previsão é zero. O modelo está dizendo, com os próprios parâmetros, que não encontrou estrutura.
- **O sinal de φ troca de trimestre para trimestre** (0,89 → −0,14 → 0,65 → −0,65 → −0,96…). Se houvesse um padrão real (momentum ou reversão), o sinal seria estável.
- **β do WDO só é significativo em 1 de 9 trimestres**, e com sinal que não se mantém.

Resultado: o desvio da previsão é **0,016%** contra **0,288%** do retorno real (5,7%). Em pontos, a previsão típica é de ±10 a ±20 pts, contra movimentos reais de ±400 pts por hora.

### 5.3 Correlação WIN × WDO – por que o dólar não ajuda

| Candle | Mesmo candle: WIN(t) × WDO(t) | Defasada: WIN(t+1) × WDO(t) |
|---|---|---|
| 1 min | −0,49 | −0,00 |
| 15 min | −0,50 | 0,01 |
| 60 min | −0,47 | 0,01 |

A relação existe e é forte (dólar sobe, bolsa cai), mas é **instantânea**: os dois reagem à mesma notícia ao mesmo tempo. No candle seguinte não sobra nada. O ARIMAX só pode usar o WDO **até t**, e o WDO até *t* não diz nada sobre o WIN em *t+1*. Por isso a ARIMAX **piorou** o Theil U (1,0016 → 1,0043). Bate com o teste exploratório da Ana Beatriz.

> Se o mercado deixasse uma defasagem previsível de 1 hora entre dólar e índice, os operadores de arbitragem a eliminariam. É o que se espera de um mercado líquido.

### 5.4 Sazonalidade – o que o SARIMAX pode e não pode capturar

Autocorrelação na validação (60 min, banda de ruído ±0,041):

| Defasagem | Retorno | \|Retorno\| (volatilidade) |
|---|---|---|
| 1 candle | −0,022 | 0,090 |
| 9 candles (1 pregão, s efetivo) | −0,009 | **0,149** |
| 10 candles (s do plano) | 0,020 | 0,044 |

- **A direção não tem sazonalidade:** o retorno das 10h de hoje não diz nada sobre o das 10h de amanhã (−0,009, dentro do ruído).
- **A volatilidade tem, e forte:** um candle agitado hoje às 10h indica um candle agitado amanhã às 10h (0,149). A abertura é sempre mais volátil que a tarde.
- O SARIMA/SARIMAX modela a **média** do retorno, então não tem o que explorar (SARIMA: Theil U 1,0018, igual ao ARIMA). O padrão que existe está na **variância**, e o SARIMAX não modela a variância (ver Seção 6).

### 5.5 Regra de operação – Q3 e Q4

**Validação** (ARIMA 60 min × 6 m, 36 regras, custo 5):

- Só 2 das 36 regras ficam com esperança > 0 e fator de lucro > 1. Ambas usam limiar no quantil 90% (opera em ~10% dos candles) e stop de 1× a volatilidade.
- A melhor dá **+1,39 ± 18,1 pts/op**. O "±" é o erro-padrão: o resultado é indistinguível de zero.

**Acertar mais não é ganhar mais (Q3):**

| Regra (validação) | Taxa de acerto | Ganho médio | Perda média | Esperança |
|---|---|---|---|---|
| Opera sempre + TP 1× | **53,0%** (a maior) | 180 pts | −212 pts | **−2,40** |
| Limiar 90% + TP 2× + SL 1× | 48,7% | 235 pts | −222 pts | **+1,39** (a melhor) |

A regra que mais acerta perde dinheiro, porque o take profit curto corta os ganhos e as perdas correm. A que acerta menos de 50% tem a melhor esperança, porque os ganhos são maiores que as perdas. A esperança depende de **taxa de acerto × tamanho do ganho − taxa de erro × tamanho da perda**, não só da taxa de acerto.

**Teste** (modelo final congelado: ARIMAX, 60 min × 6 m, limiar 90%, TP 1×, sem stop):

| Série | Custo | Nº ops | Taxa de acerto | Ganho médio | Perda média | Esperança (pts/op) | Fator de lucro | Resultado |
|---|---|---|---|---|---|---|---|---|
| Modelo | 0 | 214 | 52,3% | 397 | −516 | −36,1 | 0,85 | R$ −1.547 |
| **Modelo** | **5** | **214** | **52,3%** | **392** | **−516** | **−41,1 ± 42** | **0,83** | **R$ −1.761** |
| Modelo | 10 | 214 | 52,3% | 387 | −521 | −46,1 | 0,81 | R$ −1.975 |
| Aleatório | 5 | 2.791 | 51,0% | 244 | −294 | −18,1 | 0,87 | R$ −10.079 |

- **Q2 no teste:** Theil U 1,0027, R²_oos −0,005. Não bate o naive.
- **Q3 no teste:** acerta 52,3% das operações (mais da metade) e mesmo assim perde, com payoff de 0,76. Sem stop, as perdas grandes não são cortadas (média −516 pts), e o TP 1× limita os ganhos.
- **Q4:** não é lucrativa em nenhum custo, nem com custo zero.
- **Contra o aleatório:** em R$ o modelo perde menos, mas só porque opera 13× menos. **Por operação** ele perde mais (−41 contra −18). A comparação justa é por operação.
  - **Corrigido (nov/2026):** a comparação justa é com o aleatório **nos mesmos candles**. Nos 214 candles operados pelo modelo, o aleatório dá −7,9 pts/op em média, com faixa de 90% de −68 a +55. O modelo (−41) é indistinguível do acaso; com o sinal invertido, daria +26.

---

## 6. As bandas de previsão do SARIMAX

### 6.1 O que o professor quis dizer

O professor tem razão: o SARIMAX entrega **duas** coisas para cada candle.

1. **A previsão pontual** (`predicted_mean`): o retorno esperado, que diz **quanto se espera** que suba ou desça.
2. **O intervalo de previsão** (`conf_int`): uma faixa onde o retorno real deve cair com probabilidade de 80%, 95% etc. Diz **até onde pode ir**.

O intervalo vem da variância estimada do choque (σ²) e da incerteza que o modelo acumula. Para 1 passo à frente, a banda de 95% é aproximadamente:

```
r̂ ± 1,96 · σ        →  em preço:  [ P(t)·exp((r̂ − 1,96σ)/100) ,  P(t)·exp((r̂ + 1,96σ)/100) ]
```

É isso que permite dizer, por exemplo: "na próxima hora, o WIN deve ficar entre 140.200 e 141.800 pontos, com 95% de confiança".

> **Importante:** a banda e a previsão pontual respondem a perguntas diferentes. A banda mede **risco** ("o preço pode variar ±785 pts"), não **direção** ("o preço vai subir 200 pts"). Uma banda larga e centrada em zero não diz se é para comprar ou vender.

### 6.2 O que medimos (Etapa G do notebook)

Modelo final (ARIMAX 60 min × 6 m), mesmo walk-forward do teste:

| Período | Cobertura da banda de 95% | Cobertura da banda de 80% | Meia-largura da banda de 95% |
|---|---|---|---|
| Validação 2024 | **95,0%** | 85,9% | ±0,46% ≈ ±595 pts |
| Teste 2025–26 | **92,2%** | 82,0% | ±0,49% ≈ ±785 pts |

**No total, as bandas funcionam razoavelmente:** a de 95% contém o valor real em ~92–95% dos casos. Já a previsão pontual é ~30× menor que a meia-largura da banda: o modelo diz essencialmente "deve ficar onde está, ± 785 pontos".

Veja `bandas_exemplo_semana.png`: a linha prevista é praticamente o fechamento anterior, e a faixa azul é a informação útil. Em 45 candles, 3 ficaram fora da banda.

### 6.3 O problema: a banda tem a mesma largura o dia todo

`bandas_por_hora_teste.png` / `bandas_por_hora_teste.csv`:

| Candle previsto | Desvio real do retorno | Faixa real de 95% | Banda do modelo | Cobertura real da banda de 95% |
|---|---|---|---|---|
| 10h | 0,47% | ±0,93% | ±0,51% | **74%** (estreita demais) |
| 11h | 0,39% | ±0,82% | ±0,51% | 83% |
| 12h | 0,30% | ±0,61% | ±0,51% | 91% |
| 13h–17h | 0,19–0,28% | ±0,39–0,47% | ±0,51% | 95–98% |
| 18h | 0,16% | ±0,24% | ±0,51% | **99%** (larga demais) |

O SARIMAX assume **variância constante** (um único σ² por trimestre). Mas o mercado é ~3× mais volátil na abertura que no fim do dia. Resultado: a banda é estreita demais de manhã (o preço "fura" a banda 1 em cada 4 vezes às 10h) e larga demais à tarde. A média geral (~92%) esconde esse erro.

É a mesma sazonalidade da volatilidade vista na Seção 5.4 (autocorrelação do |retorno| de 0,149 a 1 pregão de distância), agora com consequência prática.

### 6.4 Como as bandas poderiam ser úteis

- **Para gestão de risco e dimensionamento de TP/SL**, não para direção. Saber que às 10h o movimento típico é ±0,93% e às 18h é ±0,24% é útil para definir stops e o tamanho da posição.
- Para isso a banda precisa **variar com o horário e com o momento do mercado**. Caminhos possíveis (próximos passos, fora do escopo de hoje):
  - padronizar o retorno pela volatilidade típica de cada horário antes do SARIMAX e "desfazer" depois;
  - um modelo de volatilidade como **GARCH** (variância condicional), eventualmente com componente intradiária.

### 6.5 O cone de previsão: vários passos à frente

Notebook: [`frentes/estatistico/cone_previsao_arima.ipynb`](../../frentes/estatistico/cone_previsao_arima.ipynb) · Gráficos: `cone_arima_diario.png`, `cone_arima_60min.png`.

As bandas acima são de **1 passo** (o próximo candle). Quando o modelo projeta **h passos**, a incerteza se acumula e a banda se abre em forma de cone. Esse é o tipo de gráfico que mostra "qual o range que o preço pode alcançar".

- **Por que abre em √h:** o preço daqui a h candles = preço atual + soma de h retornos. Cada retorno tem um choque de variância σ², então a variância da soma é h·σ² e a meia-largura da banda é z·σ·√h.
- **Como foi feito:** ARIMA(p,1,q) no log-preço (d = 1 equivale a modelar o retorno), com p, q ∈ {0,1,2} e com ou sem deriva, escolhido por **AIC** numa janela anterior à avaliada. As bandas de 1 passo foram avaliadas fora da amostra. Esse notebook é um exemplo ilustrativo, **fora do protocolo** de comparação, e usa os dados do WIN até 25/09/2026.

| | Diário (cone de 20 pregões) | 60 min (cone de 20 candles ≈ 2 pregões) |
|---|---|---|
| Modelo escolhido pelo AIC | **ARIMA(0,1,0)**, sem deriva | **ARIMA(0,1,0)**, sem deriva |
| σ por candle | 1,11% | 0,39% |
| Previsão central | 0,00% (= último preço, 184.700) | 0,00% (= último preço, 184.700) |
| Banda de 80% no horizonte | 173.361 a 196.781 (±6,3%) | 180.628 a 188.864 (±2,2%) |
| RMSE / sem mudança | 1,000 | 1,000 |
| Cobertura de 1 passo, 80% / 95% | 74% / 90% (n 250) | 87% / 94% (n 500) |
| Ljung-Box (resíduos) | p = 0,091 (ruído branco) | p = 0,347 (ruído branco) |

**Leitura:**

- **O AIC escolheu o passeio aleatório.** O ARIMA "ótimo" é o próprio naive do projeto (prever retorno zero). É por isso que o centro do cone é uma reta no último preço e o Theil U da grade fica em 1. O exemplo da LuxAlgo com o SPY chega ao mesmo resultado: ARIMA(0,1,0) e "no edge".
- **O cone não diz a direção, mas quantifica o risco:** "com 80% de probabilidade, o WIN fica entre 173 e 197 mil nos próximos 20 pregões".
- **A calibração não é perfeita:**
  - **no diário** as bandas são estreitas demais (74% em vez de 80%), porque a volatilidade muda de um mês para o outro e o σ é fixo;
  - **em 60 min** a de 80% é larga demais (87%) e a de 95% é levemente estreita (94%). É a assinatura de **caudas gordas**, somada à volatilidade intradiária da Seção 6.3.
- **Em 60 min o cone ignora a noite:** o gap de abertura e a abertura mais volátil não entram no cálculo.

---

## 7. Por que a ARIMAX foi escolhida e não funcionou no teste

| Modelo (60 min × 6 m) | Theil U (validação) | Esperança com a melhor regra (validação) |
|---|---|---|
| ARIMA (base) | 1,0016 | +1,39 ± 18,1 |
| **ARIMAX** | 1,0043 | **+10,22 ± 18,5** |
| SARIMA s=9 | 1,0018 | −0,72 ± 6,0 |
| SARIMAX s=10 (plano) | 1,0046 | +5,93 ± 19,6 |
| SARIMAX s=9 (efetivo) | 1,0044 | +3,77 ± 12,7 |

- A regra fixada antes do teste mandava escolher a maior esperança, então a ARIMAX foi escolhida. **Isso foi seguir o protocolo, não um erro.**
- Mas o **erro-padrão** (±18,5) é quase o dobro da vantagem (+10,2). A diferença para o ARIMA (+1,4) não é estatisticamente distinguível, e a ARIMAX tem Theil U **pior**.
- Na validação testamos 5 modelos × 36 regras = **180 combinações**. Mesmo que nenhuma tivesse vantagem real, a melhor delas pareceria lucrativa por sorte. Esse é o **viés de seleção** (*winner's curse*): o vencedor de uma disputa entre ruídos tende a decepcionar fora da amostra.
- No teste, −41 ± 42 pts/op: também compatível com "vantagem nula". O resultado da validação **não se repetiu**, como esperado pelo erro-padrão e registrado no notebook **antes** de rodar o teste.

**Lição para a apresentação:** reportar a esperança sempre com o erro-padrão (ou o número de operações). Com ~200 operações e desvio de ~250 pts por operação, só uma esperança acima de ~35 pts/op seria distinguível de zero.

---

## 8. Limitações

| Limitação | Impacto |
|---|---|
| Ordem (1,0,1) fixa | Não testamos outras ordens (AIC/BIC). Dada a autocorrelação ≈ 0 em todas as defasagens, é improvável que outra ordem mude a conclusão. |
| Variância constante | As bandas não acompanham a volatilidade intradiária (Seção 6.3). |
| Parâmetros fixos dentro do trimestre | Se o mercado mudar no meio do trimestre, o modelo só se ajusta no próximo re-treino. |
| Simulação com 1 contrato e custo fixo | Não considera liquidez, slippage variável nem o horário. |
| TP/SL no mesmo candle → stop | Hipótese conservadora; pode subestimar levemente o resultado. |
| Teste com ~210 operações | A esperança tem erro-padrão de ~42 pts; não dá para afirmar nem que o modelo perde nem que ganha com precisão. |
| WDO termina em 01/04/2026 | Coincide com o fim do teste; sem impacto. |
| **Buraco nos dados de jun–out/2023** | O CSV do WIN não tem pregões entre o início de junho e 17/10/2023 (o WDO também falha nesse período). As janelas de treino que cobrem esse trecho ficam menores: em 60 min × 6 m, o treino do 1º trimestre de 2024 teve 429 candles, contra ~1.100 nos demais. Afeta igualmente as três frentes (mesmos dados), mas deixa os primeiros trimestres da validação menos confiáveis, sobretudo nas janelas de 6 e 12 meses. |

---

## 9. Para a apresentação

### Três mensagens

1. **O ARIMA não prevê o retorno do WIN no T+1.** Não bate o naive em nenhum candle ou janela, e os próprios parâmetros (φ ≈ −θ) mostram que ele não encontrou estrutura.
2. **O dólar e a sazonalidade não ajudam na direção.** O WDO reage junto com o WIN, não antes (correlação −0,49 no mesmo candle e 0 com defasagem). A sazonalidade intradiária existe, mas na volatilidade, não na direção.
3. **As bandas do SARIMAX medem risco, e aí há algo aproveitável.** Elas acertam ~92–95% no total, mas falham por horário, porque a volatilidade da abertura é 3× a do fim do dia. Esse é o caminho para um próximo trabalho: modelar a volatilidade para dimensionar TP/SL.

### Perguntas prováveis do professor

| Pergunta | Resposta curta |
|---|---|
| "Vocês preveem retorno ou preço?" | Retorno (log, em %), e o preço sai por P̂(t+1) = P(t)·exp(r̂/100). ARIMA(1,0,1) no retorno = ARIMA(1,1,1) no log-preço. |
| "Por que d = 0?" | O retorno já é a primeira diferença do log-preço; ele é estacionário. |
| "E as bandas?" | Calculamos: ±0,49% (≈ ±785 pts) por hora, cobertura de 92% no teste. Mas a largura é constante e a volatilidade não, por isso ela cobre só 74% às 10h e 99% às 18h. |
| "Dá para mostrar até onde o preço pode ir?" | Sim, com o cone de previsão (`cone_arima_diario.png`): ±6,3% em 20 pregões com 80% de probabilidade. O AIC escolheu ARIMA(0,1,0), então o centro é plano; a informação está na largura. |
| "Por que a ARIMAX foi escolhida se é pior?" | Pelo protocolo (maior esperança na validação). A vantagem era menor que o erro-padrão, e o teste confirmou que era ruído. |
| "Por que s = 9 e não 10?" | O último candle do dia é descartado (não tem alvo); cada pregão tem 9 linhas na série. A autocorrelação da volatilidade confirma (0,149 em 9 contra 0,044 em 10). |
| "Por que o modelo acerta 52% e perde?" | Ganho médio de 392 pts contra perda média de 516: sem stop, as perdas correm. Acurácia não é lucro. |
| "O resultado ruim significa que o trabalho falhou?" | Não: é um resultado. Consistente com um mercado eficiente nessa escala, e obtido com protocolo limpo (sem olhar o teste). |

### Slides sugeridos (2 da frente)

1. **Heatmaps** (`heatmap_theilU.png`, `heatmap_esperanca.png`) + `correlacao_win_wdo.png`. Conclusão: "nenhuma combinação bate o naive; o dólar reage junto, não antes".
2. **`curva_capital_teste.png` + tabela do teste + `bandas_por_hora_teste.png`.** Conclusão: "sem vantagem na direção; a informação útil do SARIMAX é a banda de risco, que precisa variar com o horário".

---

## 10. Arquivos gerados

Em `resultados/estatistico/`:

| Arquivo | Conteúdo |
|---|---|
| `grade_val.csv` | 12 combinações candle × janela: métricas e simulação |
| `heatmap_theilU.png`, `heatmap_acuracia.png`, `heatmap_esperanca.png` | Heatmaps da grade |
| `regras_val.csv` | 36 regras do ARIMA base (60 min × 6 m) |
| `regras_val_arimax_60min_6m.csv` | 36 regras do modelo final (ARIMAX) |
| `variante_val.csv` | ARIMA, ARIMAX, SARIMA, SARIMAX (s=10 e s=9), com o erro-padrão da esperança |
| `correlacao_win_wdo.png` | Correlação contemporânea × defasada |
| `config_final.json` | Escolhas congeladas antes do teste |
| `teste_final.csv` | Modelo × aleatório, custos 0/5/10 |
| `curva_capital_teste.png`, `previsto_vs_real_teste.png` | Gráficos do teste |
| `parametros_estimados.csv` | φ, θ, β, σ² e p-valores por trimestre |
| `bandas_por_hora_teste.csv`, `bandas_por_hora_teste.png` | Cobertura das bandas por horário |
| `bandas_exemplo_semana.png` | Exemplo em preço: previsão, banda e fechamento real |
| `cone_arima_diario.png`, `cone_arima_60min.png` | Cone de previsão multi-passos com painel de diagnóstico (notebook `cone_previsao_arima.ipynb`) |
| `resumo.md` | 5 linhas no formato do plano |
| `previsoes/` | CSVs de previsão (**só locais**, fora do git; regenerados pelo notebook) |
