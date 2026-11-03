# Relatório da frente de redes neurais – MLP / GRU / LSTM

> Responsável: Gabriel Modesto · Notebook: [`frentes/redes/frente_redes.ipynb`](../../frentes/redes/frente_redes.ipynb) · Resumo de 5 linhas: [`resumo.md`](resumo.md) · Plano: [`planos/PLANO_FRENTES_02.11.md`](../../planos/PLANO_FRENTES_02.11.md)

O `resumo.md` segue o formato fixo do plano. Aqui está o raciocínio: o que foi rodado, por que cada escolha saiu como saiu e como o resultado se compara com as frentes estatística e de ML.

---

## 1. Em uma frase

As redes neurais **não preveem o retorno do próximo candle do WIN melhor que o naive** e a regra de operação escolhida na validação **perde dinheiro no teste** (−20,4 pts/op com custo de 5 pts). O MLP sobreajusta quando a janela de treino é curta; a GRU e a LSTM evitam isso encolhendo a previsão para perto de zero, ou seja, aprendendo a ser quase o naive.

---

## 2. O que foi rodado

| Item | Escolha |
|---|---|
| Ativo | Só **WIN**. Nenhuma feature do mini dólar; o arquivo do WDO é lido só porque o `montar_base` faz o join |
| Modelo base | MLP `(32, 16)`, `alpha = 1e-3`, *early stopping*, padronização, features `COMPLETO` (Seção 3.3) |
| Grade | **12 rodadas**: candle 1/15/60 min × janela 1/3/6/12 meses, re-treino trimestral, validação 2024 |
| Regra | 36 regras (limiar × TP × SL), custo 5 pts, escolhida pela **maior esperança** (critério do plano comum) |
| Variante | GRU e LSTM (16 unidades) nos últimos 10 retornos (`LAGS`); MLP com `alpha = 0,1` (Seção 8) |
| Teste | 01/2025 → 03/2026, rodado com tudo congelado; a execução foi repetida e deu resultado idêntico (sementes fixas e operações determinísticas no TensorFlow) |
| Extra (só leitura) | Erro-padrão e *t* de cada esperança, Clark-West contra o naive e as 12 combinações no teste (Etapa G) |

O tempo não foi problema: a rodada mais pesada (1 min × 12 meses) levou 14 s; o notebook inteiro roda em cerca de 3 minutos numa máquina de 2 núcleos.

---

## 3. Resultados

### 3.1 Grade candle × janela (validação 2024) – Q1 e Q2

| Candle | Theil U (1 / 3 / 6 / 12 m) | Esperança operando sempre, pts/op (1 / 3 / 6 / 12 m) |
|---|---|---|
| 1 min | 1,111 / 1,033 / 1,013 / 1,004 | −5,0 / −5,0 / −5,1 / −5,1 |
| 15 min | 1,704 / 1,256 / 1,155 / 1,085 | −4,3 / −4,8 / −6,7 / −7,4 |
| 60 min | 1,748 / 1,356 / 1,248 / 1,140 | **+2,5** / −11,0 / −21,1 / −12,2 |

Três leituras:

1. **Nenhuma das 12 combinações bate o naive** (Theil U < 1 em 0 de 12).
2. **Quanto menos dados, pior o erro.** O Theil U cai de forma monotônica com a janela e é melhor no candle de 1 min. Em 60 min × 1 mês a rede treina com 171–188 candles por trimestre e passa a prever retornos **maiores que os reais** (desvio da previsão 145% do real): é sobreajuste puro. Em 1 min × 12 meses o desvio da previsão cai para 9% do real e o Theil U chega a 1,004: a rede só se aproxima do naive quando aprende a prever quase zero.
3. **A combinação escolhida é justamente a pior pelo erro.** O critério do plano é a esperança, e a única combinação com esperança positiva foi 60 min × 1 mês (+2,5 pts/op, t = 0,40). Isso é o mesmo padrão que a frente de ML encontrou na primeira rodada: a janela mais curta gera previsões ruidosas que, por acaso, acertam a direção em alguns candles grandes.

### 3.2 Regra e variante (validação) – Q3 e Q4

| Modelo (60 min × 1 m) | Theil U | Desvio prev./real | Melhor regra | Nº ops | Esperança (pts/op) | t |
|---|---|---|---|---|---|---|
| **LSTM** (escolhida) | 1,0005 | 8% | q90, TP 1×, SL 1× | 226 | **+28,2** | 1,45 |
| MLP `alpha = 0,1` | 1,745 | 145% | q90, TP 2× | 226 | +23,1 | 1,22 |
| MLP base | 1,748 | 145% | q90, TP 2× | 226 | +21,5 | 1,14 |
| GRU | 1,008 | 11% | q50, TP 2×, SL 2× | 1.127 | +11,0 | 1,24 |

- Aumentar o `alpha` 100× quase não mudou o MLP: com ~180 linhas de treino, o *early stopping* para cedo e a regularização L2 não é o que limita.
- A **LSTM** venceu e virou o modelo da frente. Ela tem o melhor Theil U da tabela, mas porque prevê quase zero.
- **Nenhuma esperança é estatisticamente diferente de zero** (todas com t < 2, erro-padrão de ~19 pts/op com 226 operações). A diferença entre a LSTM e o MLP (+28 contra +21) é bem menor que o erro-padrão. Pelo critério do plano a escolha está correta, mas a validação não dá evidência de lucro.

### 3.3 Teste (01/2025 → 03/2026) – resultado oficial

| Custo | Nº ops | Taxa de acerto | Payoff | Esperança (pts/op) | t | Fator de lucro | Resultado R$ |
|---|---|---|---|---|---|---|---|
| 0 pts | 445 | 47,9% | 0,99 | −15,4 | −0,73 | 0,917 | −1.374 |
| **5 pts** | 445 | 47,9% | 0,97 | **−20,4** | −0,96 | 0,892 | **−1.819** |
| 10 pts | 445 | 46,7% | 0,97 | −25,4 | −1,20 | 0,867 | −2.264 |

- **Q2:** Theil U = 1,0022, R²_oos = −0,0044, Clark-West t = 0,32: **não bate o naive**.
- **Q3:** a acurácia direcional ficou em 50,6% e a taxa de acerto das operações em 47,9%. Na grade de validação, a correlação de Spearman entre acurácia e esperança foi 0,67: acertar a direção ajuda, mas com acurácia de ~50% não sobra vantagem para explorar.
- **Q4:** a regra perde até **sem custo** (−15,4 pts/op): não existe ganho bruto que o custo esteja comendo; o modelo simplesmente não acerta.
- **Contra o aleatório:** os sinais aleatórios com a mesma regra perdem −29,9 pts/op (t = −4,9) e R$ 16.715. A rede perde menos em R$ porque opera 6× menos (445 contra 2.791 operações), não porque tenha vantagem: a diferença por operação (−20,4 contra −29,9) está dentro do erro-padrão da rede (21 pts).
- **Mais operações que na validação** (445 contra 226): o limiar foi congelado em % (0,028%), e a volatilidade de 2025 foi ~23% maior que a de 2024, então mais previsões passaram do limiar. É o comportamento esperado de um limiar fixo, não um erro.

### 3.4 O ranking da validação se repete no teste? (Etapa G, só leitura)

| Métrica | Spearman validação × teste (12 combinações) |
|---|---|
| Theil U | **0,97** |
| Esperança (operando sempre) | **0,17** |

O mesmo achado da frente de ML (0,99 e −0,24): **o erro de previsão é uma propriedade estável** do par candle × janela, mas **o lucro na simulação não é**. Escolher pela esperança da validação é, na prática, escolher por sorte.

Uma ressalva honesta: no teste, o **MLP base** em 60 min × 1 mês operando sempre teria dado +11,9 pts/op. Isso não muda a conclusão: com custo de 5 pts o t é 1,43 (não significativo), é a melhor de 12 combinações olhada depois do fato, e com a regra que ele teria levado da validação (q90, TP 2×) cai para +2,4 pts/op (t = 0,07). Por trimestre, oscila entre +37 e −21 pts/op.

---

## 4. Comparação com as outras frentes (teste, custo de 5 pts)

| Frente | Modelo · candle · janela | Theil U | R²_oos | Clark-West t | Acurácia | Nº ops | Taxa de acerto | Esperança (pts/op) | Fator de lucro | Resultado R$ | Aleatório (pts/op) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive | retorno = 0 | 1,000 | 0 | – | – | – | – | – | – | – | – |
| Estatístico | ARIMAX · 60 min · 6 m | 1,0027 | −0,0053 | – | 49,2% | 214 | 52,3% | −41,1 | 0,833 | −1.761 | −18,1 |
| ML | Random Forest · 1 min · 12 m | **0,9999** | **+0,0002** | **3,72** | 51,4% | 174.579 | 42,6% | **−4,4** | 0,805 | −153.992 | −4,8 |
| Redes | LSTM · 60 min · 1 m | 1,0022 | −0,0044 | 0,32 | 50,6% | 445 | 47,9% | −20,4 | **0,892** | −1.819 | −29,9 |

O "aleatório" de cada frente usa a mesma regra (TP/SL) da frente, por isso difere entre elas.

Leituras para a apresentação:

1. **Nenhuma das três famílias lucra no teste.** Todas têm esperança negativa com 5 pts de custo; ARIMAX e LSTM perdem mesmo com custo zero.
2. **Só o Random Forest bate o naive com significância** (Clark-West t = 3,72), e só no candle de 1 min com a janela mais longa. O ganho bruto dele (+0,59 pt/op) é ~8× menor que o custo.
3. **As três frentes chegaram à mesma lição sobre a escolha:** o Theil U se repete entre validação e teste; a esperança, não. As duas frentes que escolheram 60 min pela esperança (estatístico e redes) caíram em combinações com poucos dados e esperança positiva por ruído (ARIMAX +10,2 ± 18,5; LSTM +28,2 ± 19,4), que não se repetiu.
4. **Candle e janela:** pelo erro de previsão, as três frentes concordam que o **candle de 1 min** é onde o Theil U fica mais perto de 1 e o de 60 min é o pior; no ML e nas redes, a **janela mais longa** também ajuda sempre (no ARIMA, com só 2 parâmetros, a janela quase não importa). É onde há mais dados por parâmetro. Pela esperança, cada frente escolheu um lugar diferente – o que, por si só, já indica ruído.
5. **Rede neural não é vantagem aqui.** Com features de preço e volume do próprio WIN, a capacidade extra do MLP vira sobreajuste e a das recorrentes vira "prever zero". O limite está na informação disponível, não na flexibilidade do modelo.

---

## 5. Limitações

| Limitação | Efeito |
|---|---|
| Escolha pela maior esperança sem exigir significância (critério do plano) | Levou a 60 min × 1 m, a combinação mais ruidosa. Com o critério da frente de ML (t ≥ 2), nenhuma regra passaria e a escolha cairia no menor Theil U (1 min × 12 m) |
| Arquitetura fixa e pequena (sem busca de hiperparâmetros) | Escolha deliberada do plano para caber no prazo; a grade de variantes é pequena |
| Sequência das recorrentes atravessa a virada do dia | Aceito no plano (Seção 3.3); o 1º candle do dia não carrega o gap noturno |
| GRU/LSTM com 10 épocas e ~180 linhas de treino por trimestre | Pouco dado para uma rede recorrente aprender algo além da média |
| Custo de 5 pts não confirmado com a corretora | Não muda a conclusão: a LSTM perde com custo zero |

---

## 6. Arquivos gerados (`resultados/redes/`)

| Arquivo | Conteúdo |
|---|---|
| `grade_val.csv`, `heatmap_*.png` | 12 rodadas do MLP na validação |
| `regras_val.csv` | 36 regras do MLP base em 60 min × 1 m |
| `variante_val.csv` | MLP, MLP `alpha = 0,1`, GRU e LSTM com a melhor regra de cada |
| `config_final.json` | LSTM, 60 min, 1 mês, q90 (0,0282%), TP 1×, SL 1× |
| `teste_final.csv`, `curva_capital_teste.png`, `previsto_vs_real_teste.png` | Teste oficial |
| `grade_teste_complementar.csv`, `validacao_vs_teste.png` | As 12 combinações no teste (não altera escolhas) |
| `previsoes/` | Previsões de 15 e 60 min (as de 1 min ficam fora do git pelo `.gitignore`; são recriadas pelo notebook) |
