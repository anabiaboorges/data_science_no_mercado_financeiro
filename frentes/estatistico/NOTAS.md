# Notas da reabordagem – frente estatística

> Documento vivo: achados, cuidados e pendências da reabordagem iniciada em 04/11/2026. Atualizar a cada estudo.
> Material relacionado: [estudos da reabordagem (índice)](estudos/README.md) · [modelo de custo `custos.py`](custos.py) · [relatório original da frente](../../resultados/estatistico/relatorio.md) · [EDA do grupo](../../eda/analise_exploratoria_WIN_WDO.ipynb)

## Sumário

0. [Resumo](#resumo)
1. [Objetivo atual](#1-objetivo-atual)
2. [Dados](#2-dados)
3. [Divisão dos dados (proposta)](#3-divisão-dos-dados-proposta)
4. [Achados](#4-achados)
5. [Cuidados permanentes](#5-cuidados-permanentes)
6. [Pendências](#6-pendências)
7. [Para estudar](#7-para-estudar)

---

## Resumo

**Pergunta do grupo:** qual a melhor abordagem para ter ganhos reais usando previsão T+1?

**O que a frente estatística encontrou (04–05/11/2026):**

1. **Direção do próximo candle: nenhuma vantagem sobreviveu fora da amostra.** Seis hipóteses exploradas com dados até 2024 (Estudos 02–07) geraram dois candidatos, pré-registrados e testados uma vez em 01/2025–03/2026 (Estudo 08). Nenhum se confirmou. O mais forte, momentum no fechamento, foi de +29 ± 8 para −7 ± 15 pts por operação.
2. **Tamanho dos movimentos (volatilidade): muito previsível.** O HAR explica 47–90% da volatilidade diária fora da amostra; bandas por hora calibradas com perfil intradiário recente × HAR (Estudo 07). Serve para stops, bandas e tamanho da posição. Não dá lucro sozinho, porque não diz a direção.
3. **Custo:** os 5 pts eram um exemplo do plano. Modelo derivado: 2,5 / 7,5 / 12,5 pts por cenário de execução, e o **custo de equilíbrio** de cada estratégia (Seção 4.7, `custos.py`).
4. **Avaliação:** vários resultados "significativos" eram artefatos (o VR da EDA, a régua 2/√n, a comparação com o aleatório). Os cuidados que evitam isso estão na Seção 5.
5. **Ideias do professor (Estudo 09):** altas e baixas são simétricas na direção e assimétricas na volatilidade; o consenso de modelos amplifica habilidade, não cria; para direção, medir correlação (IC) e não MSE, e em séries persistentes medir nas mudanças; regras de saída mudam o acerto, não a esperança.

**Estado:** o teste final (abr–set/2026) continua intocado. As decisões que dependem do grupo e as opções para a próxima sessão estão na Seção 6 ("Retomar daqui").

---

## 1. Objetivo atual

A apresentação de 03/11 não aconteceu. O grupo trocou a pergunta:

> **Qual a melhor abordagem para ter ganhos reais usando previsão T+1 em séries temporais?**

- As frentes **não** precisam mais usar a mesma grade (candles de 1/15/60 min, janelas de 1/3/6/12 meses). Cada uma escolhe o que faz sentido para o seu modelo. Exemplo: o ARIMA pode testar qualquer tamanho de candle, a rede neural não.
- Se o professor exigir comparação direta entre modelos, voltamos ao protocolo comum.
- **O que continua igual para todos** (sem isso, comparar retornos não faz sentido):

| Item | Por quê |
|---|---|
| Mesmo simulador (`tt1_comum.simular`) e mesmas hipóteses de execução | Senão comparamos o otimismo de cada um, não as estratégias |
| Mesmo modelo de custo e de execução (proposta: custo de equilíbrio + cenários 2,5 / 7,5 / 12,5 pts; Seção 4.7) | O custo decide o resultado em candles pequenos |
| Mesmo período final de teste, usado **uma vez** | Única defesa contra o viés de seleção (Seção 4.3) |
| Esperança por operação **com erro-padrão** (ou t) e nº de operações | Resultado total em R$ esconde se a diferença é ruído |

---

## 2. Dados

### 2.1 O que temos

Exportação do MetaTrader feita pelo professor, candles de 1 min:

| Série | Período | Observação |
|---|---|---|
| `WIN$N` (mini índice, contínuo) | 15/10/2020 → 25/09/2026 | 757 mil candles |
| `WDO$N` (mini dólar, contínuo) | 17/09/2020 → **01/04/2026** | **Acaba antes do WIN** |

**Qualidade no nível do candle:** sem timestamps duplicados, sem candles impossíveis (máxima < mínima), sem volume zero. Quase todo pregão tem os ~565 minutos completos.

### 2.2 Problemas e particularidades

| Item | Detalhe | Consequência |
|---|---|---|
| **Buracos** | 15/11–01/12/2021 · 15–27/07/2022 · **05/06–16/10/2023** (96 pregões). No WDO, os mesmos períodos com datas um pouco diferentes | Nunca calcular retorno "por cima" de um buraco. Janelas de treino que cobrem 2023 ficam menores |
| **Rolagem do WIN** | A série troca de contrato na abertura do dia de vencimento (quarta-feira mais próxima do dia 15, meses pares). O gap médio nesse dia é de **+1,66%**, contra ±0,55% num dia normal. É o prêmio de juros (*carry*) do contrato 2 meses mais longo, por isso acompanha a Selic: 0,1–0,5% em 2021, 2–2,5% em 2025–26 | **Os retornos intradiários não são afetados**: o `montar_base` descarta o gap noturno. Qualquer análise diária (fechamento → fechamento) ou do gap de abertura **precisa tirar ou ajustar esses dias** |
| **Rolagem do WDO** | O salto aparece no **último** pregão do mês (+0,34% em média, contra ≈ 0 nos outros dias). A EDA do grupo diz "primeiro dia útil": está errado | Mesmo cuidado em análises diárias do WDO |
| **WDO termina em 01/04/2026** | Não cobre o período de teste final proposto (abr–set/2026) | Modelo com WDO **não pode** ser avaliado no teste final |
| **Horário do pregão** | Até 2023: fim às 18:24 de nov a mar e às 17:54 de mar a nov, acompanhando o horário de verão dos EUA. **Desde nov/2023: sempre 18:24** | Afeta features de horário (`hora`, `candle_do_dia`), o nº de candles por dia (`s`) e comparações por horário |
| Dias atípicos | Quarta-feira de Cinzas abre às 13:00 (325 min). 17/10/2023 é parcial (57 min), logo depois do buraco | Excluir ou tratar à parte em análises por horário |
| Coluna `SPREAD` | No WDO muda de escala em 2023 (500 → 1) | Não usar |

### 2.3 Fontes de dados novos (investigado em 04/11/2026)

| Opção | Situação |
|---|---|
| Professor | **Melhor caminho, mas indisponível agora.** Quando der: (1) exportar de 25/09/2026 em diante; (2) tentar preencher jun–out/2023 rolando o gráfico de 1 min até 2023 ou aumentando *Ferramentas → Opções → Gráficos → Máx. barras no gráfico* (o MT5 baixa o histórico sob demanda) |
| Conta em corretora / MT5 próprio | Não temos. A conta de um colega no BTG foi recusada |
| Backtester.com.br | WIN/WDO contínuos só no plano pago (B3PRO) |
| B3, arquivos tick a tick | Grátis, mas só os últimos ~20 pregões. Serve para montar um teste futuro, coletando todo dia (ex.: projeto `b3-data-collector`). Não preenche 2023 |
| NegocIA, Cedro | Pagos |

---

## 3. Divisão dos dados (proposta)

**Pendente: o grupo precisa concordar.**

| Período | Situação | Uso |
|---|---|---|
| 10/2020 → 12/2024 | Já usado para escolhas | **Desenvolvimento:** EDA, ideias, escolha de modelos e regras |
| 01/2025 → 03/2026 | Resultados já vistos por todos | **Segunda validação:** confirmar o que sair do desenvolvimento ("visto, mas não ajustado nele") |
| 04/2026 → 25/09/2026 | A frente de ML rodou um modelo congelado uma vez | **Teste final:** rodar uma vez no fim, sem escolher nada nele |

- O teste final tem só ~6 meses (≈ 1.100 candles de 60 min). Ele confirma um efeito grande, mas não detecta um pequeno. Por isso, levar **poucos** candidatos até ele.
- Sem WDO no teste final (Seção 2.2).
- Toda EDA nova usa **só dados até 2024**.

---

## 4. Achados

### 4.1 Frente estatística original (protocolo de 02/11)

Detalhes no [relatório](../../resultados/estatistico/relatorio.md).

- **O ARIMA(1,0,1) não bate o naive** (prever retorno 0) em nenhum candle ou janela: Theil U ≥ 1 nas 12 combinações.
- **φ ≈ −θ em quase todo trimestre:** as partes AR e MA se cancelam e o modelo vira ruído branco. O sinal de φ troca de trimestre para trimestre.
- **O WDO não antecipa o WIN:** correlação de −0,47 a −0,50 no mesmo candle e ≈ 0 com 1 candle de defasagem.
- **A sazonalidade intradiária está na volatilidade, não na direção:** autocorrelação do |retorno| de 0,149 a um pregão de distância, contra −0,009 do retorno.
- **As bandas de 95% do SARIMAX** cobrem 92% no total, mas 74% às 10h e 99% às 18h. A variância é constante no modelo e não no mercado.
- **O modelo final (ARIMAX)** venceu a validação por ruído de seleção: +10 ± 18 pts/op entre 180 combinações. No teste deu −41 ± 42 pts/op.

### 4.2 Reavaliação do teste (04/11/2026)

- **A comparação com o aleatório estava injusta.** O modelo operou 214 candles escolhidos, o aleatório operou todos os 2.791.
- **O limiar funciona como filtro de volatilidade:** o |previsão| tem correlação de 0,35 com a volatilidade, e a volatilidade mediana dos candles operados é 0,36%, contra 0,25% no total.
- **Aleatório nos mesmos 214 candles, com a mesma regra** (2.000 sorteios): média de −7,9 pts/op, faixa de 90% de [−68; +55]. O modelo (−41) fica no percentil 19: **indistinguível de jogar moeda.**
- **Com o sinal do modelo invertido**, a esperança vira **+26** pts/op. O sinal do resultado é ruído.
- Portanto, a frase "o modelo perde mais por operação que o aleatório" (−41 contra −18) **não vale**.

### 4.3 Estudo 01: ruído, autocorrelação e custo

Notebook: [estudos/01_ruido_e_autocorrelacao.ipynb](estudos/01_ruido_e_autocorrelacao.ipynb).

**Conceitos verificados por simulação:**
- O ρ₁ medido em ruído puro varia ≈ 1/√n; 95% dos valores ficam em ±2/√n.
- **Caudas gordas sozinhas não alargam essa faixa; o agrupamento de volatilidade alarga.** Num GARCH, só 84% ficam dentro de ±2/√n. Com o **erro-padrão robusto**, volta a 95%.
- **Comparações múltiplas:**
  - 12 testes em ruído dão pelo menos 1 "significativo" em ~45% das vezes.
  - A melhor de 180 estratégias **sem vantagem nenhuma** aparece com ≈ **+40 pts/op** (200 operações, desvio de 250 pts por operação, custo de 7,5 pts).
- **Da correlação ao dinheiro:**
  - acurácia ≈ 50% + arcsen(ρ)/π;
  - ganho bruto por operação (= **custo de equilíbrio**) ≈ ρ × movimento médio absoluto (M);
  - para pagar o custo c, é preciso **ρ\* = c / M**.

**No WIN, 2020–2024, autocorrelação de defasagem 1 por tamanho de candle:**

| Candle (min) | 1 | 3 | 10 | 30 | 60 | 120 |
|---|---|---|---|---|---|---|
| ρ₁ medido | −0,004 | −0,011 | −0,014 | −0,001 | −0,010 | +0,007 |
| Faixa robusta (±) | 0,008 | 0,012 | 0,017 | 0,022 | 0,034 | 0,045 |
| Custo de equilíbrio \|ρ₁\|·M (pts) | 0,1 | 0,7 | 1,7 | 0,1 | 3,0 | 2,7 |
| ρ necessário, otimista (2,5 pts) | 0,067 | 0,038 | 0,021 | 0,012 | 0,009 | 0,007 |
| ρ necessário, base (7,5 pts) | 0,200 | 0,115 | 0,063 | 0,037 | 0,026 | 0,019 |

- **Com a faixa robusta, nenhum tamanho de candle tem ρ₁ significativo.** Com a régua clássica, 1, 2, 3 e 10 min pareciam ter.
- **Cenário base, até 45 min:** o ρ necessário fica acima do medido e da faixa de ruído. Mesmo um efeito real não pagaria o custo: até 30 min, o custo de equilíbrio é de 0,1 a 1,7 pt, abaixo até do otimista.
- **Onde o ρ necessário cai dentro da faixa de ruído** (60–120 min no base; a partir de 15 min no otimista): não há dados suficientes para saber.
- Em 45–120 min o custo de equilíbrio estimado (≈ 3 pts) passa do otimista, mas esses ρ₁ não são significativos e trocam de sinal: é ruído. **Ler sempre o custo de equilíbrio junto com o erro dele.**
- **Conclusão:** procurar o "melhor tamanho de candle" para o ARIMA é procurar onde não há nada. Se existe algo explorável, ele é **condicional** (horário, depois de grandes movimentos, gap de abertura, regime de volatilidade) ou está na **volatilidade**.

### 4.4 EDA do grupo (Ana Beatriz)

Já cobre: qualidade, buracos, caudas pesadas, agrupamento de volatilidade, perfil intradiário em "L", volume × movimento, ADF, WIN × WDO (correlação de −0,2 a −0,8 ao longo do tempo, lead-lag ≈ 0).

A EDA reporta **razão de variâncias VR(60) = 0,89 no WIN** (0,76 no WDO), lida como reversão à média intradiária. **O Estudo 02 mostrou que é, em grande parte, artefato do cálculo** (Seção 4.6).

Limitações:
- usa o período todo, inclusive o teste;
- a data de rolagem do WDO está errada (Seção 2.2);
- o cálculo da VR é enviesado para baixo (Seção 4.6).

### 4.5 Inconsistência entre frentes

- A frente de ML só aceita uma escolha se tiver t ≥ 2; senão usa o padrão. Também reporta o teste de Clark-West contra o naive.
- A frente estatística escolheu a ARIMAX com t = 0,55.
- Convém adotar o mesmo critério.

### 4.6 Estudo 02 – H1: reversão intradiária

Notebook: [estudos/02_h1_reversao_intradiaria.ipynb](estudos/02_h1_reversao_intradiaria.ipynb). Dados até 2024.

**O VR(60) da EDA é, em boa parte, artefato.**
- A primeira hora do dia (a mais volátil) sai do numerador mas não do denominador.
- O candle "de 60 min" das 18h tem só 24 minutos.
- Prova: um passeio aleatório simulado **sem reversão nenhuma**, com o perfil de volatilidade do WIN, dá VR(60) = 0,94 com o mesmo cálculo. Nos dados reais até 2024 dá 0,92. Ou seja, ~3/4 da queda é artefato.

**Medida direito** (pelas autocorrelações, só pares dentro do pregão, teste robusto de Lo-MacKinlay):
- VR ≈ 0,98–0,99 em todos os horizontes; **nenhum é significativo** (|z| robusto ≤ 1,5);
- o sinal muda de ano para ano;
- não há reversão estável no agregado.

**Estrutura escondida: o relógio de 5 minutos.**
- A barra que termina em cada marca de 5 min (:x0/:x5) é ~7% mais volátil.
- O movimento dela é parcialmente devolvido nos 1–2 min seguintes (ρ = −0,026, −0,016).
- Ela se correlaciona com a barra da **próxima** marca (ρ = +0,029, z = 5,7; sobrevive à correção de Bonferroni).
- Não é efeito de notícias: aparece em todas as marcas, não só em :00/:30. Provável microestrutura: algoritmos de execução ou operadores agindo num relógio de 5 min.
- Forte em 2021–2023, **ausente em 2020 e 2024**.

**Dinheiro:**
- Apostar contra a barra da marca e sair 2 min depois: custo de equilíbrio de **+1,3 pt** no geral e +2,9 ± 1,1 pts depois de movimentos grandes. Fica **muito abaixo do cenário base (7,5 pts)**. Só encosta no otimista (2,5), o menos crível para reversão: uma ordem limitada contra o movimento é executada justamente quando ele continua (seleção adversa).
- Nas outras barras há leve continuação.
- Uso possível: **execução** (evitar entrar logo depois de uma barra forte da marca na mesma direção), não sinal.

**Lições gerais:**
1. O método de cálculo pode criar o efeito. Testar o método numa simulação em que a resposta é conhecida.
2. **O alinhamento dos candles com o relógio importa.** Diferenças entre discretizações (5, 7, 10 min…) podem vir do alinhamento com o ciclo de 5 min, não de uma escala "certa".
3. A estrutura foi encontrada **explorando**. Se for usada, confirmar na segunda validação.

### 4.7 O custo por operação (levantado por Ana, 05/11/2026)

**De onde veio:** não foi calculado. O `PLANO_COMPARACAO_MODELOS_Tt+1.md` diz "ex.: WIN 0 / 5 / 10 pontos… Confirmar com o professor valores realistas", e o item "Confirmar custos operacionais realistas" nunca foi feito. A reunião de 02/11 não discute custos.

**Do que é feito o custo de uma operação completa** (entrada + saída, 1 contrato; 1 ponto = R$ 0,20):

| Componente | Valor | Em pontos | Observação |
|---|---|---|---|
| Taxas da B3 (emolumentos + registro), day trade | ≈ R$ 0,25 por lado → **R$ 0,50** | **2,5** | Faixa de menor volume; cai com volume alto |
| Corretagem | Frequentemente **zero** em minicontratos | 0 (varia) | Depende da corretora |
| **Spread** (cruzar compra/venda) | Tick do WIN = 5 pts; o spread quase sempre é 1 tick (coluna `SPREAD` = 5 em 99,8% dos minutos) | **≈ 5** com ordens a mercado | Meio spread na entrada + meio na saída. Com ordens limitadas, ≈ 0, mas sem garantia de execução |
| Slippage | Desconhecido | ? | Stops e notícias executam pior que o preço da barra. Não dá para medir com candles de 1 min |
| Imposto | 20% do lucro líquido mensal de day trade (+1% retido na fonte) | — | Não é custo por operação; reduz o lucro, não muda o sinal da esperança |

**Cenários:**

| Execução | Custo por operação |
|---|---|
| Ordens limitadas (passivas) | ≈ 2,5 pts, mas com risco de não executar e de seleção adversa |
| Ordens a mercado, spread de 1 tick | ≈ **7,5 pts** |
| Com slippage em stops e notícias | 10+ pts |

O valor de 5 pts fica no meio: equivale a taxas + meio spread (um lado passivo, outro agressivo). É defensável como cenário intermediário, mas **não foi derivado**.

**O custo e o simulador são o mesmo assunto.** O `simular`:
- entra no fechamento (último negócio, que pode ter sido na compra ou na venda);
- executa o TP no preço exato assim que a máxima o toca. É otimista: tocar não garante execução; o seguro é o preço passar 1 tick além;
- executa o stop no preço exato. Também é otimista: o stop vira ordem a mercado e escorrega.

**Medir com os dados que temos:**
- a coluna `SPREAD` só mostra o spread mínimo do candle;
- o estimador de Roll em 1 min dá resultado sem sentido (covariância positiva em 2020–21), porque se mistura com o ciclo de 5 min e com movimentos reais.

Para medir o spread efetivo e o slippage seriam precisos os **arquivos tick a tick da B3** (grátis, últimos ~20 pregões). É um bom uso deles.

**Implementação local:** [custos.py](custos.py). Não altera o `tt1_comum.py`.
- **Cenários:**

  | Cenário | Custo fixo | TP | Stop |
  |---|---|---|---|
  | otimista | 2,5 pts | executa se o preço passar 1 tick além | no preço do stop |
  | base | 7,5 pts | idem | no preço do stop |
  | conservador | 12,5 pts | idem | 1 tick pior |

  Distâncias de TP/SL arredondadas ao tick.
- `resumo` devolve o **custo de equilíbrio** (esperança bruta) com erro-padrão, e o t da esperança.
- `LEGADO` reproduz exatamente o `tt1_comum.simular` (verificado nas 36 regras).
- **Resultado oficial da frente** (ARIMAX, teste) no modelo novo: custo de equilíbrio de **−36 ± 42 pts**, ou seja, perde mesmo com custo zero. Aleatório nos mesmos candles: −4 [−64; +58]. Conclusão inalterada.
- Estudos 01 e 02 adaptados para os cenários e o custo de equilíbrio.


### 4.8 Estudo 03 – H2: gap de abertura

Notebook: [estudos/03_h2_gap_de_abertura.ipynb](estudos/03_h2_gap_de_abertura.ipynb). Dados até 2024, sem dias de rolagem e sem pregões logo após buraco (903 dias).

**Poder estatístico (conceito novo):** com 1 operação por dia, o erro-padrão da esperança é ~16 pts (σ ≈ 490 pts na 1ª hora). O **efeito mínimo detectável** (80% de poder) é ~**45 pts por operação**. "Não significativo" aqui quer dizer "não dá para saber".

| Estratégia | Efeito mínimo detectável |
|---|---|
| 1 operação por dia | ~45 pts |
| A cada candle de 60 min | ~13 pts |
| A cada candle de 5 min | ~1 pt |

**O gap prevê a manhã ou o dia?** Nenhuma correlação significativa (|z| ≤ 1,4). Apostar contra o gap dá de +2 a +18 pts, com erro de 8 a 44 pts. Inconclusivo.

**"O gap fecha 90% das vezes" não é informação.** A frequência é real (89% dos gaps pequenos fecham no dia), mas o **placebo** (gap de um dia com o caminho do preço de outro dia) dá o mesmo (88%).

**Candidato C1 (gaps grandes, continuação):**
- Com |gap| > 0,5% (221 dias), entrar no fechamento do 1º minuto **a favor** do gap e sair 5 min depois dá **+44 ± 19 pts** (t ≈ 2,3).
- Por ano: positivo em 2021, 2023 e 2024; ≈ 0 em 2022 (o ano com mais dias); negativo em 2020.
- Foi encontrado entre ~36 variações.
- A abertura tem spread mais largo, então o cenário de custo adequado é o conservador.
- **Pré-registrado** (Seção 6).


### 4.9 Estudo 04 – H3: horário e momentum intradiário

Notebook: [estudos/04_h3_horario.ipynb](estudos/04_h3_horario.ipynb). Dados até 2024.

- **ρ₁ por hora** (candles de 5/15/30 min, 29 testes): 4 com |z| > 2 sem correção (o acaso daria ~1,5). **Nenhum sobrevive à correção de Holm.**
- **Deriva por hora:** o candle das 11h tem −39 pts (t = −2,4), mas é 1 em 10 testes e não sobrevive à correção.
- **Gao et al. (1ª meia hora → última):** não se confirma. z ≈ 1 com o gap (definição do artigo); ≈ 2 sem o gap, limítrofe e não robusto.
- **Baltussen et al. (resto do dia → última meia hora): confirmado e forte.**
  - ρ ≈ 0,18–0,20, z robusto 3,9–4,7 conforme a definição.
  - Custo de equilíbrio de **+28 a +36 pts por operação** (uma por dia), acerto de 53–56%.
  - **Específico do fechamento:** a mesma conta em qualquer outra janela do dia dá ≈ 0.
  - Positivo em todos os anos completos (2021: +15; 2022: +44; 2023: +45; 2024: +27). Continua positivo sem os 5% de dias extremos (+21); mediana de +25.
  - **Só existe nos pregões até 18:24** (nov–mar até 2023; o ano todo desde 2024): +24 ± 11 quando NY fecha às 17h e +63 ± 14 quando fecha às 18h. Com o pregão até 17:54 (mar–nov de 2021–23), é **zero** (+0,3 ± 12). O "regime" fica parcialmente confundido com o ano.
  - **Não se confirmou na segunda validação** (Seção 4.13).
  - Mecanismo incerto: preso ao fechamento do próprio WIN, não ao de NY. Candidatos: fluxos no fechamento (fundos, hedge de opções), janela do ajuste diário, fim do mercado à vista. Pergunta para o professor: o que muda na última meia hora entre os dois regimes de horário?
- **Relação com T+1:** não aparece como autocorrelação de candles seguidos. É uma previsão do último candle usando o dia inteiro. Um modelo T+1 só com lags não vê isso; features como "retorno acumulado do dia" e "minutos até o fim do pregão" poderiam ver.
- **Candidato C2 pré-registrado** (Seção 6). Poder da segunda validação para um efeito de ~29 pts: **~55%**.


### 4.10 Estudo 05 – H4: depois de movimentos grandes

Notebook: [estudos/05_h4_grandes_movimentos.ipynb](estudos/05_h4_grandes_movimentos.ipynb). Dados até 2024. "Grande" = |z| > 2 ou > 3, com z = retorno ÷ volatilidade recente (só o passado).

- **Caudas gordas:** movimentos de mais de 3 desvios são 5 a 8 vezes mais frequentes que numa normal.
- **Nem continuação nem reversão:** 16 testes (candles de 1/5/15/60 min × limiar × horizonte), nenhum |t| > 1,2. Altas e baixas iguais.
- **Intervalo de confiança contra o custo (conceito novo):** o melhor caso compatível com os dados (|estimativa| + 2 ep) foi comparado com os cenários de custo.

  | Candle | Melhor caso | Conclusão |
  |---|---|---|
  | 1 min | 1,1 pt | não paga nem no otimista |
  | 5 min | 5,6 pts | não paga no base |
  | 15 min | 13,7 pts | não dá para saber |
  | 60 min | 52,4 pts | não dá para saber |

  Onde há muitos dados, dá para **descartar**; onde há poucos, só "inconclusivo".
- **O que é previsível é o tamanho:** depois de um candle grande, o próximo movimento é 30–50% maior, em direção imprevisível. Agrupamento de volatilidade de novo; leva à H6.
- Nenhum candidato novo.


### 4.11 Estudo 06 – H5: regime de volatilidade

Notebook: [estudos/06_h5_regime_volatilidade.ipynb](estudos/06_h5_regime_volatilidade.ipynb). Dados até 2024.

- **Regime decidido só com o passado:** a volatilidade realizada de ontem, comparada com os 60 pregões anteriores, define três terços (calmo/médio/agitado). Ela separa bem a volatilidade de **hoje** (0,89% × 1,25%): a volatilidade persiste.
- **ρ₁ por regime** (5/15/60 min, 9 testes): nenhum significativo (|z| ≤ 1,5). Os sinais sugerem momentum no calmo e reversão no agitado, mas a diferença agitado − calmo tem |z| ≤ 1,4 (**bootstrap em blocos de dias**, conceito novo).
- **Custo:** no agitado os movimentos são ~35% maiores e o ρ necessário cai ~25%, mas o |ρ₁| medido continua bem abaixo dele.
- **C2 por regime** (descritivo; a regra não muda): +25 / +43 / +35 pts (calmo/médio/agitado), acima dos custos nos três. **Ponto de robustez a favor do C2.**
- Nenhum candidato novo.


### 4.12 Estudo 07 – H6: previsibilidade da volatilidade

Notebook: [estudos/07_h6_volatilidade.ipynb](estudos/07_h6_volatilidade.ipynb). Dados até 2024; avaliação fora da amostra em 2022, 2023 e 2024 (treino só com anos anteriores).

- **Volatilidade diária:** autocorrelação de 0,76 com o dia anterior (contra ≈ 0 da direção). O **HAR** (ontem + semana + mês) explica **47–90%** da variação fora da amostra e é o melhor em todos os anos (R² e QLIKE).
- **Bandas de 95% para candles de 60 min** (2022–24, fora da amostra):

  | Modelo | 10h | Horas fora de 95% (Kupiec) |
  |---|---|---|
  | constante (como o SARIMAX) | 89% | 8 de 10 |
  | **`volat` do simulador** | **77%** | 6 de 10 (pior QLIKE) |
  | perfil × nível do dia (HAR) | 94% | 4 de 10 |
  | GARCH(1,1) no perfil | 94% | 5 de 10 |
  | **perfil dos últimos 12 meses × nível** | 94% | **1 de 10** (só 9h, com 90%) |

- **Não eram as caudas gordas:** o multiplicador empírico da abertura no treino é ≈ 2,0. **O padrão intradiário mudou:** a hora das 9h foi de ~1× a média do dia (2020–21) para ~2× (2022–24). Por isso o perfil precisa ser recente.
- **O `volat` do simulador comum é o pior modelo:** stops e alvos ficam apertados de manhã e largos à tarde.
- **Risco por operação:** com σ constante, a hora mais arriscada tem ~3× o risco da menos arriscada; com perfil × nível, ~1,5×.
- **Para que serve:** não dá lucro sozinha (não diz a direção). Serve para bandas e stops calibrados e para dimensionar posições (risco igual por operação).
- **Resolve a pendência da frente:** a banda que cobria 74% às 10h passa a ~94%.


### 4.13 Estudo 08 – segunda validação (01/2025 → 03/2026)

Notebook: [estudos/08_validacao_candidatos.ipynb](estudos/08_validacao_candidatos.ipynb). Rodado **uma vez**, com as regras pré-registradas, cenário de custo conservador. O mesmo código reproduz os números do desenvolvimento (verificação da implementação).

| Candidato | Validação (custo de equilíbrio) | Desenvolvimento | Resultado |
|---|---|---|---|
| C1 – gap grande | +31 ± 53 pts (n = 51) | +44 ± 19 (n = 221) | **inconclusivo** (sem poder: só 51 operações) |
| C2 – momentum no fechamento | **−7 ± 15 pts** (n = 302) | +29 ± 8 (n = 901) | **não confirmado**; intervalo de 95% [−37; +23] exclui o valor do desenvolvimento (diferença com z ≈ 2,1) |
| C2b – pregões até 18:24 | −6 ± 15 pts (n = 300) | +51 ± 11 (n = 510) | **não confirmado** (diferença com z ≈ 3,1) |

- Critério decidido antes de rodar: leitura estrita (t da esperança líquida > 2) como oficial; branda (custo de equilíbrio > 12,5 e t bruto > 2) também reportada. **Nenhum passa em nenhuma.**
- **Nenhum candidato direcional segue para o teste final**, que continua intocado.
- Desde 2024 o pregão vai sempre até 18:24: o subgrupo do C2b misturava nov–mar de 2020–23 com o ano todo de 2024.
- **Lição:** um efeito com z ≈ 4, estável ano a ano, robusto à definição e apoiado na literatura **falhou fora da amostra**. Possíveis causas: o efeito sumiu (microestrutura e regras de mercado mudam), ou o tamanho estimado estava inflado pela exploração. Sem a validação, o C2 teria sido apresentado como estratégia lucrativa.


### 4.14 Estudo 09 – ideias do professor (aula de 05/11/2026)

Notebook: [estudos/09_ideias_do_professor.ipynb](estudos/09_ideias_do_professor.ipynb). Dados até 2024 e as previsões de 2024 das frentes estatística (ARIMA) e de ML (Random Forest).

**1. "Altas e baixas são fundamentalmente diferentes" → dois modelos.** O ARMA usa o mesmo coeficiente para altas e baixas; "diferente" pode ser na direção, nas caudas, na volatilidade ou em regimes.
- **Direção, intradiário: simétrica.** b(alta) e b(baixa) estatisticamente iguais (|z| ≤ 1,3); caudas simétricas; o tamanho do próximo movimento é igual depois de altas e baixas grandes.
- **Volatilidade, diário: assimétrica.** Um dia de queda aumenta a volatilidade de amanhã em +7,3% por desvio, contra +4,0% de um dia de alta (diferença com z ≈ 2,5): o **efeito alavancagem**.
- O lugar natural para "dois modelos" nos nossos dados é o **modelo de volatilidade**, não o de direção.

**2. "Entrar só quando os dois concordam."**
- Com erros independentes: P(acerto | concordam) = p²/(p² + (1−p)²). Dois modelos de 55% viram 60%; dois de 50% continuam em 50%. O consenso **amplifica** habilidade, não cria; e corta ~metade das operações.
- ARIMA × RF em 2024: as previsões quase não se correlacionam (0,07–0,10), o que seria ideal, mas nenhum tem habilidade (IC de −0,04 a +0,01). Concordando, acertam 48–49%.

**3. "O alvo errado": erro × correlação.**
- Minimizar o MSE leva à média condicional; em séries persistentes, ela fica perto do último valor, e a previsão parece a série deslocada.
- O MSE encolhe previsões fracas (o tamanho ótimo é σ_f = ρ·σ_y) e o melhor Theil U possível é √(1 − ρ²): com ρ = 0,05 (51,6% de acerto), Theil U ≥ 0,9987. **MSE e Theil U são cegos a correlações pequenas mas úteis**; para direção, a métrica é o **IC**.
- **Mas a correlação em nível também se engana com a persistência.** HAR (volatilidade diária, 2022–24): correlação de 0,97–0,99 com o valor de ontem; com o alvo, 0,46–0,71, contra 0,44–0,68 de simplesmente repetir ontem. **Nas mudanças**, o HAR mostra habilidade real: correlação de 0,45–0,53, acerta se a volatilidade sobe ou desce em 56–64% dos dias, e tem MSE ~19% menor que "repetir ontem".
- Em modelos lineares, minimizar o MSE já maximiza a correlação dentro da amostra. O objetivo faz diferença em modelos não lineares e com volatilidade variando; remédio prático: treinar no retorno dividido pela volatilidade prevista.

**4. "Acertar pouco e ganhar muito."**
- Entradas aleatórias em candles de 60 min (2024), quatro regras de saída: acerto de 35% a 61%, payoff de 0,55 a 1,75, e esperança antes do custo **≤ +0,2 pt em todas**. Os valores negativos vêm dos atritos do simulador (stop primeiro quando os dois são tocados; TP só passando 1 tick).
- É um **formato**, não uma vantagem (teorema da parada opcional). Só dá lucro se o modelo souber **quando** os movimentos grandes vão a favor. Nos nossos dados, o **tamanho** dos movimentos é previsível; a **direção** não.

---

## 5. Cuidados permanentes

1. **Significância estatística ≠ significância econômica.** Um efeito pode ser real e não pagar o custo. Reportar o **custo de equilíbrio** (± erro-padrão) e compará-lo com os cenários de 2,5 / 7,5 / 12,5 pts (ou ρ\* = c / M).
2. **Usar o erro-padrão robusto** (agrupamento de volatilidade). A régua 2/√n exagera a significância.
3. **Contar quantas coisas foram testadas.** Quanto mais hipóteses, cortes e regras, mais a melhor engana. Levar poucos candidatos ao teste.
4. **Teste final intocado:** abr–set/2026, uma vez, no fim.
5. **EDA só com dados até 2024.**
6. **Nunca atravessar a noite nem os buracos** sem tratar: rolagem do WIN (vencimento), rolagem do WDO (último dia do mês), buracos da base.
7. **Horário do pregão:** até 2023 mudava em março e novembro (fim às 17:54 ou 18:24); **desde nov/2023 vai sempre até 18:24**. Afeta features de horário e comparações entre épocas.
8. **Comparar com o aleatório nos mesmos candles**, não com o aleatório operando sempre.
9. **Reportar esperança ± erro-padrão e nº de operações**, não só o resultado em R$.
10. TP e SL tocados no mesmo candle → conta o stop (hipótese conservadora do simulador).
11. **Testar o método numa simulação com resposta conhecida** antes de confiar numa estatística. Perfil intradiário, candles incompletos e fronteiras do dia distorcem cálculos simples (Estudo 02).
12. **Alinhamento dos candles com o relógio** muda o que se enxerga (ciclo de 5 min). Ao comparar discretizações, considerar também a fase dos candles.
13. **Poder estatístico:** antes de testar, calcular o efeito mínimo detectável (≈ 2,8 × σ/√n). Com poucas operações, "não significativo" não quer dizer "não existe".
14. **Placebo:** uma frequência impressionante só vale comparada com uma referência que tenha tudo igual, menos o efeito.
15. **Pré-registro:** quando a exploração encontra algo, escrever a regra exata antes de olhar dados novos e testar uma vez.
16. **Correção para comparações múltiplas (Holm)** em toda grade de horários, tamanhos ou limiares.
17. **Hipóteses da literatura valem mais que as garimpadas.** Subgrupos descobertos nos dados (ex.: regime de horário) são hipóteses secundárias.
18. **Buscar o mecanismo:** saber por que um efeito existe diz quando ele pode sumir.
19. **Comparar o intervalo de confiança com o custo**, não só o p-valor com 0,05. Intervalo inteiro abaixo do custo: "não paga" (evidência de ausência). Intervalo contendo o custo: "não dá para saber".
20. **Limites e classificações só com o passado** (ex.: terços de volatilidade com janela móvel), mesmo em estudos descritivos.
21. **Bootstrap em blocos** quando não há fórmula para o erro-padrão; os blocos (ex.: dias) preservam a dependência.
22. **Testar a diferença entre grupos diretamente**: "significativo num grupo e não no outro" não prova que diferem.
23. **Padrões "estáveis" também mudam:** o perfil intradiário de volatilidade do WIN dobrou na abertura entre 2021 e 2024. Re-estimar com dados recentes.
24. **Erro e correlação em nível enganam em séries persistentes** (preço, volatilidade): comparar sempre com "repetir o último valor" e medir nas **mudanças**. Para direção, avaliar pelo **IC** (correlação com o retorno), não só pelo Theil U.
25. **Regras de saída (TP/SL) não criam vantagem:** mudam a taxa de acerto e o payoff, não a esperança.
26. **Consenso de modelos amplifica, não cria:** só combinar modelos que já tenham IC > 0 sozinhos.

---

## 6. Pendências

### Retomar daqui (próxima sessão, depois de 05/11/2026)

Opções em aberto, sem ordem definida:

- [ ] **Perguntar ao professor** sobre a pesquisa dele com dois modelos: qual era o acerto (ou o IC) de **cada** modelo sozinho, e se os erros dos dois eram pouco correlacionados. O consenso só amplifica habilidade que já existe (Estudo 09, Seção 2).
- [ ] **Modelo em dois estágios:** volatilidade primeiro (perfil recente × HAR, Estudo 07), depois a direção no retorno dividido pela volatilidade prevista, avaliado por **IC**. É uma hipótese nova: explorar só com dados até 2024 e pré-registrar a regra antes de qualquer dado novo.
- [ ] **Pré-registrar o modelo de volatilidade** para o teste final (abr–set/2026), com critério escrito antes (por exemplo, cobertura das bandas de 95% por hora pelo teste de Kupiec). É a única coisa da frente pronta para um teste em dados nunca vistos.
- [ ] Opcional: **resumo visual de uma página** da reabordagem para a reunião do grupo.
- [ ] Enviar o branch `estatistico-reabordagem` (push) e abrir o PR para o grupo revisar.

### Grupo

- [ ] Concordar com a divisão dos dados (Seção 3), em especial: **abr–set/2026 fica intocado** a partir de agora.
- [ ] Concordar com o que é comum a todos (Seção 1): simulador, custos, teste final, esperança ± erro-padrão.
- [ ] Avisar que o **WDO termina em 01/04/2026** (não cobre o teste final).
- [ ] Avisar a Ana: a rolagem do WDO é no **último** pregão do mês, não no primeiro (EDA, Seção 7).
- [ ] Avisar a Ana: o VR(60) = 0,89 da EDA (Seção 11) é, em boa parte, artefato. A correção e a prova por simulação estão no Estudo 02, Seção 2.
- [ ] **Decidir o modelo de custo** (Seção 4.7), na próxima reunião. Já implementado localmente em `custos.py`, com os estudos adaptados. Proposta:
  - reportar o **custo de equilíbrio** de cada estratégia (o custo máximo que ela aguenta com esperança ≥ 0);
  - avaliar em 3 cenários: 2,5 pts (só taxas, ordens limitadas, exigindo que o preço passe 1 tick do alvo) · **7,5 pts** (ordens a mercado, base) · 12,5 pts (+1 tick de slippage);
  - ajustar o simulador: TP só executa se o preço passar 1 tick além; stop com slippage.
- [ ] Alinhar o critério de escolha (t ≥ 2, como na frente de ML) e o teste contra o naive (Clark-West).
- [ ] Quando possível, pedir ao professor: nova exportação a partir de 25/09/2026 e tentativa de preencher o buraco de 2023 (Seção 2.3).

### Frente estatística – correções

- [ ] **Notebook do cone** ([cone_previsao_arima.ipynb](cone_previsao_arima.ipynb)): o cone diário usa fechamento → fechamento da série sem ajuste de rolagem, então inclui um salto falso de ~+1,7% a cada 2 meses (σ e largura do cone inflados). O cone de 60 min atravessa os gaps noturnos. Corrigir: retirar os dias de vencimento ou ajustar a série (*back-adjust*).
- [ ] **Aleatório nos mesmos candles** em `frente_estatistico.ipynb` e no relatório (Seção 4.2).
- [ ] Teste de Clark-West ou Diebold-Mariano contra o naive.
- [ ] Opcional: *bootstrap* em blocos ou *Reality Check* de White para medir o viés de seleção das 180 combinações.
- [x] Repositório: tudo commitado no branch `estatistico-reabordagem` (05/11/2026), ainda sem push nem merge. Inclui a curva de capital só do modelo (versão para slide, de 03/11) no `frente_estatistico.ipynb`.

### Análise exploratória orientada por hipóteses (dados até 2024)

Cada hipótese responde: tamanho do efeito, erro-padrão robusto, estabilidade ano a ano, comparação com o custo.

- [x] **H1 – Reversão à média intradiária.** Feito (Estudo 02, Seção 4.6). Não há reversão estável no agregado; há um ciclo de 5 min que não paga o custo.
  - [ ] Opcional: o ciclo de 5 min existe no WDO? Ele se concentra em algum horário ou dia?
- [x] **H2 – Gap de abertura.** Feito (Estudo 03, Seção 4.8). Inconclusivo no geral por falta de poder; "fechar o gap" não é informação; candidato C1 nos gaps grandes.
- [x] **H3 – Previsibilidade por horário.** Feito (Estudo 04, Seção 4.9). Nada por hora depois da correção; **momentum no fechamento** (Baltussen et al.) forte. Candidato C2.
- [x] **H4 – Depois de grandes movimentos.** Feito (Estudo 05, Seção 4.10). Nem continuação nem reversão; em 1 e 5 min dá para descartar; o tamanho do próximo movimento é previsível.
- [x] **H5 – Por regime de volatilidade.** Feito (Estudo 06, Seção 4.11). Nada muda entre regimes; o C2 é robusto ao regime.
- [x] **H6 – Previsibilidade da volatilidade.** Feito (Estudo 07, Seção 4.12). Muito previsível; o melhor modelo é perfil recente × HAR; o `volat` do simulador é o pior.

**Análise exploratória concluída** (H1–H6). Candidatos para a segunda validação: C1, C2 (e C2b).

### Depois da EDA

- [ ] Levar 1 ou 2 candidatos para a segunda validação (01/2025–03/2026). **Candidatos pré-registrados** (não alterar depois de ver dados novos):
  - **C1 – gap grande, continuação:** em dias sem rolagem e não seguintes a buraco, se |gap| > 0,5%, entrar no fechamento do 1º minuto na direção do gap e sair no fechamento do 5º minuto. Avaliar com `custos.py`, cenário conservador. Sucesso: custo de equilíbrio > 12,5 pts **e** t > 2.
  - **C2 – momentum no fechamento (principal):** em pregões completos, sem rolagem e não seguintes a buraco, calcular o retorno do fechamento de ontem até o fechamento do minuto 30 min antes do último minuto do pregão; entrar nesse minuto na direção desse retorno e sair no fechamento do último minuto. Cenário conservador. Sucesso: custo de equilíbrio > 12,5 pts **e** t > 2.
  - **C2b – secundária:** a mesma regra só nos pregões que vão até 18:24. (Desde 2024 isso inclui quase todos os pregões.) Subgrupo descoberto nos dados: se só ela passar, o resultado é sugestivo, não confirmado.
  - Poder esperado na segunda validação (~300 pregões): ~55% para um efeito de 29 pts. "Não confirmou" não será prova de que não existe.
- [x] Modelo de volatilidade: perfil intradiário + HAR / GARCH, com Kupiec e QLIKE (Estudo 07).
- [ ] **Proposta ao grupo:** trocar o `volat` do simulador comum pelo "perfil 12 meses × nível (HAR)" na distância de TP e SL. Junto com a proposta de custos.
- [ ] Opcional: re-estimar o perfil a cada trimestre (a abertura continua mudando); teste de Christoffersen (furos agrupados no tempo).
- [ ] Opcional, como hipótese **separada** (não altera as regras pré-registradas): dimensionar as operações de C1/C2 pela volatilidade prevista (risco igual por operação).
- [x] Segunda validação de C1, C2 e C2b (Estudo 08, Seção 4.13): **nenhum confirmado**.
- [ ] **Decidir o uso do teste final** (abr–set/2026). Nenhuma estratégia direcional sobreviveu. Opção: pré-registrar o **modelo de volatilidade** (perfil 12 meses × HAR) e usar o teste final para confirmar a calibração das bandas por hora. Atenção: o WDO não cobre esse período (irrelevante para o modelo de volatilidade, que só usa o WIN).
- [ ] Levar ao grupo a resposta da frente estatística para a pergunta do projeto (Seção 4.13 + 4.12): nenhuma vantagem direcional T+1 sobreviveu fora da amostra; o que é robustamente previsível é a volatilidade.
- [ ] Teste final único em abr–set/2026.
- [ ] Discutir com o grupo as ideias do professor (Estudo 09, Seção 4.14), em especial o **IC** como métrica comum para previsão de direção, ao lado do Theil U.
- [ ] Opcional: termo de assimetria (efeito alavancagem) no modelo de volatilidade (HAR ou GARCH com quedas separadas).
- [ ] Opcional: modelo em dois estágios (volatilidade → direção no retorno dividido pela volatilidade prevista), avaliado por IC. Seria uma hipótese nova: exploração até 2024 e pré-registro antes de qualquer dado novo.
- [ ] Opcional: começar a coletar os arquivos tick a tick da B3 para um teste futuro.

---

## 7. Para estudar

| Tema | Onde |
|---|---|
| Correlação, p-valor, comparações múltiplas | StatQuest (YouTube); Khan Academy (estatística) |
| Séries temporais, autocorrelação, ARIMA | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (otexts.com/fpp3), cap. 2, 5 e 9; Morettin & Toloi, *Análise de Séries Temporais* |
| Fatos estilizados dos retornos | R. Cont (2001), "Empirical properties of asset returns: stylized facts and statistical issues" |
| Overfitting em backtest | Bailey, Borwein, López de Prado & Zhu (2014), "Pseudo-Mathematics and Financial Charlatanism"; depois, López de Prado, *Advances in Financial Machine Learning* |
| Volatilidade (GARCH) | Tsay, *Analysis of Financial Time Series*; Morettin, *Econometria Financeira* |
| Volatilidade realizada (HAR) | Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility" |
| Efeito alavancagem | Black (1976); Glosten, Jagannathan & Runkle (1993), o GJR-GARCH |
| Mudança de regime | Hamilton (1989), modelos de Markov com troca de regime |
| Correlação espúria em séries persistentes | Granger & Newbold (1974), "Spurious Regressions in Econometrics" |
| IC e a "lei fundamental" da gestão ativa | Grinold & Kahn, *Active Portfolio Management* |
| Momentum no fechamento | Gao, Han, Li & Zhou (2018); Baltussen, Da, Lammers & Martens (2021) |
