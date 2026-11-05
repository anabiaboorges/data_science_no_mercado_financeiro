# Estudos da reabordagem – frente estatística

Notebooks da reabordagem iniciada em 04/11/2026, depois da apresentação que não aconteceu. O grupo trocou a pergunta para **"qual a melhor abordagem para ter ganhos reais usando previsão T+1?"**, e a frente estatística decidiu entender o problema antes de modelar: dados, custos, ruído, e seis hipóteses sobre onde poderia haver previsibilidade.

- Achados, cuidados e pendências: [`../NOTAS.md`](../NOTAS.md) (começar pelo **Resumo**).
- Modelo de custo e de execução usado nos estudos: [`../custos.py`](../custos.py).
- Protocolo original da frente (02/11): [`../README.md`](../README.md) e [`relatorio.md`](../../../resultados/estatistico/relatorio.md).

## Como rodar

- **Ambiente:** o `.venv` do projeto (Python 3.12 com numpy, pandas, matplotlib, scipy e statsmodels). Não precisa de pacote extra: o GARCH do Estudo 07 é implementado no próprio notebook.
- **Caminhos:** cada notebook encontra a raiz do repositório sozinho (sobe as pastas até achar o `tt1_comum.py`) e importa o `tt1_comum` e o `custos.py`. Funciona com o diretório de trabalho na pasta do notebook ou na raiz.
- **Sempre reiniciar o kernel e executar tudo** (*Restart* → *Run All*). As células dependem das anteriores. Rodar uma célula isolada depois de reiniciar o kernel dá `NameError`.
- Os notebooks não gravam nada em `resultados/`; tudo fica nas saídas das células.
- Tempo total: ~1,5 min para os nove, verificado em 05/11/2026 com *Restart* → *Run All* em cada um (medido rodando todos do zero; tempos individuais na tabela).

## Que dados cada estudo usa

| Estudos | Período | Por quê |
|---|---|---|
| 01–07 e 09 | **Desenvolvimento:** até 31/12/2024 | Exploração: aqui se pode olhar, testar e escolher à vontade |
| 08 | **Segunda validação:** 01/2025 → 03/2026 | Rodado **uma vez**, com regras escritas antes (pré-registro) |
| — | **Teste final:** 04/2026 → 25/09/2026 | **Intocado.** Reservado para um único teste no fim |

O Estudo 09 usa também as previsões de 2024 já geradas pelas frentes estatística (ARIMA) e de ML (Random Forest), que estão dentro do período de desenvolvimento.

## Índice

| Nº | Notebook | Pergunta | Conceitos novos | Resultado principal | Tempo |
|---|---|---|---|---|---|
| 01 | [`01_ruido_e_autocorrelacao`](01_ruido_e_autocorrelacao.ipynb) | Quanto um número "balança" por acaso? Quanta correlação é preciso para pagar o custo? | correlação, autocorrelação, faixa 2/√n, p-valor, comparações múltiplas, erro-padrão robusto, custo de equilíbrio, ρ\* = c/M | Com o erro-padrão robusto, **nenhum** tamanho de candle (1 a 120 min) tem autocorrelação significativa. Até ~45 min, o ρ necessário para pagar o custo base fica acima do medido; acima disso, não há dados para saber | 12 s |
| 02 | [`02_h1_reversao_intradiaria`](02_h1_reversao_intradiaria.ipynb) | **H1:** o preço desfaz parte dos movimentos dentro do pregão? | razão de variâncias, teste de Lo-MacKinlay, testar o método numa simulação | O VR(60) = 0,89 da EDA é ~3/4 **artefato** do cálculo. Medido direito, não há reversão. Há um **ciclo de 5 min** real (z = 5,7), mas o ganho bruto (1–3 pts) não paga o custo | 19 s |
| 03 | [`03_h2_gap_de_abertura`](03_h2_gap_de_abertura.ipynb) | **H2:** o gap da noite prevê o começo do pregão? | poder estatístico, efeito mínimo detectável, placebo, pré-registro | Inconclusivo por falta de poder (1 operação por dia só detecta efeitos de ~45 pts). "O gap fecha 90% das vezes" é igual ao placebo. Candidato **C1** (continuação em gaps > 0,5%) | 4 s |
| 04 | [`04_h3_horario`](04_h3_horario.ipynb) | **H3:** a previsibilidade depende do horário? | correção de Holm, hipóteses da literatura, buscar o mecanismo | Nada por hora depois da correção. **Momentum no fechamento** (Baltussen et al., 2021): o movimento do dia continua na última meia hora, z ≈ 4, +29 pts/op. Candidato **C2** | 8 s |
| 05 | [`05_h4_grandes_movimentos`](05_h4_grandes_movimentos.ipynb) | **H4:** depois de um movimento grande, continua ou reverte? | intervalo de confiança contra o custo (evidência de ausência × "não dá para saber") | Nem continuação nem reversão (16 testes). Em 1 e 5 min dá para **descartar**; o que é previsível é o **tamanho** do próximo movimento (30–50% maior) | 8 s |
| 06 | [`06_h5_regime_volatilidade`](06_h5_regime_volatilidade.ipynb) | **H5:** muda entre mercado calmo e agitado? | classificar só com o passado, bootstrap em blocos, testar a diferença | Nada muda entre regimes. O C2 é positivo nos três regimes | 12 s |
| 07 | [`07_h6_volatilidade`](07_h6_volatilidade.ipynb) | **H6:** a volatilidade é previsível? Para que serve? | HAR, QLIKE, GARCH(1,1), teste de Kupiec, perfil intradiário | **Muito previsível:** o HAR explica 47–90% da volatilidade diária fora da amostra. Bandas por hora calibradas com perfil dos últimos 12 meses × HAR (10h: 74% → 94%). O `volat` do simulador comum é o pior modelo. O perfil intradiário mudou entre 2020 e 2024 | 7 s |
| 08 | [`08_validacao_candidatos`](08_validacao_candidatos.ipynb) | Os candidatos se confirmam em dados novos? | validação única, critério decidido antes | **Nenhum confirmado.** C1 inconclusivo (51 operações); C2 −7 ± 15 pts/op contra +29 ± 8 no desenvolvimento | 5 s |
| 09 | [`09_ideias_do_professor`](09_ideias_do_professor.ipynb) | Dois modelos (alta/baixa) e consenso; erro × correlação; acertar pouco e ganhar muito | modelos com limiar, efeito alavancagem, consenso de modelos, IC, decomposição do MSE, correlação nas mudanças, parada opcional | Direção simétrica, volatilidade assimétrica. Consenso amplifica habilidade, não cria. MSE e correlação em nível enganam com persistência; o que mede habilidade é a correlação nas mudanças. Saídas mudam o acerto, não a esperança | 8 s |

## Leitura sugerida

1. **01** primeiro: os conceitos dele são usados em todos os outros.
2. **02 a 07** na ordem: cada um acrescenta uma ferramenta.
3. **08**: o desfecho dos candidatos.
4. **09**: a ponte entre os estudos e os modelos de previsão.

## Conceitos, por estudo

Para revisar um conceito, o lugar onde ele é explicado do zero:

| Conceito | Estudo |
|---|---|
| Correlação, autocorrelação, ruído amostral (2/√n), p-valor | 01 |
| Comparações múltiplas e "maldição do vencedor" | 01 (Seção 5) |
| Erro-padrão robusto (agrupamento de volatilidade) | 01 (Seção 6) |
| Da correlação ao dinheiro: acurácia, ganho por operação, ρ\* = c/M | 01 (Seção 7) |
| Custo de equilíbrio e cenários de custo | 01 (Seção 7), `custos.py` |
| Razão de variâncias, teste de Lo-MacKinlay | 02 |
| Testar o método numa simulação com resposta conhecida | 02 (Seção 2) |
| Poder estatístico, efeito mínimo detectável | 03 (Seção 1) |
| Placebo | 03 (Seção 3) |
| Pré-registro | 03 (Seção 4) |
| Correção de Holm | 04 (Seção 1) |
| Intervalo de confiança contra o custo | 05 (Seção 3) |
| Bootstrap em blocos | 06 (Seção 3) |
| HAR, QLIKE, GARCH, Kupiec | 07 |
| Efeito alavancagem, modelos com limiar | 09 (Seção 1) |
| Consenso de modelos | 09 (Seção 2) |
| IC, decomposição do MSE, correlação nas mudanças | 09 (Seção 3) |
| Teorema da parada opcional (saídas não criam vantagem) | 09 (Seção 4) |
