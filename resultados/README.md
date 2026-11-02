# Resultados

Uma pasta por frente (`estatistico/`, `ml/`, `redes/`), todas com os mesmos arquivos, gerados pelo `tt1_comum.py`:

| Arquivo | Conteúdo |
|---|---|
| `previsoes/WIN_{candle}_{janela}m_{modelo}_{val\|teste}.csv` | Previsões: datetime, y_true, y_pred, close, high_prox, low_prox, close_prox, volat |
| `grade_val.csv` | 12 combinações candle × janela: métricas e simulação |
| `heatmap_theilU.png`, `heatmap_acuracia.png`, `heatmap_esperanca.png` | Heatmaps da grade |
| `regras_val.csv` | Limiar × take profit × stop loss na validação |
| `variante_val.csv` | Variante da frente na melhor combinação |
| `config_final.json` | Escolhas congeladas antes do teste |
| `teste_final.csv` | Modelo e sinais aleatórios no teste, custos 0/5/10 pts |
| `curva_capital_teste.png`, `previsto_vs_real_teste.png` | Gráficos do teste |
| `resumo.md` | 5 linhas de conclusão |

## Regra de operação: limiar, take profit, stop loss e custo

Todas as frentes simulam com a função `simular` do [`tt1_comum.py`](../tt1_comum.py). A regra decide, a cada candle *t*, a partir da previsão do retorno de *t+1* (em %).

### Mecânica

| Item | Definição |
|---|---|
| Entrada | **Compra** se previsão > +limiar; **vende** se previsão < −limiar; senão fica de fora. Entra no fechamento de *t* |
| Saída | Fechamento de *t+1*, ou antes, se tocar o take profit ou o stop loss dentro do candle *t+1* (máxima/mínima) |
| TP e SL no mesmo candle | Conta o **stop** (hipótese conservadora: com candles não se sabe qual veio primeiro) |
| Fim do pregão | Nenhuma posição atravessa a noite (o último candle do dia não tem alvo) |

### Parâmetros

| Parâmetro | Definição | Opções testadas na validação |
|---|---|---|
| **Limiar** | Quantil de \|previsão\| calculado nas previsões da **validação 2024** do próprio modelo, convertido em valor fixo (em %) e congelado para o teste. Usar quantil deixa a regra comparável entre modelos que preveem em escalas diferentes | 0 (opera sempre), 50%, 75%, 90% |
| **Take profit (TP)** | Múltiplo da volatilidade recente: desvio-padrão dos retornos do último pregão (s candles: 565 em 1 min, 38 em 15 min, 10 em 60 min), em % × preço de entrada | sem, 1×, 2× |
| **Stop loss (SL)** | Mesma base do TP | sem, 1×, 2× |
| **Custo** | Valor fixo em pontos por operação completa (entrada + saída), descontado de cada operação. Estimativa: spread de ~1 tick (5 pts) com corretagem zero; **não confirmado** com corretora/B3 | **5 pts** na escolha; 0 e 10 pts como sensibilidade no teste |
| Valor do ponto | R$ 0,20 por ponto por contrato (WIN) | – |

São 4 × 3 × 3 = **36 regras** por modelo. A escolha é feita só na validação (2024), na melhor combinação candle × janela da frente.

### Critério de escolha da regra

- **Plano comum:** maior esperança por operação (pts, já com o custo de 5 pts).
- **Frente de ML (ajuste em avaliação pelo grupo):** maior esperança **entre as regras com t ≥ 2** (esperança ÷ erro-padrão das operações). Se nenhuma passar, usa a regra padrão (opera sempre, sem TP/SL), que não é escolhida olhando o resultado. Motivo: com poucas operações, a "melhor" regra da validação é ruído e não se repete no teste (ver `ml/README.md` e a seção 7 de `estatistico/relatorio.md`).

### Regras adotadas por frente (`config_final.json`)

| Frente | Modelo / candle / janela | Limiar | TP | SL | Custo na escolha | Como foi escolhida |
|---|---|---|---|---|---|---|
| Estatístico | ARIMAX, 60 min, 6 meses | quantil 90% (0,0287% ≈ 37 pts) | 1× a volatilidade | sem | 5 pts | Maior esperança na validação (+10,2 ± 18,5 pts/op, não significativa) |
| ML | RF `rf_d4_f2`, 1 min, 12 meses | 0 (opera sempre) | sem | sem | 5 pts | Nenhuma das 36 regras teve t ≥ 2 → regra padrão |
| Redes neurais | – | – | – | – | – | Pendente |

Variantes da frente de ML testadas depois (hipóteses registradas antes de rodar, em `ml/hipotese_*.json`): segurar a posição até a previsão virar de sinal e alvo de 10/20 minutos, com os mesmos limiares por quantil, custo de 5 pts (2,5 na entrada + 2,5 na saída) e sem TP/SL.

---

`comparacao/` reúne as três frentes (`comparacao_frentes.ipynb` e a tabela final). Ver Seções 5 e 6 do [plano](../planos/PLANO_FRENTES_02.11.md).
