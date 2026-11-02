# Frente de machine learning – Ana Beatriz

Modelo: **Random Forest**. Saídas em `resultados/ml/`. Ver Seção 3.2 do [plano](../../planos/PLANO_FRENTES_02.11.md).

## Notebooks (rodar nesta ordem)

| Notebook | O que faz | Principais saídas |
|---|---|---|
| `frente_ml.ipynb` | Protocolo do plano (Etapas A–F): nº de árvores, grade candle × janela, regra, variantes, teste oficial (01/2025–03/2026) | `config_final.json`, `grade_val.csv`, `teste_final.csv`, `resumo.md` |
| `teste_complementar_grade.ipynb` | As 12 combinações no teste, só para comparar validação × teste (não muda a escolha) | `grade_teste_complementar.csv`, `validacao_vs_teste.png` |
| `teste_abr_set_2026.ipynb` | Hipótese registrada: 15 e 60 min com limiar alto no período nunca visto (04–09/2026) | `hipotese_abr_set_2026.json`, `resultado_abr_set_2026.csv` |
| `variante_segurar_posicao.ipynb` | Hipótese registrada: segurar a posição até a previsão virar de sinal | `hipotese_segurar_posicao.json`, `segurar_posicao.csv` |
| `variante_horizonte.ipynb` | Hipótese registrada: alvo de 10 e 20 minutos à frente | `hipotese_horizonte.json`, `horizonte_resultados.csv` |

Os arquivos de previsão de **1 min** (`resultados/ml/previsoes/*_1min_*`, ~280 MB) não vão para o repositório; são recriados ao rodar os notebooks na ordem acima.

## Ajustes desta frente em relação ao plano comum (em avaliação pelo grupo)

1. **Folha mínima proporcional ao treino** (`max(10, 1% das linhas)`): com valor fixo de 100, a janela de 60 min × 1 mês (~200 candles) virava uma árvore de uma divisão só.
2. **Escolha com significância:** só escolhe pela esperança se ela for positiva com **t ≥ 2**; senão, menor Theil U e regra padrão (opera sempre, sem TP/SL).
3. **Clark-West** para testar se a previsão é melhor que o naive.

## Resultados (custo de 5 pts por operação)

| Abordagem | Período | Esperança (pts/op) | Leitura |
|---|---|---|---|
| Oficial: `rf_d4_f2`, 1 min, 12 meses, opera todo minuto | Teste 01/2025–03/2026 | −4,41 (t = −28,8) | Previsão estatisticamente melhor que o naive (Clark-West t = 3,72), mas o ganho bruto (+0,59 pt) é ~8× menor que o custo |
| Grade completa no teste (complementar) | Teste | todas negativas | O ranking por Theil U se repete (Spearman 0,99); o ranking por lucro, não (−0,24) |
| 15 / 60 min, limiar 90% | Novo 04–09/2026 | −22,2 / −109,5 (sem significância) | Candle maior não resolve |
| **Segurar até virar o sinal** | Teste / novo | limiar 90%: −0,73 / +1,93 (t = −0,4 / 0,6) | Melhor resultado: quase empate, sem evidência de lucro |
| Alvo de 10 / 20 min | Validação, teste, novo | todas negativas | O sinal só existe no minuto seguinte |

**Conclusão:** o Random Forest acha uma previsibilidade real, mas restrita ao minuto seguinte e menor que o custo de operar. A melhor forma de usá-la (segurar a posição enquanto o sinal se mantém) chega perto do empate, sem evidência estatística de lucro. Hipótese para discussão: o sinal de 1 minuto (features mais usadas: corpo do candle e último retorno) pode ser em boa parte *bid-ask bounce*, que não é capturável na prática.
