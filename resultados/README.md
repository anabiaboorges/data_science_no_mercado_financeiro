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

`comparacao/` reúne as três frentes (`comparacao_frentes.ipynb` e a tabela final). Ver Seções 5 e 6 do [plano](../planos/PLANO_FRENTES_02.11.md).
