# Frente de redes neurais – Gabriel Modesto

Notebook: [`frente_redes.ipynb`](frente_redes.ipynb). Modelo base: **MLP** na grade candle × janela (12 rodadas); variante: **GRU / LSTM** na melhor combinação. Só **WIN**: nenhuma feature do mini dólar entra nos modelos.

Saídas em `resultados/redes/`. Ver Seção 3.3 do [plano](../../planos/PLANO_FRENTES_02.11.md), o [`resumo.md`](../../resultados/redes/resumo.md) e o [`relatorio.md`](../../resultados/redes/relatorio.md).

## Como rodar

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
jupyter nbconvert --to notebook --execute --inplace frentes/redes/frente_redes.ipynb
```

Roda em ~3 minutos (2 núcleos). As sementes são fixas e o TensorFlow roda em modo determinístico: duas execuções dão o mesmo `teste_final.csv`. As previsões de **1 min** não vão para o repositório (`.gitignore`); são recriadas pelo notebook.

## Escolhas e critério

Critério do **plano comum**: maior esperança por operação com custo de 5 pts (empate → menor Theil U). O *t* de cada esperança e o Clark-West são reportados só para leitura.

| Etapa | Resultado |
|---|---|
| Grade (validação 2024) | Melhor: 60 min × 1 mês (+2,5 pts/op, t = 0,40). Theil U > 1 nas 12 combinações |
| Regra | Quantil 90%, TP 1×, SL 1× (da LSTM) |
| Variante | LSTM (+28,2 pts/op, t = 1,45) > MLP `alpha = 0,1` > MLP > GRU |
| Teste 01/2025–03/2026 | LSTM: Theil U 1,0022, −20,4 pts/op com custo de 5 pts (t = −0,96), R$ −1.819 em 445 operações; aleatório −29,9 pts/op |

## Resultados (custo de 5 pts por operação)

**Conclusão:** as redes não batem o naive (Clark-West t = 0,32) e a regra perde no teste mesmo sem custo. O MLP sobreajusta com janelas curtas; GRU e LSTM só chegam perto do naive encolhendo a previsão para perto de zero. Como nas outras frentes, o Theil U se repete entre validação e teste (Spearman 0,97), mas a esperança não (0,17).
