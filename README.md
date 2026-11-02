# Data Science no Mercado Financeiro – Previsão T(t+1) no Mini Índice

Trabalho da disciplina **Data Science no Mercado Financeiro** (UFG). O objetivo é prever o **retorno do próximo candle** do mini índice (WIN) e verificar se essa previsão vira lucro numa simulação de operações com custos.

Três frentes comparam famílias de modelos sob o **mesmo protocolo**: mesmos dados, mesmo alvo, mesmas janelas de treino e teste, mesmas métricas e a mesma simulação.

| Frente | Responsável | Modelo base | Variante |
|---|---|---|---|
| Estatístico | Itamar da Silva Rocha Filho | ARIMA | ARIMAX / SARIMAX (WDO como exógena) |
| Machine learning | Ana Beatriz Borges Cordeiro | Random Forest | Ajuste de hiperparâmetros + importância das features |
| Redes neurais | Gabriel Modesto | MLP | GRU / LSTM |

A referência de todos é o modelo **naive** (retorno previsto = 0). Na simulação, a referência é um conjunto de **sinais aleatórios**.

---

## Estrutura do repositório

```
├── README.md
├── tt1_comum.py              ← código comum às três frentes (NÃO editar sem combinar com o grupo)
├── dados/                    ← candles de 1 minuto exportados do MetaTrader
│   ├── WINN_M1/WIN$N_M1.csv      mini índice (série contínua)
│   └── WDON_M1/WDO$N_M1.csv      mini dólar (série contínua)
├── planos/                   ← planos e referências teóricas
│   ├── PLANO_FRENTES_02.11.md        plano atual: o que cada frente faz e entrega
│   ├── PLANO_COMPARACAO_MODELOS_Tt+1.md
│   └── MODELO_Tt+1.md
├── reunioes/                 ← transcrições e atas
├── eda/                      ← análise exploratória (WIN × WDO)
├── frentes/                  ← notebooks de cada frente
│   ├── estatistico/
│   ├── ml/
│   └── redes/
├── resultados/               ← saídas geradas pelos notebooks (CSVs, gráficos, resumos)
│   ├── estatistico/
│   ├── ml/
│   ├── redes/
│   └── comparacao/               consolidação das três frentes
└── apresentacoes/            ← slides, roteiros e material de apresentação
```

---

## Protocolo comum

Definido na reunião de 02/11 e implementado em [`tt1_comum.py`](tt1_comum.py). Detalhes em [`planos/PLANO_FRENTES_02.11.md`](planos/PLANO_FRENTES_02.11.md).

| Item | Decisão |
|---|---|
| Ativo | WIN (o WDO entra só como exógena na variante do estatístico) |
| Alvo | Log-retorno do próximo candle, em %: prevê **quanto** sobe ou desce |
| Tamanho do candle | 1, 15 e 60 minutos |
| Janela de treino | 1, 3, 6 e 12 meses (deslizante), com re-treino trimestral |
| Validação | 2024: escolha de candle, janela e regra de operação |
| Teste | 01/2025 → 03/2026: rodado **uma vez**, com tudo congelado |
| Simulação | Compra/venda por limiar, com take profit e stop loss; custo de 0, 5 e 10 pontos por operação |
| Critério | **Esperança matemática por operação** após custos (não a taxa de acerto) |

---

## Como começar

1. Clonar o repositório:
   ```bash
   git clone https://github.com/anabiaboorges/data_science_no_mercado_financeiro.git
   ```
2. Instalar as dependências:
   ```bash
   pip install pandas numpy matplotlib statsmodels scikit-learn
   # redes neurais (GRU/LSTM): pip install tensorflow
   ```
3. No notebook da sua frente, apontar `PASTA` para a raiz do repositório e importar o módulo comum:
   ```python
   import sys; sys.path.append(PASTA)
   import tt1_comum as t
   win = t.carregar(f"{PASTA}/dados/WINN_M1/WIN$N_M1.csv")
   wdo = t.carregar(f"{PASTA}/dados/WDON_M1/WDO$N_M1.csv")
   ```
   No Colab: clonar o repositório no Drive (ou na sessão) e usar `PASTA = "/content/data_science_no_mercado_financeiro"`.
4. Seguir as Etapas A–F da Seção 4 do [plano](planos/PLANO_FRENTES_02.11.md). Se for usar o Claude, entregue o plano junto com a frase da Seção 9.

---

## Como subir a sua frente

- Notebook em `frentes/{frente}/frente_{frente}.ipynb`.
- Saídas em `resultados/{frente}/`, com os nomes de arquivo do checklist da Seção 5 do plano.
- Antes do commit, confira se o notebook roda do início ao fim.
- Commits pequenos e com mensagem clara, por exemplo: `estatistico: grade candle x janela na validação`.
- Cada um mexe só na sua pasta. Mudanças em `tt1_comum.py` ou nos planos são combinadas no grupo antes.

```bash
git pull
git add frentes/ml resultados/ml
git commit -m "ml: grade candle x janela e regra de operação"
git push
```

---

## Andamento

| Entrega | Estatístico | ML | Redes |
|---|---|---|---|
| Grade candle × janela (validação) | ☐ | ☐ | ☐ |
| Regra de operação (validação) | ☐ | ☐ | ☐ |
| Variante | ☐ | ☐ | ☐ |
| Teste final + simulação | ☐ | ☐ | ☐ |
| Gráficos e `resumo.md` | ☐ | ☐ | ☐ |
| Slides | ☐ | ☐ | ☐ |

Comparação consolidada: `resultados/comparacao/` · Apresentação: 03/11/2026.
