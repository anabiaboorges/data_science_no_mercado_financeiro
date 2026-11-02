# Frente estatística – Itamar

Modelo: **ARIMA(1, 0, 1)** na grade candle × janela; variantes **ARIMAX / SARIMA / SARIMAX** (WDO como exógena e sazonalidade intradiária) na melhor combinação. Saídas em `resultados/estatistico/`. Ver Seção 3.1 do [plano](../../planos/PLANO_FRENTES_02.11.md).

Relatório completo (decisões, resultados, bandas de previsão e roteiro para a apresentação): [`resultados/estatistico/relatorio.md`](../../resultados/estatistico/relatorio.md).

## Notebooks (rodar nesta ordem)

| Notebook | O que faz | Principais saídas |
|---|---|---|
| `frente_estatistico.ipynb` | Protocolo do plano (Etapas A–F): grade candle × janela, regra, variantes, teste oficial (01/2025–03/2026). Etapa G (extra): parâmetros estimados e bandas de previsão | `config_final.json`, `grade_val.csv`, `variante_val.csv`, `teste_final.csv`, `resumo.md`, `bandas_por_hora_teste.png` |
| `cone_previsao_arima.ipynb` | Exemplo ilustrativo, fora do protocolo: cone de previsão multi-passos (ordem por AIC), candle diário e 60 min | `cone_arima_diario.png`, `cone_arima_60min.png` |
| `sensibilidade_retreino.ipynb` | Re-treino a cada 1, 3, 6 e 12 meses × janela × candle (48 combinações), só na validação | `sensibilidade_retreino/grade_retreino_val.csv`, heatmaps |

Tempo total: ~6 min (o protocolo leva ~1,5 min). Os arquivos de previsão de **1 min** (`resultados/estatistico/previsoes/*_1min_*`, ~66 MB) não vão para o repositório; são recriados ao rodar `frente_estatistico.ipynb`. Os de 15 e 60 min estão versionados.

## Ajustes desta frente em relação ao plano comum

1. **Período sazonal efetivo `s − 1`:** o `montar_base` calcula `s` (38 em 15 min, 10 em 60 min) **antes** de descartar o último candle do dia, então cada pregão tem `s − 1` linhas na série. O SARIMAX foi rodado com os dois valores. A autocorrelação do |retorno| confirma o `s − 1`: 0,149 na defasagem 9 contra 0,044 na 10.
2. **SARIMA sem exógena** como variante extra, para separar o efeito da sazonalidade do efeito do WDO.
3. **Critério da variante fixado antes do teste:** maior esperança com a melhor regra na validação (empate → ARIMA). Cada linha traz o **erro-padrão** da esperança.
4. **Etapa G (descritiva, depois do teste):** parâmetros estimados por trimestre e cobertura das bandas de previsão do SARIMAX. Não altera nenhuma escolha.

## Resultados (custo de 5 pts por operação)

| Abordagem | Período | Esperança (pts/op) | Leitura |
|---|---|---|---|
| Grade ARIMA(1,0,1), 12 combinações, opera sempre | Validação 2024 | −1,4 a −18,7 | Theil U ≥ 1 em todas: nenhuma bate o naive. Melhor: 60 min × 6 meses |
| Variantes em 60 min × 6 m, cada uma com a melhor regra | Validação 2024 | ARIMAX +10,2 (t = 0,55); ARIMA +1,4; SARIMA −0,7 | A ARIMAX vence por ruído de seleção (180 combinações testadas) e tem Theil U pior |
| **Oficial: ARIMAX, 60 min, 6 meses, limiar 90%, TP 1×, sem stop** | **Teste 01/2025–03/2026** | **−41,1 (t = −1,0)** | Theil U 1,0027. Acerta 52% das operações, mas o payoff é 0,76: sem stop, as perdas correm (pior: −2.625 pts) |
| Sinais aleatórios, mesma regra | Teste | −18,1 | Perde menos por operação que o modelo |
| Bandas de 95% de 1 passo | Teste | – | Cobertura de 92% no total, mas 74% às 10h e 99% às 18h: largura fixa, volatilidade não |
| Cone multi-passos (ordem por AIC) | Diário e 60 min | – | O AIC escolhe ARIMA(0,1,0), o passeio aleatório: centro plano, informação só na largura |
| Re-treino a cada 1, 3, 6, 12 meses | Validação 2024 | melhor −0,9 (t = −0,1) | A frequência não muda a conclusão; mantido o trimestral |

**Conclusão:** a família ARIMA não encontra estrutura linear explorável no retorno T+1 do WIN. Os parâmetros φ e θ se cancelam e o AIC escolhe o passeio aleatório, que é o próprio naive. O WDO não ajuda porque reage **junto** com o WIN (correlação −0,49 no mesmo candle e ≈ 0 com 1 candle de defasagem). A sazonalidade intradiária existe, mas está na **volatilidade**, não na direção. O que o SARIMAX oferece de útil é a banda de risco ("até onde o preço pode ir"), que só ficaria calibrada com um modelo de volatilidade (GARCH ou volatilidade por horário).

**Limitação comum às três frentes:** o CSV do WIN não tem pregões entre jun e 17/10/2023. Por isso, no 1º trimestre de 2024 as janelas de 3 e 6 meses usam os mesmos dados, e a janela de 12 meses teve ~7 meses na maior parte da validação. O teste não é afetado.
