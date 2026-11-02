# Resumo – Frente de ML (Random Forest)

- **Q1 – Melhor candle e janela:** 1min com 12 meses de treino. Critério: nenhuma com t ≥ 2: menor Theil U (0 de 12 combinações tiveram esperança positiva com t ≥ 2 na validação 2024). Modelo final: `rf_d4_f2` com 100 árvores, profundidade 4 e folha mínima de 2% do treino.
- **Q2 – Bate o naive?** No teste, Theil U = 0.9999 e R²_oos = 0.0002: a previsão é estatisticamente melhor que o naive (Clark-West t = 3.72), mas o ganho é minúsculo: o erro cai 0.01%. Na validação, 0 de 12 combinações tiveram Theil U < 1.
- **Q3 – Acertar mais é ganhar mais?** Na grade, a correlação entre acurácia direcional e esperança foi 0.85. No teste, a taxa de acerto das operações foi 42.6% com payoff 0.98.
- **Q4 – Regra e resultado no teste:** opera quando |previsão| passa o quantil 0%, take profit sem, stop loss sem. Critério: nenhuma regra com t ≥ 2: regra padrão (opera sempre, sem TP/SL). Esperança negativa: 0.59 / -4.41 / -9.41 pts por operação com custo 0 / 5 / 10 pts (t = -28.82 com 5 pts; aleatório: -4.83). Resultado com 5 pts: R$ -153,992 em 174579 operações.
- **Observação:** as features mais usadas foram corpo, ret_lag1, amplitude. A árvore simples teve Theil U 1.0012 contra 1.0000 da floresta base, em 6.32s contra 26.42s.
