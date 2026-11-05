"""Modelo de custo e de execução mais realista para a simulação de operações no WIN.

Proposta local da frente estatística (NOTAS.md, Seção 4.7), ainda NÃO adotada pelo grupo.
Não altera o `tt1_comum.py`: recebe as mesmas previsões (`prev`) e devolve operações no mesmo
formato, então `tt1_comum.resumo_operacoes` continua funcionando.

Diferenças em relação a `tt1_comum.simular`:
1. Custo por cenário de execução, derivado dos componentes (taxas da B3, spread, slippage),
   em vez de um valor fixo arbitrário.
2. Take profit é ordem limitada: só executa se o preço passar 1 tick além do alvo
   (tocar não garante execução, por causa da fila).
3. Stop vira ordem a mercado: no cenário conservador, escorrega 1 tick além do preço do stop.
4. Distâncias de TP/SL arredondadas para o tick do WIN (5 pontos).
5. Esperança bruta por operação = **custo de equilíbrio**: o maior custo que a estratégia aguenta.

Valores (1 ponto = R$ 0,20 por contrato; operação completa = entrada + saída):
- taxas da B3 em day trade ≈ R$ 0,50 → 2,5 pts (faixa de menor volume);
- spread ≈ 1 tick = 5 pts → ordens a mercado pagam meio spread na entrada e meio na saída;
- corretagem: zero na maioria das corretoras para minicontratos;
- imposto (20% do lucro mensal) não é custo por operação e fica de fora.
"""
import numpy as np
import pandas as pd

TICK_WIN = 5.0          # variação mínima de preço do WIN, em pontos
TAXAS_B3_PTS = 2.5      # R$ 0,50 por operação completa ÷ R$ 0,20 por ponto
SPREAD_PTS = 5.0        # 1 tick

CENARIOS = {
    # ordens limitadas na entrada e na saída: só taxas (risco de não executar fica fora do modelo)
    "otimista": dict(custo_pts=TAXAS_B3_PTS, tp_folga_pts=TICK_WIN, sl_slip_pts=0.0),
    # ordens a mercado: taxas + 1 spread (meio na entrada, meio na saída)
    "base": dict(custo_pts=TAXAS_B3_PTS + SPREAD_PTS, tp_folga_pts=TICK_WIN, sl_slip_pts=0.0),
    # mercado + 1 tick de slippage em toda operação + 1 tick extra quando o stop é acionado
    "conservador": dict(custo_pts=TAXAS_B3_PTS + SPREAD_PTS + TICK_WIN, tp_folga_pts=TICK_WIN, sl_slip_pts=TICK_WIN),
}
# reproduz exatamente o tt1_comum.simular (custo de 5 pts, TP/SL no preço exato, sem tick); só para comparação
LEGADO = dict(custo_pts=5.0, tp_folga_pts=0.0, sl_slip_pts=0.0, arredonda_tick=False)


def simular(prev, limiar=0.0, tp_mult=None, sl_mult=None, cenario="base", custo_pts=None,
            tp_folga_pts=None, sl_slip_pts=None, arredonda_tick=True):
    """Uma operação por candle, como no tt1_comum.simular: entra no fechamento de t e sai no fechamento
    de t+1, ou antes no TP/SL dentro do candle t+1. Se TP e SL podem ter sido tocados no mesmo candle,
    conta o stop. Parâmetros explícitos (custo_pts etc.) sobrepõem os do cenário.

    Devolve DataFrame com: lado (−1, 0, 1), saida ('tp', 'sl', 'fechamento' ou ''), bruto (pontos antes
    do custo fixo, já com o slippage do stop) e pontos (líquido)."""
    p = dict(CENARIOS[cenario]) if isinstance(cenario, str) else dict(cenario)
    for k, v in dict(custo_pts=custo_pts, tp_folga_pts=tp_folga_pts, sl_slip_pts=sl_slip_pts).items():
        if v is not None:
            p[k] = v
    arred = p.pop("arredonda_tick", arredonda_tick)

    lado = np.where(prev.y_pred > limiar, 1, np.where(prev.y_pred < -limiar, -1, 0))
    entrada = prev.close.values
    dist = prev.volat.values / 100 * entrada
    favor = np.where(lado == 1, prev.high_prox - entrada, entrada - prev.low_prox)
    contra = np.where(lado == 1, entrada - prev.low_prox, prev.high_prox - entrada)
    bruto = lado * (prev.close_prox.values - entrada)
    saida = np.where(lado != 0, "fechamento", "").astype(object)

    def no_tick(x):
        return np.maximum(np.round(x / TICK_WIN) * TICK_WIN, TICK_WIN) if arred else x

    if tp_mult is not None:
        tp = no_tick(tp_mult * dist)
        bate_tp = favor >= tp + p["tp_folga_pts"]           # limitada: precisa negociar além do alvo
        bruto = np.where(bate_tp, tp, bruto)
        saida = np.where(bate_tp & (lado != 0), "tp", saida)
    if sl_mult is not None:
        sl = no_tick(sl_mult * dist)
        bate_sl = contra >= sl                                # stop dispara ao tocar
        bruto = np.where(bate_sl, -sl - p["sl_slip_pts"], bruto)   # e executa pior (ordem a mercado)
        saida = np.where(bate_sl & (lado != 0), "sl", saida)
    bruto = np.where(lado != 0, bruto, 0.0)
    pontos = np.where(lado != 0, bruto - p["custo_pts"], 0.0)
    return pd.DataFrame({"lado": lado, "saida": saida, "bruto": bruto, "pontos": pontos}, index=prev.index)


def resumo(ops):
    """Resumo no formato do tt1_comum.resumo_operacoes, mais:
    erro-padrão e t da esperança, custo de equilíbrio (esperança bruta) e fração de saídas por TP/SL."""
    import tt1_comum as t
    r = t.resumo_operacoes(ops)
    o = ops[ops.lado != 0]
    n = len(o)
    if n == 0:
        return dict(r, ep_esperanca=np.nan, t_esperanca=np.nan, custo_equilibrio_pts=np.nan,
                    ep_custo_equilibrio=np.nan, frac_tp=np.nan, frac_sl=np.nan)
    ep = o.pontos.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(r, ep_esperanca=ep, t_esperanca=r["esperanca_pts"] / ep if ep else np.nan,
                custo_equilibrio_pts=o.bruto.mean(), ep_custo_equilibrio=o.bruto.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                frac_tp=(o.saida == "tp").mean(), frac_sl=(o.saida == "sl").mean())


def avaliar(prev, limiar=0.0, tp_mult=None, sl_mult=None, cenarios=("otimista", "base", "conservador")):
    """Uma linha por cenário. O custo de equilíbrio do cenário conservador é menor porque o stop escorrega."""
    return pd.DataFrame([dict(cenario=c, **resumo(simular(prev, limiar, tp_mult, sl_mult, c))) for c in cenarios])


def grade_regras(prev_validacao, cenario="base"):
    """Mesma grade do tt1_comum.grade_regras (limiar × TP × SL), com o novo modelo de execução.
    Ordena pelo custo de equilíbrio (critério que não depende do valor do custo) e mostra a esperança no cenário."""
    import tt1_comum as t
    linhas = []
    for q in t.LIMIAR_QUANTIS:
        lim = t.limiar_por_quantil(prev_validacao, q)
        for tp in t.TP_MULTS:
            for sl in t.SL_MULTS:
                linhas.append(dict(limiar_q=q, limiar=lim, tp_mult=tp, sl_mult=sl, cenario=cenario,
                                   **resumo(simular(prev_validacao, lim, tp, sl, cenario))))
    return pd.DataFrame(linhas).sort_values("custo_equilibrio_pts", ascending=False, na_position="last")
