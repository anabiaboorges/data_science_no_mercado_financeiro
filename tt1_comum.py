"""Módulo comum do trabalho T(t+1) – usado pelas três frentes (estatístico, ML e redes neurais).

Todas as frentes importam ESTE arquivo, sem alterar nada. É isso que garante que as bases,
as janelas, as métricas e a simulação de operações sejam idênticas entre os modelos.
Plano completo: PLANO_FRENTES_02.11.md
"""
import time
import numpy as np
import pandas as pd

# ---------- grade comum (decidida na reunião de 02/11) ----------
CANDLES = ["1min", "15min", "60min"]          # tamanho do candle
JANELAS_MESES = [1, 3, 6, 12]                 # janela de treino deslizante (meses para trás)
VALIDACAO = ("2024-01-01", "2025-01-01")      # escolha de candle, janela e regra de operação
TESTE = ("2025-01-01", "2026-04-01")          # rodado UMA vez, com tudo congelado
RETREINO = "QS"                               # re-treino trimestral
PONTO_WIN = 0.20                              # R$ por ponto por contrato (WIN)
CUSTOS_PTS = [0, 5, 10]                       # custo ida+volta em pontos (sensibilidade)

# ---------- grade comum da regra de operação ----------
LIMIAR_QUANTIS = [0.0, 0.5, 0.75, 0.9]        # opera só quando |previsão| passa este quantil
TP_MULTS = [None, 1.0, 2.0]                   # take profit em múltiplos da volatilidade recente
SL_MULTS = [None, 1.0, 2.0]                   # stop loss em múltiplos da volatilidade recente


def carregar(caminho):
    df = pd.read_csv(caminho, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    return df.set_index("datetime").drop(columns=["date", "time", "spread"])


def reamostrar(df, regra):
    if regra == "1min":
        return df.copy()
    return (df.resample(regra)
              .agg({"open": "first", "high": "max", "low": "min", "close": "last",
                    "tickvol": "sum", "vol": "sum"})
              .dropna())


def montar_base(win, wdo, regra, n_lags=10):
    """Uma linha por candle t: features conhecidas em t + alvo = log-retorno de t+1 (em %).
    Também guarda a máxima/mínima/fechamento do candle t+1, usados só na simulação."""
    win_r, wdo_r = reamostrar(win, regra), reamostrar(wdo, regra)
    df = win_r.join(wdo_r[["close", "vol"]].add_suffix("_wdo"), how="inner")
    dia = pd.Series(df.index.date, index=df.index)
    novo_dia = dia != dia.shift()
    df["ret"] = np.log(df["close"]).diff() * 100
    df["ret_wdo"] = np.log(df["close_wdo"]).diff() * 100
    # 1º candle do dia: retorno intracandle (abertura -> fechamento), sem o gap noturno
    df.loc[novo_dia, "ret"] = np.log(df["close"] / df["open"])[novo_dia] * 100
    df.loc[novo_dia, "ret_wdo"] = 0.0
    df["alvo"] = df["ret"].shift(-1)
    ultimo = novo_dia.shift(-1, fill_value=True)
    df.loc[ultimo, "alvo"] = np.nan                                  # último candle do dia: sem alvo
    for c in ["high", "low", "close"]:
        df[f"{c}_prox"] = df[c].shift(-1)
    s = int(dia.groupby(dia).size().median())                         # candles por pregão
    for k in range(1, n_lags + 1):
        df[f"ret_lag{k}"] = df["ret"].shift(k - 1)
    df["ret_wdo_lag1"] = df["ret_wdo"]
    df["amplitude"] = (df["high"] - df["low"]) / df["close"] * 100
    df["corpo"] = (df["close"] - df["open"]) / df["close"] * 100
    df["vol_rel"] = df["vol"] / df["vol"].rolling(s, min_periods=max(s // 2, 1)).mean()
    df["volat"] = df["ret"].rolling(s, min_periods=max(s // 2, 2)).std()
    df["hora"] = df.index.hour + df.index.minute / 60
    df["candle_do_dia"] = dia.groupby(dia).cumcount()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, s


def features(n_lags=10):
    """Conjuntos de features comuns. LAGS = só o passado do próprio retorno."""
    lags = [f"ret_lag{k}" for k in range(1, n_lags + 1)]
    return {"LAGS": lags,
            "COMPLETO": lags + ["amplitude", "corpo", "vol_rel", "volat", "hora", "candle_do_dia"]}


# ---------- modelos de referência ----------
class Naive:
    """Retorno previsto = 0 (equivale a P(t+1) = P(t)). É a régua de todos os modelos."""
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


# ---------- walk-forward comum ----------
def walk_forward(df, feats, fabrica, inicio, fim, janela_meses, freq=RETREINO, embargo=1):
    """Re-treina no início de cada trimestre usando só os últimos `janela_meses` meses (janela deslizante).
    Modelos com atributo `usa_y_teste = True` (ARIMA/SARIMAX) recebem o y do teste para atualizar o estado
    sem re-estimar parâmetros: a previsão de t+1 continua usando só dados até t."""
    cortes = list(pd.date_range(inicio, fim, freq=freq)) + [pd.Timestamp(fim)]
    cortes = sorted(set(cortes))
    blocos = []
    for ini, fim_bloco in zip(cortes[:-1], cortes[1:]):
        tr = df[(df.index >= ini - pd.DateOffset(months=janela_meses)) & (df.index < ini)].iloc[:-embargo]
        te = df[(df.index >= ini) & (df.index < fim_bloco)]
        if te.empty or tr.empty:
            continue
        m = fabrica().fit(tr[feats], tr["alvo"])
        if getattr(m, "usa_y_teste", False):
            p = m.predict(te[feats], te["alvo"])
        else:
            p = m.predict(te[feats])
        blocos.append(pd.DataFrame({"y_true": te["alvo"], "y_pred": np.asarray(p, dtype=float),
                                    "close": te["close"], "high_prox": te["high_prox"],
                                    "low_prox": te["low_prox"], "close_prox": te["close_prox"],
                                    "volat": te["volat"]}, index=te.index))
    return pd.concat(blocos)


def metricas(prev):
    """Métricas de previsão, sempre contra o naive (retorno previsto = 0)."""
    e = prev.y_true - prev.y_pred
    rmse, rmse_naive = np.sqrt((e ** 2).mean()), np.sqrt((prev.y_true ** 2).mean())
    nz = (prev.y_true != 0) & (prev.y_pred != 0)
    return dict(n=len(prev), RMSE=rmse, MAE=e.abs().mean(), TheilU=rmse / rmse_naive,
                R2_oos=1 - (e ** 2).sum() / (prev.y_true ** 2).sum(),
                acc_dir=(np.sign(prev.y_pred[nz]) == np.sign(prev.y_true[nz])).mean() if nz.any() else np.nan,
                corr=np.corrcoef(prev.y_true, prev.y_pred)[0, 1] if prev.y_pred.std() > 0 else np.nan)


# ---------- simulação de operações (comum a todas as frentes) ----------
def limiar_por_quantil(prev_validacao, q):
    """Converte um quantil de |previsão| (medido na VALIDAÇÃO) em limiar absoluto, em %."""
    return float(np.quantile(np.abs(prev_validacao.y_pred), q)) if q > 0 else 0.0


def simular(prev, limiar=0.0, tp_mult=None, sl_mult=None, custo_pts=0.0):
    """Uma operação por candle: entra no fechamento de t, sai no fechamento de t+1,
    ou antes se tocar o take profit / stop loss dentro do candle t+1.
    Compra se previsão > +limiar; vende se previsão < −limiar; senão fica de fora.
    TP/SL = múltiplo da volatilidade recente (desvio-padrão do retorno, em %) × preço.
    Se TP e SL forem tocados no mesmo candle, assume o STOP (hipótese conservadora)."""
    lado = np.where(prev.y_pred > limiar, 1, np.where(prev.y_pred < -limiar, -1, 0))
    entrada = prev.close.values
    dist = prev.volat.values / 100 * entrada
    favor = np.where(lado == 1, prev.high_prox - entrada, entrada - prev.low_prox)   # melhor excursão
    contra = np.where(lado == 1, entrada - prev.low_prox, prev.high_prox - entrada)  # pior excursão
    pts = lado * (prev.close_prox.values - entrada)
    if tp_mult is not None:
        tp = tp_mult * dist
        pts = np.where(favor >= tp, tp, pts)
    if sl_mult is not None:
        sl = sl_mult * dist
        pts = np.where(contra >= sl, -sl, pts)                         # stop tem prioridade
    pts = np.where(lado != 0, pts - custo_pts, 0.0)
    ops = pd.DataFrame({"lado": lado, "pontos": pts}, index=prev.index)
    return ops


def resumo_operacoes(ops):
    o = ops[ops.lado != 0].pontos
    if o.empty:
        return dict(n_ops=0, taxa_acerto=np.nan, ganho_medio=np.nan, perda_media=np.nan, payoff=np.nan,
                    esperanca_pts=np.nan, fator_lucro=np.nan, resultado_pts=0.0, resultado_R=0.0,
                    drawdown_max_pts=0.0)
    ganhos, perdas = o[o > 0], o[o < 0]
    acum = o.cumsum()
    return dict(n_ops=len(o), taxa_acerto=(o > 0).mean(),
                ganho_medio=ganhos.mean() if len(ganhos) else 0.0,
                perda_media=perdas.mean() if len(perdas) else 0.0,
                payoff=(ganhos.mean() / -perdas.mean()) if len(ganhos) and len(perdas) else np.nan,
                esperanca_pts=o.mean(),
                fator_lucro=(ganhos.sum() / -perdas.sum()) if len(perdas) else np.inf,
                resultado_pts=o.sum(), resultado_R=o.sum() * PONTO_WIN,
                drawdown_max_pts=(acum.cummax() - acum).max())


def grade_regras(prev_validacao, custo_pts=5):
    """Testa todas as combinações de limiar × TP × SL na VALIDAÇÃO e devolve a tabela ordenada
    pela esperança matemática por operação (critério de escolha comum a todas as frentes)."""
    linhas = []
    for q in LIMIAR_QUANTIS:
        lim = limiar_por_quantil(prev_validacao, q)
        for tp in TP_MULTS:
            for sl in SL_MULTS:
                r = resumo_operacoes(simular(prev_validacao, lim, tp, sl, custo_pts))
                linhas.append(dict(limiar_q=q, limiar=lim, tp_mult=tp, sl_mult=sl, custo_pts=custo_pts, **r))
    return pd.DataFrame(linhas).sort_values("esperanca_pts", ascending=False, na_position="last")


def sinais_aleatorios(prev, seed=42):
    """Referência para a simulação: mesmo número de candles, direção sorteada."""
    rng = np.random.default_rng(seed)
    return prev.assign(y_pred=rng.choice([-1.0, 1.0], size=len(prev)))


# ---------- rodada da grade candle × janela ----------
def rodar_grade(win, wdo, nome_modelo, fabrica, feats_nome, pasta_saida, periodo=VALIDACAO,
                candles=CANDLES, janelas=JANELAS_MESES, ativo="WIN"):
    """Roda o modelo em todas as combinações candle × janela e salva previsões + tabela-resumo.
    Mesmo formato de saída para as três frentes."""
    import os
    os.makedirs(f"{pasta_saida}/previsoes", exist_ok=True)
    rotulo = "val" if periodo == VALIDACAO else "teste"
    linhas = []
    for candle in candles:
        df, s = montar_base(win, wdo, candle)
        feats = features()[feats_nome]
        for jm in janelas:
            t0 = time.time()
            prev = walk_forward(df, feats, fabrica, *periodo, janela_meses=jm)
            seg = time.time() - t0
            prev.to_csv(f"{pasta_saida}/previsoes/{ativo}_{candle}_{jm}m_{nome_modelo}_{rotulo}.csv")
            sim = resumo_operacoes(simular(prev, custo_pts=5))
            linhas.append(dict(modelo=nome_modelo, candle=candle, janela_meses=jm, tempo_s=round(seg, 1),
                               **metricas(prev), **{f"sim_{k}": v for k, v in sim.items()}))
            print(nome_modelo, candle, f"{jm}m", {k: round(float(v), 4) for k, v in metricas(prev).items()},
                  f"{seg:.0f}s")
    tab = pd.DataFrame(linhas)
    tab.to_csv(f"{pasta_saida}/grade_{rotulo}.csv", index=False)
    return tab


def heatmap(tab, coluna, titulo, caminho, fmt=".4f", menor_melhor=False):
    """Heatmap candle × janela de uma métrica da grade (mesmo layout para todas as frentes)."""
    import matplotlib.pyplot as plt
    piv = tab.pivot(index="candle", columns="janela_meses", values=coluna).reindex(
        [c for c in CANDLES if c in set(tab.candle)])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r" if menor_melhor else "RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [f"{c} m" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, format(piv.values[i, j], fmt), ha="center", va="center", fontsize=9)
    ax.set_xlabel("janela de treino"); ax.set_ylabel("candle"); ax.set_title(titulo)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)
