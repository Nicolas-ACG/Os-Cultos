"""
Núcleo do projeto de index tracking (S&P 100 e Ibovespa) com algoritmo genético.

Este módulo é a parte "sem tela" do Projeto_Final: carregamento/limpeza dos dados,
otimização de pesos, algoritmo genético, backtest walk-forward e métricas.
A interface Streamlit está em app.py.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ─────────────────────────── Parâmetros do projeto ───────────────────────────
INICIO, FIM = '2018-01-01', '2025-05-01'      # período fixo (exigência da banca)
TRAIN_END = '2022-12-31'
VAL_START, VAL_END = '2023-01-01', '2023-12-31'
TEST_START = '2024-01-01'
VAL_INI, VAL_FIM = '2021-01-01', '2023-12-31'  # janela de validação usada na busca de hiperparâmetros
CAP = 0.10                                     # teto de peso por ação
ANO = 252
LIMITE_REMOCAO = 20                            # % máx. de dados faltantes por ação
LIMITE_OUTLIER = 0.15                          # winsorização dos retornos (±15% ao dia)

BASE_DIR = Path(os.environ.get('PROJETO_DIR', Path(__file__).resolve().parent))
DATA_DIR = BASE_DIR / 'data'
RES_DIR = BASE_DIR / 'resultados'

# ───────────────────────── Universos e setores (copiados do notebook) ─────────────────────────
tickers_sp100 = ['AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'ANET', 'AVGO', 'AXP', 'BA', 'BAC', 'BKNG', 'BLK', 'BMY', 'BNY', 'BRK-B', 'C', 'CAT', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'DE', 'DELL', 'DHR', 'DIS', 'DUK', 'EMR', 'FDX', 'GD', 'GE', 'GEV', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'INTU', 'ISRG', 'JNJ', 'JPM', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX', 'MA', 'MCD', 'MDLZ', 'MDT', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NFLX', 'NOW', 'NVDA', 'ORCL', 'PANW', 'PEP', 'PFE', 'PG', 'PLTR', 'PM', 'QCOM', 'RTX', 'SBUX', 'SCHW', 'SNDK', 'SO', 'T', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UBER', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WFC', 'WMT', 'XOM']
tickers_ibov = ['ABEV3.SA', 'ALOS3.SA', 'ASAI3.SA', 'AURE3.SA', 'AXIA3.SA', 'AZZA3.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BEEF3.SA', 'BPAC11.SA', 'BRAP4.SA', 'BRAV3.SA', 'CEAB3.SA', 'CMIG4.SA', 'CMIN3.SA', 'COGN3.SA', 'CPFE3.SA', 'CPLE3.SA', 'CSAN3.SA', 'CSMG3.SA', 'CSNA3.SA', 'CURY3.SA', 'CXSE3.SA', 'CYRE3.SA', 'DIRR3.SA', 'EGIE3.SA', 'EMBJ3.SA', 'ENEV3.SA', 'ENGI11.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GOAU4.SA', 'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'ISAE4.SA', 'ITSA4.SA', 'ITUB4.SA', 'KLBN11.SA', 'LREN3.SA', 'MBRF3.SA', 'MGLU3.SA', 'MOTV3.SA', 'MRVE3.SA', 'MULT3.SA', 'NATU3.SA', 'PETR3.SA', 'PETR4.SA', 'POMO4.SA', 'PRIO3.SA', 'PSSA3.SA', 'RADL3.SA', 'RAIL3.SA', 'RDOR3.SA', 'RENT3.SA', 'SANB11.SA', 'SBSP3.SA', 'SMFT3.SA', 'SUZB3.SA', 'TAEE11.SA', 'TEND3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VAMO3.SA', 'VBBR3.SA', 'VIVA3.SA', 'VIVT3.SA', 'WEGE3.SA', 'YDUQ3.SA']
setores_sp100 = {
    'AAPL': 'Tecnologia da Informação', 'ABBV': 'Assistência médica', 'ABT': 'Assistência médica',
    'ACN': 'Tecnologia da Informação', 'ADBE': 'Tecnologia da Informação', 'AMAT': 'Tecnologia da Informação',
    'AMD': 'Tecnologia da Informação', 'AMGN': 'Assistência médica', 'AMT': 'Imobiliária',
    'AMZN': 'Consumo discricionário', 'ANET': 'Tecnologia da Informação', 'AVGO': 'Tecnologia da Informação',
    'AXP': 'Finanças', 'BA': 'Industriais', 'BAC': 'Finanças', 'BKNG': 'Consumo discricionário',
    'BLK': 'Finanças', 'BMY': 'Assistência médica', 'BNY': 'Finanças', 'BRK-B': 'Finanças',
    'C': 'Finanças', 'CAT': 'Industriais', 'CMCSA': 'Serviços de comunicação', 'COF': 'Finanças',
    'COP': 'Energia', 'COST': 'Bens de consumo essenciais', 'CRM': 'Tecnologia da Informação',
    'CSCO': 'Tecnologia da Informação', 'CVS': 'Assistência médica', 'CVX': 'Energia',
    'DE': 'Industriais', 'DELL': 'Tecnologia da Informação', 'DHR': 'Assistência médica',
    'DIS': 'Serviços de comunicação', 'DUK': 'Serviços públicos', 'EMR': 'Industriais',
    'FDX': 'Industriais', 'GD': 'Industriais', 'GE': 'Industriais', 'GEV': 'Industriais',
    'GILD': 'Assistência médica', 'GM': 'Consumo discricionário', 'GOOG': 'Serviços de comunicação',
    'GOOGL': 'Serviços de comunicação', 'GS': 'Finanças', 'HD': 'Consumo discricionário',
    'IBM': 'Tecnologia da Informação', 'INTC': 'Tecnologia da Informação', 'INTU': 'Tecnologia da Informação',
    'ISRG': 'Assistência médica', 'JNJ': 'Assistência médica', 'JPM': 'Finanças',
    'KO': 'Bens de consumo essenciais', 'LIN': 'Materiais', 'LLY': 'Assistência médica',
    'LMT': 'Industriais', 'LOW': 'Consumo discricionário', 'LRCX': 'Tecnologia da Informação',
    'MA': 'Finanças', 'MCD': 'Consumo discricionário', 'MDLZ': 'Bens de consumo essenciais',
    'MDT': 'Assistência médica', 'META': 'Serviços de comunicação', 'MMM': 'Industriais',
    'MO': 'Bens de consumo essenciais', 'MRK': 'Assistência médica', 'MS': 'Finanças',
    'MSFT': 'Tecnologia da Informação', 'MU': 'Tecnologia da Informação', 'NEE': 'Serviços públicos',
    'NFLX': 'Serviços de comunicação', 'NOW': 'Tecnologia da Informação', 'NVDA': 'Tecnologia da Informação',
    'ORCL': 'Tecnologia da Informação', 'PANW': 'Tecnologia da Informação', 'PEP': 'Bens de consumo essenciais',
    'PFE': 'Assistência médica', 'PG': 'Bens de consumo essenciais', 'PLTR': 'Tecnologia da Informação',
    'PM': 'Bens de consumo essenciais', 'QCOM': 'Tecnologia da Informação', 'RTX': 'Industriais',
    'SBUX': 'Consumo discricionário', 'SCHW': 'Finanças', 'SNDK': 'Tecnologia da Informação',
    'SO': 'Serviços públicos', 'T': 'Serviços de comunicação', 'TMO': 'Assistência médica',
    'TMUS': 'Serviços de comunicação', 'TSLA': 'Consumo discricionário', 'TXN': 'Tecnologia da Informação',
    'UBER': 'Industriais', 'UNH': 'Assistência médica', 'UNP': 'Industriais', 'UPS': 'Industriais',
    'USB': 'Finanças', 'V': 'Finanças', 'VZ': 'Serviços de comunicação', 'WFC': 'Finanças',
    'WMT': 'Bens de consumo essenciais', 'XOM': 'Energia',
}
setores_ibov = {
    'ABEV3.SA': 'Bens de consumo essenciais', 'ALOS3.SA': 'Imobiliária', 'ASAI3.SA': 'Bens de consumo essenciais',
    'AURE3.SA': 'Serviços públicos', 'AXIA3.SA': 'Serviços públicos', 'AZZA3.SA': 'Consumo discricionário',
    'B3SA3.SA': 'Finanças', 'BBAS3.SA': 'Finanças', 'BBDC3.SA': 'Finanças', 'BBDC4.SA': 'Finanças',
    'BBSE3.SA': 'Finanças', 'BEEF3.SA': 'Bens de consumo essenciais', 'BPAC11.SA': 'Finanças',
    'BRAP4.SA': 'Materiais', 'BRAV3.SA': 'Energia', 'CEAB3.SA': 'Consumo discricionário',
    'CMIG4.SA': 'Serviços públicos', 'CMIN3.SA': 'Materiais', 'COGN3.SA': 'Consumo discricionário',
    'CPFE3.SA': 'Serviços públicos', 'CPLE3.SA': 'Serviços públicos', 'CSAN3.SA': 'Energia',
    'CSMG3.SA': 'Serviços públicos', 'CSNA3.SA': 'Materiais', 'CURY3.SA': 'Imobiliária',
    'CXSE3.SA': 'Finanças', 'CYRE3.SA': 'Imobiliária', 'DIRR3.SA': 'Imobiliária', 'EGIE3.SA': 'Serviços públicos',
    'EMBJ3.SA': 'Industriais', 'ENEV3.SA': 'Serviços públicos', 'ENGI11.SA': 'Serviços públicos',
    'EQTL3.SA': 'Serviços públicos', 'FLRY3.SA': 'Assistência médica', 'GGBR4.SA': 'Materiais',
    'GOAU4.SA': 'Materiais', 'HAPV3.SA': 'Assistência médica', 'HYPE3.SA': 'Assistência médica',
    'IGTI11.SA': 'Imobiliária', 'ISAE4.SA': 'Serviços públicos', 'ITSA4.SA': 'Finanças', 'ITUB4.SA': 'Finanças',
    'KLBN11.SA': 'Materiais', 'LREN3.SA': 'Consumo discricionário', 'MBRF3.SA': 'Bens de consumo essenciais',
    'MGLU3.SA': 'Consumo discricionário', 'MOTV3.SA': 'Industriais', 'MRVE3.SA': 'Imobiliária',
    'MULT3.SA': 'Imobiliária', 'NATU3.SA': 'Consumo discricionário', 'PETR3.SA': 'Energia', 'PETR4.SA': 'Energia',
    'POMO4.SA': 'Industriais', 'PRIO3.SA': 'Energia', 'PSSA3.SA': 'Finanças', 'RADL3.SA': 'Bens de consumo essenciais',
    'RAIL3.SA': 'Industriais', 'RDOR3.SA': 'Assistência médica', 'RENT3.SA': 'Consumo discricionário',
    'SANB11.SA': 'Finanças', 'SBSP3.SA': 'Serviços públicos', 'SMFT3.SA': 'Consumo discricionário',
    'SUZB3.SA': 'Materiais', 'TAEE11.SA': 'Serviços públicos', 'TEND3.SA': 'Imobiliária',
    'TIMS3.SA': 'Serviços de comunicação', 'TOTS3.SA': 'Tecnologia da Informação', 'UGPA3.SA': 'Energia',
    'USIM5.SA': 'Materiais', 'VALE3.SA': 'Materiais', 'VAMO3.SA': 'Industriais', 'VBBR3.SA': 'Energia',
    'VIVA3.SA': 'Consumo discricionário', 'VIVT3.SA': 'Serviços de comunicação', 'WEGE3.SA': 'Industriais',
    'YDUQ3.SA': 'Consumo discricionário',
}

INDICES = {
    'S&P 100': dict(tag='sp100', tickers=tickers_sp100, ticker_idx='^OEX',
                    arq_precos='v2_sp100_precos.csv', arq_idx='v2_sp100_indice.csv',
                    setores=setores_sp100, excluir=[]),
    'IBOVESPA': dict(tag='ibov', tickers=tickers_ibov, ticker_idx='^BVSP',
                     arq_precos='v2_ibov_precos.csv', arq_idx='v2_ibov_indice.csv',
                     setores=setores_ibov, excluir=['UGPA3.SA', 'POMO4.SA']),   # dados ruins
}

CFG_BASE = dict(
    K=20, cap=0.10,            # restrições do problema (fixas, não entram na busca)
    metodo='ag',               # 'ag' | 'guloso'
    janela_anos=3,             # janela de treino de cada rebalanceamento
    halflife=None,             # None = pesos iguais; ou meia-vida em dias (ex.: 126, 252)
    shrink=0.0,                # 0-1: encolhimento da covariância para correlação constante
    cv_blocos=0,               # 0 = fitness no treino todo; 4 = fitness por validação cruzada em blocos
    lam_turn=0.0,              # penaliza mudança de pesos (reduz turnover)
    gamma_troca=0.0,           # penaliza trocar de ações no AG (pp de TE por 100% de troca)
    freq_meses=6,              # frequência de rebalanceamento
    corr_min=0.50, limite_par=0.90,   # filtro de candidatas (corr_min=-1 desliga o filtro de correlação)
)
AG_FINAL = dict(pop=60, geracoes=60, paciencia=15)

# Configuração final de cada índice (mesma do notebook; se existir resultados/config_final.json, ele manda)
CFG_FINAL = {
    'S&P 100': dict(CFG_BASE),
    'IBOVESPA': {**CFG_BASE, 'halflife': 126, 'shrink': 0.1, 'corr_min': 0.3},
}


def carregar_cfg_final():
    arq = RES_DIR / 'config_final.json'
    cfg = {k: dict(v) for k, v in CFG_FINAL.items()}
    if arq.exists():
        try:
            salvo = json.loads(arq.read_text())
            for nome, tag in (('S&P 100', 'sp100'), ('IBOVESPA', 'ibov')):
                if tag in salvo:
                    cfg[nome] = {**CFG_BASE, **salvo[tag]}
        except Exception:
            pass
    return cfg


# ───────────────────────────── Dados ─────────────────────────────
def _baixar(tickers, nome_arquivo):
    """Lê do cache em data/; se não existir, baixa do Yahoo Finance (só preços de fechamento ajustados)."""
    arquivo = DATA_DIR / nome_arquivo
    if arquivo.exists():
        df = pd.read_csv(arquivo, index_col=0, parse_dates=True)
        if df.dropna(how='all').shape[0] > 0:
            return df
    import yfinance as yf
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = yf.download(tickers, start=INICIO, end=FIM, auto_adjust=True, progress=False)
    if df.dropna(how='all').empty:
        raise ValueError(f'Download vazio para {tickers}')
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    elif 'Close' in df.columns:
        df = df[['Close']]
    df.to_csv(arquivo)
    return df


def preparar(nome):
    """Carrega e limpa os dados de um índice. Mesmo pipeline do notebook:
    remove ações com >20% de faltantes, ffill, retornos log, winsorização em ±15% (só para AJUSTAR)."""
    c = INDICES[nome]
    precos = _baixar(c['tickers'], c['arq_precos'])
    indice = _baixar(c['ticker_idx'], c['arq_idx']).squeeze()
    precos.index = pd.to_datetime(precos.index)
    indice.index = pd.to_datetime(indice.index)

    pct_nulos = (precos.isnull().sum() / len(precos) * 100).round(2)
    validas = [t for t in pct_nulos[pct_nulos <= LIMITE_REMOCAO].index if t not in c['excluir']]
    removidas = [t for t in precos.columns if t not in validas]

    precos_clean = precos[validas].copy().ffill().dropna(how='all')
    ret_clean = np.log(precos_clean / precos_clean.shift(1)).dropna()
    ret_trata = ret_clean.clip(lower=-LIMITE_OUTLIER, upper=LIMITE_OUTLIER)
    ret_idx = np.log(indice / indice.shift(1)).dropna()

    # Xfit: retornos winsorizados (só para AJUSTAR). Xraw: retornos reais (para AVALIAR).
    datas = ret_clean.index.intersection(ret_idx.index)
    D = {'Xfit': ret_trata.loc[datas, ret_clean.columns],
         'Xraw': ret_clean.loc[datas],
         'y': ret_idx.loc[datas]}
    assert D['Xraw'].isnull().sum().sum() == 0 and D['y'].isnull().sum() == 0, 'Há NaN nos dados!'
    return dict(precos=precos, indice=indice, pct_nulos=pct_nulos, removidas=removidas,
                precos_clean=precos_clean, ret_clean=ret_clean, ret_idx=ret_idx, D=D)


def raio_x(ret_sp, ret_ib):
    """Tabela-resumo dos dois índices (retornos log diários)."""
    datas = ret_sp.index.intersection(ret_ib.index)
    sp_al, ib_al = ret_sp.loc[datas], ret_ib.loc[datas]

    def col(r_full, r_al):
        return [f'{r_al.mean() * ANO * 100:.2f}%',
                f'{r_al.std() * np.sqrt(ANO) * 100:.2f}%',
                f'{r_al.mean() / r_al.std() * np.sqrt(ANO):.2f}',
                f'{(np.exp(r_full.sum()) - 1) * 100:.1f}%',
                f'{(np.exp(r_al.min()) - 1) * 100:.2f}%',
                f'{(np.exp(r_al.max()) - 1) * 100:.2f}%']
    return pd.DataFrame({'S&P 100 (USD)': col(ret_sp, sp_al), 'IBOVESPA (BRL)': col(ret_ib, ib_al)},
                        index=['Retorno anualizado', 'Volatilidade anualizada', 'Índice de Sharpe',
                               'Retorno acumulado total', 'Pior dia', 'Melhor dia'])


def estatisticas_acoes(ret):
    return pd.DataFrame({
        'Retorno médio anualizado (%)': ret.mean() * ANO * 100,
        'Volatilidade anualizada (%)': ret.std() * np.sqrt(ANO) * 100,
        'Sharpe aprox. anualizado': ret.mean() / ret.std() * np.sqrt(ANO),
        'Retorno mínimo diário (%)': ret.min() * 100,
        'Retorno máximo diário (%)': ret.max() * 100,
        'Assimetria': ret.skew(),
        'Curtose': ret.kurtosis(),
    }).round(4)


def corr_beta(ret, ret_idx):
    """Correlação e beta de cada ação contra o índice (variância do índice no período completo)."""
    var_idx = ret_idx.var()
    return pd.DataFrame({'Correlação': ret.corrwith(ret_idx),
                         'Beta': ret.apply(lambda col: col.cov(ret_idx) / var_idx)})


# ───────────────────────── Otimização de pesos e métricas ─────────────────────────
def cap_viavel(cap, k):
    """Garante k*cap >= 1; senão a soma dos pesos nunca chega a 100% (problema inviável)."""
    return max(cap, 1.2 / k)


def resolver_pesos(S, c, cap):
    """Min w'Sw - 2c'w  s.t. 0 <= w <= cap, sum(w) = 1 (gradiente analítico)."""
    k = len(c)
    cap = cap_viavel(cap, k)
    res = minimize(lambda w: w @ S @ w - 2 * c @ w, np.full(k, 1.0 / k),
                   jac=lambda w: 2 * (S @ w - c), method='SLSQP',
                   bounds=[(0.0, cap)] * k,
                   constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
                                 'jac': lambda w: np.ones(k)}],
                   options={'ftol': 1e-12, 'maxiter': 500})
    w = np.clip(res.x, 0.0, cap)
    if (not np.all(np.isfinite(res.x))) or abs(res.x.sum() - 1) > 1e-4 or w.sum() <= 0:
        return np.full(k, 1.0 / k)          # fallback seguro
    return w / w.sum()


def pre_computar(X, y):
    """Covariâncias (dados centrados, em %) calculadas UMA vez: S=cov(X), c=cov(X,y), bb=var(y)."""
    A = (X.values - X.values.mean(axis=0)) * 100
    b = (y.values - y.values.mean()) * 100
    n = len(b)
    return A.T @ A / n, A.T @ b / n, float(b @ b / n)


def te_de(S, c, bb, ix, cap):
    """TE anualizado (%) e pesos ótimos para o subconjunto ix."""
    ix = list(ix)
    Ss, cs = S[np.ix_(ix, ix)], c[ix]
    w = resolver_pesos(Ss, cs, cap)
    var = w @ Ss @ w - 2 * cs @ w + bb
    return np.sqrt(max(var, 0.0) * ANO), w


def retorno_carteira(X_log, w):
    """Retorno log diário de carteira buy-and-hold (pesos w no início do bloco, sem rebalancear).
    Agrega em retorno SIMPLES entre ativos (log não é aditivo entre ativos)."""
    R = np.expm1(X_log[list(w.index)])
    v = (1 + R).cumprod().values @ w.values
    v_ant = np.concatenate([[1.0], v[:-1]])
    return pd.Series(np.log(v / v_ant), index=X_log.index)


def metricas(r, y):
    dif = r - y
    return {'TE (%)': dif.std() * np.sqrt(ANO) * 100,
            'TD (% a.a.)': dif.mean() * ANO * 100,
            'Corr': r.corr(y),
            'R²': 1 - dif.var() / y.var()}


def filtrar_candidatas(Xtr, ytr, corr_min=0.50, limite_par=0.90):
    ci = Xtr.corrwith(ytr)
    cand = ci[ci >= corr_min].sort_values(ascending=False).index.tolist()
    if len(cand) < 2:
        return cand
    C = np.abs(np.corrcoef(Xtr[cand].values, rowvar=False))
    mant = []
    for i in range(len(cand)):
        if all(C[i, j] < limite_par for j in mant):
            mant.append(i)
    return [cand[i] for i in mant]


# ───────────────────────── Algoritmo genético (versão simples, usada na validação de K) ─────────────────────────
def ag_selecao(S, c, bb, K, cap=CAP, pop=60, geracoes=60, p_cx=0.8, p_mut=0.3,
               n_elite=2, paciencia=15, seed=0):
    """AG para seleção de K ativos: elitismo, cruzamento por conjunto e mutação por troca (K sempre preservado)."""
    rng = np.random.default_rng(seed)
    N = len(c)
    if N <= K:
        return list(range(N)), te_de(S, c, bb, range(N), cap)[0], []
    cache = {}

    def fit(ind):
        if ind not in cache:
            cache[ind] = te_de(S, c, bb, ind, cap)[0]
        return cache[ind]

    def novo():
        return tuple(sorted(int(i) for i in rng.choice(N, K, replace=False)))

    def torneio(P, F, t=3):
        cand = rng.integers(0, len(P), t)
        return P[min(cand, key=lambda i: F[i])]

    def cruzar(a, b):
        sa, sb = set(a), set(b)
        comum = sa & sb
        resto = list(sa ^ sb)
        rng.shuffle(resto)
        return tuple(sorted(comum | set(resto[:K - len(comum)])))

    def mutar(ind):
        s = set(ind)
        for _ in range(int(rng.integers(1, 3))):
            sai = int(rng.choice(list(s)))
            livres = [j for j in range(N) if j not in s]
            s.remove(sai)
            s.add(int(rng.choice(livres)))
        return tuple(sorted(s))

    P = [novo() for _ in range(pop)]
    melhor, sem_melhora, hist = np.inf, 0, []
    for g in range(geracoes):
        F = [fit(i) for i in P]
        ordem = np.argsort(F)
        P, F = [P[i] for i in ordem], [F[i] for i in ordem]
        hist.append(F[0])
        if F[0] < melhor - 1e-4:
            melhor, sem_melhora = F[0], 0
        else:
            sem_melhora += 1
        if sem_melhora >= paciencia:
            break
        filhos = P[:n_elite]
        while len(filhos) < pop:
            a, b = torneio(P, F), torneio(P, F)
            f = cruzar(a, b) if rng.random() < p_cx else a
            if rng.random() < p_mut:
                f = mutar(f)
            filhos.append(f)
        P = filhos
    F = [fit(i) for i in P]
    i = int(np.argmin(F))
    return list(P[i]), F[i], hist


def selecao_gulosa(S, c, bb, K, cap=CAP):
    """Forward selection: a cada passo adiciona o ativo que mais reduz o TE."""
    sel, restante = [], list(range(len(c)))
    for _ in range(K):
        j = min(restante, key=lambda j: te_de(S, c, bb, sel + [j], cap)[0])
        sel.append(j)
        restante.remove(j)
    return sel


def selecionar(Xtr, ytr, K, cap=CAP, metodo='ag', seed=0, **kw):
    """metodo: 'ag' | 'guloso' | 'topcorr' | 'aleatorio'"""
    cand = filtrar_candidatas(Xtr, ytr)
    if len(cand) <= K:
        return [str(x) for x in cand]
    if metodo == 'topcorr':
        return [str(x) for x in cand[:K]]
    if metodo == 'aleatorio':
        rng = np.random.default_rng(seed)
        return [str(x) for x in rng.choice(cand, K, replace=False)]
    S, c, bb = pre_computar(Xtr[cand], ytr)
    ix = selecao_gulosa(S, c, bb, K, cap) if metodo == 'guloso' else ag_selecao(S, c, bb, K, cap, seed=seed, **kw)[0]
    return [str(cand[i]) for i in ix]


def pesos_para(Xtr, ytr, nomes, cap=CAP):
    S, c, bb = pre_computar(Xtr[nomes], ytr)
    return pd.Series(resolver_pesos(S, c, cap), index=nomes)


def avaliar_K_validacao(D, Ks=(8, 10, 15, 20, 25, 30), seeds=(0, 1, 2), cap=CAP, progresso=None):
    """TE na validação (2023) para cada K: AG (média ± dp entre sementes), guloso e top-correlação."""
    Xtr, ytr = D['Xfit'].loc[:TRAIN_END], D['y'].loc[:TRAIN_END]
    Xva, yva = D['Xraw'].loc[VAL_START:VAL_END], D['y'].loc[VAL_START:VAL_END]
    linhas = []
    for n, K in enumerate(Ks):
        te_ag = []
        for s in seeds:
            nomes = selecionar(Xtr, ytr, K, cap, 'ag', seed=s)
            te_ag.append(metricas(retorno_carteira(Xva, pesos_para(Xtr, ytr, nomes, cap)), yva)['TE (%)'])
        linha = {'K': K, 'AG (média)': np.mean(te_ag), 'AG (dp)': np.std(te_ag)}
        for met, rot in (('guloso', 'Guloso'), ('topcorr', 'Top-corr')):
            nomes = selecionar(Xtr, ytr, K, cap, met)
            linha[rot] = metricas(retorno_carteira(Xva, pesos_para(Xtr, ytr, nomes, cap)), yva)['TE (%)']
        linhas.append(linha)
        if progresso:
            progresso((n + 1) / len(Ks), f'K={K} concluído')
    return pd.DataFrame(linhas).set_index('K').round(2)


# ───────────────────────── Modelo configurável (AG com covariância suavizada) ─────────────────────────
def encolher(C, d):
    """Encolhe a covariância CONJUNTA (ações + índice) para um alvo de correlação constante.
    Mantém variâncias e a correlação média (fator de mercado), reduz só o ruído das correlações individuais."""
    if d <= 0:
        return C
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    n = len(C)
    rbar = (R.sum() - n) / (n * (n - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(C))
    return (1 - d) * C + d * F


def momentos(X, y, halflife=None, shrink=0.0):
    """Covariâncias ponderadas (meia-vida opcional) e com encolhimento opcional. Mesma escala da pre_computar."""
    Z = np.column_stack([X.values, y.values]) * 100
    T = len(Z)
    if halflife:
        w = 0.5 ** ((T - 1 - np.arange(T)) / halflife)
        w = w / w.sum()
    else:
        w = np.full(T, 1.0 / T)
    Zc = Z - w @ Z
    C = encolher((Zc * w[:, None]).T @ Zc, shrink)
    return C[:-1, :-1], C[:-1, -1], float(C[-1, -1])


def construir_te_fn(Xw, yw, cand, cfg):
    """Devolve te_fn(ind) -> TE anualizado (%) do subconjunto ind (índices de cand)."""
    cap, sh, F = cfg['cap'], cfg['shrink'], cfg['cv_blocos']
    if F and F >= 2:
        Z = np.column_stack([Xw[cand].values, yw.values]) * 100
        N, T = len(cand), len(Z)
        cortes = np.linspace(0, T, F + 1).astype(int)
        folds = []
        for j in range(F):
            Zj = Z[cortes[j]:cortes[j + 1]]
            folds.append((len(Zj), Zj.mean(0), Zj.T @ Zj / len(Zj)))
        m_all, M_all = Z.mean(0), Z.T @ Z / T

        def te_fn(ind):
            ia = list(ind) + [N]
            mt, Mt = m_all[ia], M_all[np.ix_(ia, ia)]
            vs = []
            for nj, mj, Mj in folds:
                n_tr = T - nj
                m = (T * mt - nj * mj[ia]) / n_tr                        # momentos do treino (demais blocos)
                M = (T * Mt - nj * Mj[np.ix_(ia, ia)]) / n_tr
                C = encolher(M - np.outer(m, m), sh)
                S, c = C[:-1, :-1], C[:-1, -1]
                w = resolver_pesos(S, c, cap)
                Cj = Mj[np.ix_(ia, ia)] - np.outer(mj[ia], mj[ia])       # avalia no bloco de fora
                v = w @ Cj[:-1, :-1] @ w - 2 * Cj[:-1, -1] @ w + Cj[-1, -1]
                vs.append(max(v, 0.0))
            return float(np.sqrt(np.mean(vs) * ANO))
        return te_fn
    S_all, c_all, bb = momentos(Xw[cand], yw, cfg['halflife'], sh)
    return lambda ind: te_de(S_all, c_all, bb, ind, cap)[0]


def ag_selecao2(te_fn, N, K, prev=None, gamma=0.0, pop=60, geracoes=60, p_cx=0.8, p_mut=0.3,
                n_elite=2, paciencia=15, seed=0):
    """AG com warm start na carteira anterior e custo de troca de ações."""
    rng = np.random.default_rng(seed)
    prev_set = set(prev) if prev else set()
    cache = {}

    def fit(ind):
        if ind not in cache:
            pen = gamma * len(set(ind) - prev_set) / K if prev_set else 0.0
            cache[ind] = te_fn(ind) + pen
        return cache[ind]

    def novo():
        return tuple(sorted(int(i) for i in rng.choice(N, K, replace=False)))

    def torneio(P, F, t=3):
        cand = rng.integers(0, len(P), t)
        return P[min(cand, key=lambda i: F[i])]

    def cruzar(a, b):
        sa, sb = set(a), set(b)
        comum = sa & sb
        resto = list(sa ^ sb)
        rng.shuffle(resto)
        return tuple(sorted(comum | set(resto[:K - len(comum)])))

    def mutar(ind):
        s = set(ind)
        for _ in range(int(rng.integers(1, 3))):
            sai = int(rng.choice(list(s)))
            livres = [j for j in range(N) if j not in s]
            s.remove(sai)
            s.add(int(rng.choice(livres)))
        return tuple(sorted(s))

    P = [novo() for _ in range(pop)]
    if prev_set:                                   # warm start: carteira anterior + variações
        base = set(list(prev_set)[:K])
        while len(base) < K:
            base.add(int(rng.integers(0, N)))
        base = tuple(sorted(base))
        P[0] = base
        for i in range(1, max(2, pop // 4)):
            P[i] = mutar(base)
    melhor, sem_melhora = np.inf, 0
    for g in range(geracoes):
        F = [fit(i) for i in P]
        ordem = np.argsort(F)
        P, F = [P[i] for i in ordem], [F[i] for i in ordem]
        if F[0] < melhor - 1e-4:
            melhor, sem_melhora = F[0], 0
        else:
            sem_melhora += 1
        if sem_melhora >= paciencia:
            break
        filhos = P[:n_elite]
        while len(filhos) < pop:
            a, b = torneio(P, F), torneio(P, F)
            f = cruzar(a, b) if rng.random() < p_cx else a
            if rng.random() < p_mut:
                f = mutar(f)
            filhos.append(f)
        P = filhos
    F = [fit(i) for i in P]
    return list(P[int(np.argmin(F))])


def guloso2(te_fn, N, K):
    sel, restante = [], list(range(N))
    for _ in range(K):
        j = min(restante, key=lambda j: te_fn(sel + [j]))
        sel.append(j)
        restante.remove(j)
    return sel


def selecionar_cfg(Xtr, ytr, cfg, prev_names, seed, ag_kw):
    cand = filtrar_candidatas(Xtr, ytr, cfg['corr_min'], cfg['limite_par'])
    K = cfg['K']
    if len(cand) <= K:
        return [str(x) for x in cand]
    te_fn = construir_te_fn(Xtr, ytr, cand, cfg)
    if cfg['metodo'] == 'guloso':
        ix = guloso2(te_fn, len(cand), K)
    else:
        prev_idx = [cand.index(n) for n in prev_names if n in cand] if prev_names else None
        ix = ag_selecao2(te_fn, len(cand), K, prev_idx, cfg['gamma_troca'], seed=seed, **ag_kw)
    return [str(cand[i]) for i in ix]


def pesos_cfg(Xtr, ytr, nomes, cfg, w_prev=None):
    S, c, bb = momentos(Xtr[nomes], ytr, cfg['halflife'], cfg['shrink'])
    lam = cfg['lam_turn'] * np.mean(np.diag(S))
    if w_prev is not None and lam > 0:                 # puxa para os pesos atuais (menos turnover)
        w0 = w_prev.reindex(nomes).fillna(0.0).values
        S = S + lam * np.eye(len(nomes))
        c = c + lam * w0
    return pd.Series(resolver_pesos(S, c, cfg['cap']), index=nomes)


def datas_rebalanceamento(datas, freq_meses):
    rebal = [datas[0]]
    while True:
        fut = datas[datas >= rebal[-1] + pd.DateOffset(months=freq_meses)]
        if len(fut) == 0:
            break
        rebal.append(fut[0])
    return rebal


def backtest_cfg(D, cfg, ini_av, fim_av=None, seed=0, ag_kw=None, progresso=None):
    """Backtest walk-forward: a cada rebalanceamento treina SÓ com dados anteriores, seleciona K ações (AG)
    e calcula os pesos; a carteira é avaliada nos retornos reais (Xraw) até o próximo rebalanceamento."""
    ag_kw = ag_kw or {}
    Xav = D['Xraw'].loc[ini_av:fim_av]
    rebal = datas_rebalanceamento(Xav.index, cfg['freq_meses'])
    rets, turns, hist, nomes, w_fim = [], [], [], None, None
    for i, ini in enumerate(rebal):
        fim = rebal[i + 1] if i + 1 < len(rebal) else None
        bloco = Xav.loc[ini:fim]
        if fim is not None:
            bloco = bloco.iloc[:-1]
        ini_tr = ini - pd.DateOffset(years=cfg['janela_anos'])
        Xtr = D['Xfit'].loc[ini_tr:ini].iloc[:-1]            # só dados ANTERIORES ao rebalanceamento
        ytr = D['y'].loc[ini_tr:ini].iloc[:-1]
        nomes = selecionar_cfg(Xtr, ytr, cfg, nomes, seed + i, ag_kw)
        w = pesos_cfg(Xtr, ytr, nomes, cfg, w_fim)
        if w_fim is not None:
            u = w.index.union(w_fim.index)
            t = 0.5 * (w.reindex(u, fill_value=0) - w_fim.reindex(u, fill_value=0)).abs().sum()
            turns.append((ini.year, t))
        rets.append(retorno_carteira(bloco, w))
        g = (1 + np.expm1(bloco[list(w.index)])).prod()
        w_fim = w * g
        w_fim = w_fim / w_fim.sum()
        hist.append(list(nomes))
        if progresso:
            progresso(i + 1, len(rebal))
    r = pd.concat(rets)
    y = D['y'].loc[r.index]
    dif = r - y
    te_ano = dif.groupby(dif.index.year).std() * np.sqrt(ANO) * 100
    tdf = pd.DataFrame(turns, columns=['ano', 't'])
    turn_ano = tdf.groupby('ano')['t'].sum().reindex(te_ano.index).fillna(0.0) if len(tdf) else te_ano * 0
    m = metricas(r, y)
    m['Turnover/rebal. (%)'] = (np.mean([t for _, t in turns]) * 100) if turns else np.nan
    return {'r': r, 'm': m, 'te_ano': te_ano, 'turn_ano': turn_ano, 'hist': hist, 'rebal': rebal}


def pesos_finais(D, cfg, res):
    """Pesos da carteira ATUALIZADOS para operar amanhã, usando a base de dados até o último dia disponível."""
    data_final = D['Xfit'].index[-1]
    ini_tr = data_final - pd.DateOffset(years=cfg['janela_anos'])
    Xtr = D['Xfit'].loc[ini_tr:data_final]
    ytr = D['y'].loc[ini_tr:data_final]
    return pesos_cfg(Xtr, ytr, res['hist'][-1], cfg).sort_values(ascending=False)


def ic_te_bloco(dif, bloco=21, n=1000, seed=0):
    """IC 95% (block bootstrap) do TE."""
    rng = np.random.default_rng(seed)
    d = dif.values
    T = len(d)
    nb = int(np.ceil(T / bloco))
    tes = []
    for _ in range(n):
        st = rng.integers(0, T - bloco + 1, nb)
        a = np.concatenate([d[s:s + bloco] for s in st])[:T]
        tes.append(a.std(ddof=1) * np.sqrt(ANO) * 100)
    return np.percentile(tes, [2.5, 97.5])
