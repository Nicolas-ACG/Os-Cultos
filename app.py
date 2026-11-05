"""
App Streamlit · Index tracking (S&P 100 e Ibovespa) com algoritmo genético.

Rodar:  streamlit run app.py
Precisa de modelo.py na mesma pasta e dos CSVs em data/ (se faltarem, são baixados do Yahoo Finance).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import modelo as M

st.set_page_config(page_title="Index Tracking · Algoritmo Genético", page_icon="📈", layout="wide")

COR = {'S&P 100': 'navy', 'IBOVESPA': 'darkgreen'}
PERIODOS = {
    f'Teste ({M.TEST_START[:7]} em diante)': (M.TEST_START, None),
    f'Validação ({M.VAL_INI[:4]}–{M.VAL_FIM[:4]})': (M.VAL_INI, M.VAL_FIM),
}
FMT = {'TE (%)': '{:.2f}', 'TD (% a.a.)': '{:.2f}', 'Corr': '{:.3f}', 'R²': '{:.3f}', 'Turnover/rebal. (%)': '{:.1f}'}


# ───────────────────────────── Cache ─────────────────────────────
@st.cache_data(show_spinner="Carregando e limpando os dados…")
def dados(nome):
    return M.preparar(nome)


@st.cache_data(show_spinner=False)
def rodar_backtest(nome, cfg, ag_kw, n_seeds, ini, fim):
    D = dados(nome)['D']
    return [M.backtest_cfg(D, cfg, ini, fim, seed=s * 100, ag_kw=ag_kw) for s in range(n_seeds)]


@st.cache_data(show_spinner=False)
def rodar_validacao_K(nome, Ks, n_seeds):
    return M.avaliar_K_validacao(dados(nome)['D'], Ks=Ks, seeds=tuple(range(n_seeds)))


def mostrar(fig):
    st.pyplot(fig)
    plt.close(fig)


def indice_de(opts, valor):
    return opts.index(valor) if valor in opts else 0


def ler_csv(arquivo, **kw):
    p = M.RES_DIR / arquivo
    return pd.read_csv(p, **kw) if p.exists() else None


# ───────────────────────────── Cabeçalho e dados ─────────────────────────────
st.title("📈 Index tracking com algoritmo genético")
st.caption(f"S&P 100 e Ibovespa · dados de {M.INICIO} a {M.FIM} · treino até {M.TRAIN_END[:4]}, "
           f"validação {M.VAL_START[:4]}, teste {M.TEST_START[:4]} em diante")

try:
    sp, ib = dados('S&P 100'), dados('IBOVESPA')
except Exception as e:
    st.error(f"Não consegui carregar os dados: {e}")
    st.info("Coloque os CSVs em `data/` (v2_sp100_precos.csv, v2_sp100_indice.csv, v2_ibov_precos.csv, "
            "v2_ibov_indice.csv) ou libere acesso à internet para baixar do Yahoo Finance.")
    st.stop()
DADOS = {'S&P 100': sp, 'IBOVESPA': ib}

with st.sidebar:
    st.header("Índice")
    nome = st.radio("Escolha o índice", list(M.INDICES), label_visibility="collapsed")
    d = DADOS[nome]
    st.metric("Ações elegíveis", d['precos_clean'].shape[1])
    st.metric("Pregões", len(d['D']['y']))
    st.markdown("---")
    st.markdown(
        "**Protocolo**\n\n"
        f"- Treino: até {M.TRAIN_END}\n"
        f"- Validação: {M.VAL_START[:4]}\n"
        f"- Teste (uso único): {M.TEST_START[:4]} em diante\n"
        f"- Teto de peso por ação: {M.CAP:.0%}\n"
        "- Rebalanceamento walk-forward, sem olhar o futuro"
    )

t_geral, t_dados, t_expl, t_bt, t_k, t_salvos = st.tabs(
    ["📊 Visão geral", "🧹 Dados", "🔎 Exploração", "⚙️ Backtest", "📐 Escolha de K", "📁 Resultados salvos"])

# ───────────────────────────── Visão geral ─────────────────────────────
with t_geral:
    st.subheader("Raio-X histórico: S&P 100 vs. IBOVESPA")
    st.dataframe(M.raio_x(sp['ret_idx'], ib['ret_idx']))

    c1, c2 = st.columns(2)
    for col, n_ in ((c1, 'S&P 100'), (c2, 'IBOVESPA')):
        with col:
            serie = DADOS[n_]['indice'] / DADOS[n_]['indice'].iloc[0] * 100
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(serie.index, serie, color=COR[n_], lw=2)
            ax.axvline(pd.Timestamp('2020-03-23'), color='red', ls=':', lw=1.5, label='Mínimo COVID')
            ax.axvline(pd.Timestamp('2022-01-03'), color='darkred', ls=':', lw=1.5, label='Início crise de juros')
            ax.set_title(f'{n_} (base 100)', fontweight='bold')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            mostrar(fig)

    st.subheader("Retorno por ano (%)")
    anual = pd.DataFrame({n_: (np.exp(DADOS[n_]['ret_idx'].resample('YE').sum()) - 1) * 100 for n_ in DADOS}).round(2)
    anual.index = anual.index.year
    st.dataframe(anual)

# ───────────────────────────── Dados ─────────────────────────────
with t_dados:
    st.subheader(f"Limpeza dos dados · {nome}")
    c = st.columns(4)
    c[0].metric("Ações no universo", d['precos'].shape[1])
    c[1].metric("Removidas", len(d['removidas']))
    c[2].metric("Mantidas", d['precos_clean'].shape[1])
    c[3].metric("Retornos diários", len(d['D']['y']))
    st.caption(f"Regra: ações com mais de {M.LIMITE_REMOCAO}% de dados faltantes são removidas; "
               "o restante é preenchido com o último valor disponível (forward fill).")

    completude = (1 - d['pct_nulos'] / 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.17 * len(completude))))
    completude.plot(kind='barh', color=COR[nome], ax=ax)
    ax.axvline(0.80, color='red', ls='--', label='80% de completude (limite de corte)')
    ax.set_title(f'Completude de dados por ação · {nome}')
    ax.set_xlabel('Proporção de dados presentes')
    ax.legend()
    fig.tight_layout()
    mostrar(fig)

    if d['removidas']:
        rem = d['pct_nulos'].reindex(d['removidas']).to_frame('% faltantes')
        rem['motivo'] = np.where(rem['% faltantes'] > M.LIMITE_REMOCAO,
                                 f'> {M.LIMITE_REMOCAO}% faltantes', 'exclusão manual (dados ruins)')
        st.markdown("**Ações removidas**")
        st.dataframe(rem.sort_values('% faltantes', ascending=False))

    st.markdown(f"**Retornos diários acima de ±{M.LIMITE_OUTLIER:.0%}** "
                "(winsorizados só para ajustar o modelo; a avaliação usa os retornos reais)")
    rc = d['ret_clean']
    out = rc[rc.abs() > M.LIMITE_OUTLIER].stack().dropna().reset_index()
    out.columns = ['Data', 'Ticker', 'Retorno_Log']
    st.write(f"Ocorrências: **{len(out)}**")
    if len(out):
        st.dataframe(out.sort_values('Retorno_Log', key=abs, ascending=False).head(15), hide_index=True)

# ───────────────────────────── Exploração ─────────────────────────────
with t_expl:
    st.subheader(f"Exploração · {nome}")
    X = d['D']['Xraw']
    cb = M.corr_beta(X, d['ret_idx'])
    corr = cb['Correlação'].sort_values(ascending=False)

    st.markdown("**Correlação de cada ação com o índice**")
    cores = ['steelblue' if v > 0.7 else ('orange' if v > 0.5 else 'tomato') for v in corr.values]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    corr.plot(kind='bar', color=cores, edgecolor='white', linewidth=.3, ax=ax)
    ax.axhline(0.7, color='green', ls='--', label='0,70')
    ax.axhline(0.5, color='orange', ls='--', label='0,50')
    ax.set_ylabel('Correlação de Pearson')
    ax.tick_params(axis='x', labelsize=6)
    ax.legend()
    fig.tight_layout()
    mostrar(fig)
    st.caption(f"{(corr > 0.7).mean() * 100:.1f}% das ações têm correlação > 0,70 com o índice.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Beta × correlação**")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(cb['Beta'], cb['Correlação'], color=COR[nome], s=50, alpha=.7)
        ax.axvline(1.0, color='red', ls='--', label='Beta = 1')
        ax.axhline(0.7, color='green', ls='--', label='Corr = 0,70')
        ax.set_xlabel('Beta')
        ax.set_ylabel('Correlação')
        ax.legend()
        fig.tight_layout()
        mostrar(fig)
    with c2:
        st.markdown("**Correlação por setor**")
        df_s = pd.DataFrame({'corr': corr, 'setor': corr.index.to_series().map(M.INDICES[nome]['setores']).fillna('Outros')})
        ordem = df_s.groupby('setor')['corr'].median().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.boxplot([df_s.loc[df_s['setor'] == s, 'corr'].values for s in ordem])
        ax.set_xticks(range(1, len(ordem) + 1))
        ax.set_xticklabels(ordem, rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Correlação com o índice')
        fig.tight_layout()
        mostrar(fig)

    st.markdown("**Risco × retorno por ação**")
    est = M.estatisticas_acoes(X)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(est['Volatilidade anualizada (%)'], est['Retorno médio anualizado (%)'], color=COR[nome], alpha=.6, s=50)
    for a in est['Sharpe aprox. anualizado'].nlargest(5).index:
        ax.annotate(a, (est.loc[a, 'Volatilidade anualizada (%)'], est.loc[a, 'Retorno médio anualizado (%)']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.axhline(0, color='red', ls='--', lw=.8, alpha=.5)
    ax.set_xlabel('Volatilidade anualizada (%)')
    ax.set_ylabel('Retorno médio anualizado (%)')
    ax.grid(alpha=.3)
    fig.tight_layout()
    mostrar(fig)
    st.dataframe(est.sort_values('Sharpe aprox. anualizado', ascending=False))

# ───────────────────────────── Backtest ─────────────────────────────
def mostrar_backtest(bt):
    n_, cfg, res, base = bt['nome'], bt['cfg'], bt['res'], bt['base']
    D = DADOS[n_]['D']
    ms = pd.DataFrame([r['m'] for r in res])
    medias = ms.mean()

    st.markdown(f"#### Resultado · {n_} · {bt['periodo']}")
    cols = st.columns(len(medias))
    for c_, (k_, v_) in zip(cols, medias.items()):
        c_.metric(k_, FMT.get(k_, '{:.2f}').format(v_))
    if len(res) > 1:
        te = ms['TE (%)']
        st.caption(f"TE por semente: {np.round(te.values, 2).tolist()} · média {te.mean():.2f} · dp {te.std(ddof=0):.2f}")

    if base is not None:
        mb = pd.DataFrame([r['m'] for r in base]).mean()
        comp = pd.DataFrame({'Baseline': mb, 'Configuração escolhida': medias}).round(3)
        st.markdown("**Baseline × configuração escolhida** (média das sementes)")
        st.dataframe(comp)
        st.caption(f"Melhora média do TE: {mb['TE (%)'] - medias['TE (%)']:+.2f} p.p.")

    r0 = res[0]['r']
    y0 = D['y'].loc[r0.index]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.exp(y0.cumsum()) * 100, label='Índice', lw=2, color=COR[n_])
        ax.plot(np.exp(r0.cumsum()) * 100, '--', label='Carteira (semente 0)', lw=2, color='tomato')
        ax.set_title('Retorno acumulado (base 100)')
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        mostrar(fig)
    with c2:
        te_ano = pd.concat([r['te_ano'] for r in res], axis=1).mean(axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(te_ano.index.astype(str), te_ano.values, color='steelblue')
        ax.set_title('TE por ano (% a.a.)')
        ax.grid(alpha=.3, axis='y')
        fig.tight_layout()
        mostrar(fig)

    h, rb = res[0]['hist'], res[0]['rebal']
    reb = pd.DataFrame({'Data': [x.date() for x in rb], 'Ações': [len(x) for x in h],
                        'Trocas vs. anterior': pd.array([None] + [len(set(h[i]) - set(h[i - 1])) for i in range(1, len(h))],
                                                        dtype='Int64')})
    st.markdown("**Rebalanceamentos (semente 0)**")
    st.dataframe(reb, hide_index=True)

    st.markdown("**Pesos da carteira final** (último rebalanceamento, treinando até o último pregão disponível)")
    w = M.pesos_finais(D, cfg, res[0]).to_frame('peso')
    w['setor'] = w.index.to_series().map(M.INDICES[n_]['setores']).fillna('Outros')
    w['peso (%)'] = (w['peso'] * 100).round(2)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(w[['peso (%)', 'setor']])
        st.download_button("Baixar pesos (CSV)", w[['peso']].to_csv().encode(), file_name=f"pesos_carteira_{M.INDICES[n_]['tag']}.csv",
                           mime="text/csv")
    with c2:
        fig, ax = plt.subplots(figsize=(6, max(3, 0.28 * len(w))))
        ax.barh(w.index[::-1], w['peso (%)'].values[::-1], color=COR[n_])
        ax.set_xlabel('Peso (%)')
        fig.tight_layout()
        mostrar(fig)


with t_bt:
    st.subheader(f"Backtest walk-forward · {nome}")
    st.caption("A cada rebalanceamento o modelo usa só dados anteriores àquela data: o AG escolhe as K ações e "
               "um otimizador calcula os pesos (0 ≤ peso ≤ teto, soma = 1) para minimizar o tracking error.")
    cfg_final = M.carregar_cfg_final()
    preset = st.radio("Ponto de partida", ["Configuração final", "Baseline"], horizontal=True)
    partida = dict(cfg_final[nome]) if preset == "Configuração final" else dict(M.CFG_BASE)
    kk = f"{nome}_{preset}".replace(' ', '').replace('&', '')

    with st.form(f"form_bt_{kk}"):
        c1, c2, c3 = st.columns(3)
        periodo = c1.selectbox("Período avaliado", list(PERIODOS), key=f"per_{kk}")
        K = c1.slider("K (nº de ações)", 5, 40, int(partida['K']), key=f"K_{kk}")
        cap = c1.slider("Teto de peso por ação", 0.05, 0.30, float(partida['cap']), 0.01, key=f"cap_{kk}")
        metodo = c1.selectbox("Método de seleção", ['ag', 'guloso'], indice_de(['ag', 'guloso'], partida['metodo']), key=f"met_{kk}")

        janela = c2.slider("Janela de treino (anos)", 1, 5, int(partida['janela_anos']), key=f"jan_{kk}")
        freq = c2.selectbox("Rebalancear a cada (meses)", [3, 6, 12], indice_de([3, 6, 12], partida['freq_meses']), key=f"frq_{kk}")
        hl_opts = [None, 21, 63, 126, 252]
        if partida['halflife'] not in hl_opts:
            hl_opts.append(partida['halflife'])
        halflife = c2.selectbox("Meia-vida da covariância (dias)", hl_opts, indice_de(hl_opts, partida['halflife']),
                                format_func=lambda v: 'pesos iguais' if v is None else str(v), key=f"hl_{kk}")
        shrink = c2.slider("Encolhimento da covariância", 0.0, 0.5, float(partida['shrink']), 0.05, key=f"shr_{kk}")

        corr_min = c3.slider("Correlação mínima (−1 desliga o filtro)", -1.0, 0.9, float(partida['corr_min']), 0.05, key=f"cm_{kk}")
        pop = c3.slider("AG · tamanho da população", 20, 100, M.AG_FINAL['pop'], 10, key=f"pop_{kk}")
        ger = c3.slider("AG · gerações", 10, 100, M.AG_FINAL['geracoes'], 10, key=f"ger_{kk}")
        pac = c3.slider("AG · paciência (gerações sem melhora)", 5, 30, M.AG_FINAL['paciencia'], key=f"pac_{kk}")

        with st.expander("Avançado"):
            a1, a2, a3 = st.columns(3)
            cv = a1.selectbox("Blocos de validação cruzada no fitness", [0, 4], indice_de([0, 4], partida['cv_blocos']), key=f"cv_{kk}")
            lam = a2.selectbox("Penalização de turnover nos pesos", [0.0, 0.1, 0.5], indice_de([0.0, 0.1, 0.5], partida['lam_turn']), key=f"lam_{kk}")
            gam = a3.selectbox("Penalização de troca de ações (AG)", [0.0, 0.1, 0.3], indice_de([0.0, 0.1, 0.3], partida['gamma_troca']), key=f"gam_{kk}")

        n_seeds = st.slider("Nº de sementes do AG (o relatório usa 3; 1 é bem mais rápido)", 1, 3, 1, key=f"seeds_{kk}")
        comparar = st.checkbox("Comparar com o baseline (dobra o tempo)", value=False, key=f"cmp_{kk}")
        st.caption("⚠️ O período de teste deve ser usado uma única vez no protocolo. Para explorar parâmetros, prefira a validação.")
        enviar = st.form_submit_button("▶ Rodar backtest", type="primary")

    if enviar:
        cfg = {**M.CFG_BASE, 'K': K, 'cap': cap, 'metodo': metodo, 'janela_anos': janela, 'freq_meses': freq,
               'halflife': halflife, 'shrink': shrink, 'corr_min': corr_min, 'cv_blocos': cv,
               'lam_turn': lam, 'gamma_troca': gam}
        ag_kw = dict(pop=pop, geracoes=ger, paciencia=pac)
        ini, fim = PERIODOS[periodo]
        with st.spinner(f"Rodando {n_seeds} semente(s)… o AG pode levar alguns minutos."):
            res = rodar_backtest(nome, cfg, ag_kw, n_seeds, ini, fim)
            base = rodar_backtest(nome, dict(M.CFG_BASE), ag_kw, n_seeds, ini, fim) if comparar else None
        st.session_state['bt'] = dict(nome=nome, cfg=cfg, res=res, base=base, periodo=periodo)

    bt = st.session_state.get('bt')
    if bt and bt['nome'] == nome:
        mostrar_backtest(bt)
    elif bt:
        st.info(f"O último backtest foi do {bt['nome']}. Rode de novo para ver o {nome}.")
    else:
        st.info("Ajuste os parâmetros e clique em **Rodar backtest**.")

# ───────────────────────────── Escolha de K ─────────────────────────────
with t_k:
    st.subheader(f"TE na validação ({M.VAL_START[:4]}) × número de ações K · {nome}")
    st.caption(f"Treina até {M.TRAIN_END} e avalia na validação, comparando AG, seleção gulosa e top-correlação. "
               "Leva alguns minutos, principalmente com muitos valores de K.")
    with st.form(f"form_k_{nome}".replace(' ', '').replace('&', '')):
        Ks = st.multiselect("Valores de K", [8, 10, 15, 20, 25, 30], default=[10, 20, 30])
        n_seeds_k = st.slider("Nº de sementes do AG", 1, 3, 1)
        enviar_k = st.form_submit_button("▶ Calcular", type="primary")
    if enviar_k:
        if not Ks:
            st.warning("Escolha pelo menos um valor de K.")
        else:
            with st.spinner("Calculando…"):
                st.session_state['k'] = (nome, rodar_validacao_K(nome, tuple(sorted(Ks)), n_seeds_k))
    kres = st.session_state.get('k')
    if kres and kres[0] == nome:
        df = kres[1].copy()
        te = df['AG (média)']
        df['Ganho por ação extra (p.p.)'] = ((te.shift(1) - te) / te.index.to_series().diff()).round(4)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(df.index, df['AG (média)'], yerr=df['AG (dp)'], marker='o', label='AG (média ± dp)')
        ax.plot(df.index, df['Guloso'], marker='s', label='Guloso')
        ax.plot(df.index, df['Top-corr'], marker='^', label='Top-correlação')
        ax.set_xlabel('K')
        ax.set_ylabel('TE (% a.a.)')
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        mostrar(fig)
        st.dataframe(df)
    elif kres:
        st.info(f"O último cálculo foi do {kres[0]}. Rode de novo para ver o {nome}.")

# ───────────────────────────── Resultados salvos ─────────────────────────────
with t_salvos:
    st.subheader("Arquivos gerados pelo notebook (pasta resultados/)")
    tag = M.INDICES[nome]['tag']
    busca = {'S&P 100': 'busca_sp.csv', 'IBOVESPA': 'busca_ibov.csv'}[nome]
    itens = [
        ("Números da apresentação", 'numeros_apresentacao.csv', {}),
        (f"Resultado no teste · {nome}", f'resultado_teste_{tag}.csv', {'index_col': 0}),
        (f"Pesos da carteira final · {nome}", f'pesos_carteira_{tag}.csv', {'index_col': 0}),
        (f"Features por ação · {nome}", f'features_{tag}.csv', {'index_col': 0}),
    ]
    achou = False
    for titulo, arq, kw in itens:
        tab = ler_csv(arq, **kw)
        if tab is not None:
            achou = True
            st.markdown(f"**{titulo}**")
            st.dataframe(tab)
    log = ler_csv(busca)
    if log is not None:
        achou = True
        st.markdown(f"**Busca de hiperparâmetros · {nome}** (ordenada pelo ganho)")
        st.dataframe(log.sort_values('ganho', ascending=False), hide_index=True)
    cfg_json = M.RES_DIR / 'config_final.json'
    if cfg_json.exists():
        achou = True
        st.markdown("**Configuração final**")
        st.json(cfg_json.read_text())
    if not achou:
        st.info(f"Nenhum arquivo encontrado em `{M.RES_DIR}`. Rode o notebook para gerá-los.")
