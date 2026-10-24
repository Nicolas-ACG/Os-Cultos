# Os-Cultos
# Index tracking com algoritmo genético: S&P 100 e Ibovespa

Replicação de índices com 20 ações (long-only, teto de 10% por ativo),
seleção por algoritmo genético e pesos por otimização quadrática.

## Como rodar
1. Python 3.X (veja requirements.txt)
2. `pip install -r requirements.txt`
3. Abra `Projeto_V2.ipynb` no JupyterLab
4. `Kernel > Restart Kernel and Run All Cells` (leva alguns minutos)

Os dados já estão nos CSVs `v2_*.csv`, então o notebook não baixa nada.
Para atualizar os dados, apague esses arquivos (o Yahoo Finance pode
devolver valores diferentes dos originais).

## Protocolo
- Treino até 2022, validação em 2023, teste de 2024-01-01 a 2025-04-30
- K=20 e teto de 10% fixos; o resto foi escolhido na validação (2021-2023)
- Modelo final: S&P 100 = baseline; Ibovespa = meia-vida 126 dias,
  shrinkage 0,1 e filtro de correlação mínima 0,3

## Limitações
Viés de sobrevivência (constituintes atuais), ^OEX sem dividendos,
sem custos de transação, teste de ~16 meses e já observado antes
da busca de configuração.