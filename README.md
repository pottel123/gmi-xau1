# GMI‑XAU • Gold Daily Macro Index

Um painel simples em **Streamlit** que agrega fatores macro (DXY, juros 10Y, S&P500, VIX, cobre) e estima a **% de alta/baixa diária do ouro** via regressão linear com dados do Yahoo Finance.

## Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Como implantar (grátis) no Streamlit Cloud

1. Faça fork deste projeto no seu GitHub.
2. No [Streamlit Community Cloud](https://share.streamlit.io/), crie um novo app apontando para `app.py`.
3. Defina a região próxima do Brasil e salve.

## Notas

- O ticker do ouro usa `XAUUSD=X` (spot) por padrão; você pode trocar para `GC=F`.
- O dado de **breakeven 10y (T10YIE)** é usado se disponível no Yahoo. Caso não esteja, o app automaticamente usa o **10Y nominal** como proxy de taxa real.
- O modelo é **educacional**. Não é sinal de investimento.