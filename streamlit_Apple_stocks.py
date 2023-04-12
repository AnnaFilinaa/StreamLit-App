import yfinance as yf
import streamlit as st

st.write("""
# Данные котировок Apple

На графике представлены **цена закрытия** и **объем торгов** Apple за последние 10 лет !

""")

tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2013-4-11', end='2023-4-11')

st.write("""
## Цена закрытия акций
""")
st.line_chart(tickerDf.Close)

st.write("""
## Объем торгов
""")
st.line_chart(tickerDf.Volume)