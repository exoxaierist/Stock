from pandas_datareader import data as pdr
import yfinance as yf
import matplotlib.pyplot as plt

naver = yf.download('035420.KS', interval='1h', start='2022-06-01', end='2023-02-01')
fig, (price, volume) = plt.subplots(nrows=2, ncols=1)

price.plot(naver['Close'], label='Closing Price')
volume.plot(naver['Volume'], label='Volume')

price.legend()
volume.legend()
plt.show()

print(naver)
