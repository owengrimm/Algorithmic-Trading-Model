# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import os
import matplotlib.ticker as mtick
plt.style.use('ggplot')

### 1. Load Twitter Sentiment Data

data_folder = '/Users/owengrimm/Algorithmic-Trading-Model/Data'

sentiment_df = pd.read_csv(os.path.join(data_folder, 'sentiment_data.csv'))

sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

sentiment_df = sentiment_df.set_index(['date', 'symbol'])

sentiment_df['engagement_ratio'] = sentiment_df['twitterComments']/sentiment_df['twitterLikes']

# Filter out tweets with minimal engagement
sentiment_df = sentiment_df[(sentiment_df['twitterLikes'] > 20)&(sentiment_df['twitterComments'] > 10)]

### 2. Aggregate Monthly and calculate average sentiment for each month

aggregated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq = 'M'), 'symbol'])
[['engagement_ratio']].mean())

# Rank each stock based on its average engagement
aggregated_df['rank'] = (aggregated_df.groupby(level = 0)['engagement_ratio']
                         .transform(lambda x: x.rank(ascending = False)))

### 3. Select top 5 stocks based on cross-sectional ranking for each month

filtered_df = aggregated_df[aggregated_df['rank'] < 6].copy()

filtered_df = filtered_df.reset_index(level = 1)

filtered_df.index = filtered_df.index + pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])

### 4. Extract the stocks to form portfolios at the start of each new month

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:

    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level = 0).index.tolist()

### 5. Download new stock prices for only selected/shortlisted stocks

stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()

prices_df = yf.download(tickers = stocks_list,
                        start = '2021-01-01',
                        end = '2023-03-01')

### 6. Calculate Portfolio Returns with monthly rebalancing

returns_df = np.log(prices_df['Adj Close']).diff().dropna()

portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():

    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')

    cols = fixed_dates[start_date]

    temp_df = returns_df[start_date:end_date][cols].mean(axis = 1).to_frame('portfolio_return')

    portfolio_df = pd.concat([portfolio_df, temp_df], axis = 0)

### 7. Download NASDAQ/QQQ prices and calculate returns to compare to our strategy

qqq_df = yf.download(tickers = 'QQQ',
                     start = '2021-01-01',
                     end = '2023-03-01')

qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')

portfolio_df = portfolio_df.merge(qqq_ret,
                                  left_index = True,
                                  right_index = True)

portfolios_cumulative_return = np.exp(np.log(portfolio_df).cumsum()).sub(1)

portfolios_cumulative_return.plot(figsize = (16,6))

plt.title('Twitter Engagement Ratio Strategy Return Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()
