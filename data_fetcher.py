"""
Data fetching module for portfolio backtest
Handles yfinance data retrieval and caching
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config


def get_etf_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    use_proxy: bool = True
) -> pd.DataFrame:
    """
    Fetch ETF price data from Yahoo Finance

    Args:
        ticker: ETF ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        use_proxy: Whether to use proxy for unavailable periods

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    inception = config.ETF_INCEPTION.get(ticker, '1990-01-01')

    # Check if we need a proxy
    if use_proxy and ticker in config.PROXY_MAPPING:
        proxy_ticker = config.PROXY_MAPPING[ticker]
        proxy_inception = config.ETF_INCEPTION.get(proxy_ticker, '1990-01-01')

        # Fetch proxy data for early period
        proxy_data = yf.download(
            proxy_ticker,
            start=start_date,
            end=min(inception, end_date),
            progress=False
        )

        # Fetch actual ETF data for later period
        actual_data = yf.download(
            ticker,
            start=inception,
            end=end_date,
            progress=False
        )

        # Combine data
        if not proxy_data.empty and not actual_data.empty:
            # Use proxy for early period, actual for later
            combined = pd.concat([proxy_data, actual_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            return combined
        elif not actual_data.empty:
            return actual_data
        else:
            return proxy_data

    # Direct download
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data


def get_all_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple ETFs

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Dictionary mapping tickers to DataFrames
    """
    data = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = get_etf_data(ticker, start_date, end_date)
        if not df.empty:
            data[ticker] = df
        else:
            print(f"Warning: No data retrieved for {ticker}")
    return data


def get_adj_close_prices(
    ticker_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Extract adjusted close prices for all ETFs

    Args:
        ticker_data: Dictionary of ticker -> DataFrame

    Returns:
        DataFrame with adjusted close prices, indexed by date
    """
    prices = {}
    for ticker, df in ticker_data.items():
        if 'Adj Close' in df.columns:
            prices[ticker] = df['Adj Close']
        elif 'Close' in df.columns:
            prices[ticker] = df['Close']
        else:
            print(f"Warning: No close price found for {ticker}")

    return pd.DataFrame(prices)


def resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to monthly using last available price

    Args:
        prices: Daily price DataFrame

    Returns:
        Monthly price DataFrame
    """
    return prices.resample('M').last()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from price data

    Args:
        prices: Price DataFrame

    Returns:
        Returns DataFrame (percentage change)
    """
    return prices.pct_change().dropna()


if __name__ == "__main__":
    # Test data fetching
    tickers = list(config.PORTFOLIO_ALLOCATION.keys())
    data = get_all_etf_data(tickers, '1998-01-01')
    prices = get_adj_close_prices(data)
    print(prices.head())
    print(prices.tail())
    print(f"\nDate range: {prices.index.min()} to {prices.index.max()}")