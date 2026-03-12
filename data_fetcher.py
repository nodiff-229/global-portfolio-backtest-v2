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
    use_proxy: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch ETF price data from Yahoo Finance

    Args:
        ticker: ETF ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        use_proxy: Whether to use proxy for unavailable periods
        verbose: Print proxy usage information

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    inception = config.ETF_INCEPTION.get(ticker, '1990-01-01')

    # Check if we need a proxy (only if start_date is before ETF inception)
    if use_proxy and ticker in config.PROXY_MAPPING and start_date < inception:
        proxy_ticker = config.PROXY_MAPPING[ticker]

        if verbose:
            print(f"  {ticker}: Using proxy {proxy_ticker} for period {start_date} to {inception}")

        # Fetch proxy data for early period (before ETF existed)
        proxy_data = yf.download(
            proxy_ticker,
            start=start_date,
            end=inception,
            progress=False
        )

        # Flatten multi-level columns if present
        if isinstance(proxy_data.columns, pd.MultiIndex):
            proxy_data.columns = proxy_data.columns.get_level_values(0)

        # Fetch actual ETF data for later period
        actual_data = yf.download(
            ticker,
            start=inception,
            end=end_date,
            progress=False
        )

        # Flatten multi-level columns if present
        if isinstance(actual_data.columns, pd.MultiIndex):
            actual_data.columns = actual_data.columns.get_level_values(0)

        # Combine data
        if not proxy_data.empty and not actual_data.empty:
            # Use proxy for early period, actual for later
            combined = pd.concat([proxy_data, actual_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            if verbose:
                print(f"  {ticker}: Combined {len(proxy_data)} proxy rows + {len(actual_data)} actual rows")
            return combined
        elif not actual_data.empty:
            if verbose:
                print(f"  {ticker}: No proxy data available, using actual data only ({len(actual_data)} rows)")
            return actual_data
        elif not proxy_data.empty:
            if verbose:
                print(f"  {ticker}: No actual data available, using proxy only ({len(proxy_data)} rows)")
            return proxy_data
        else:
            if verbose:
                print(f"  {ticker}: Warning - No data available from either source")
            return pd.DataFrame()

    # Direct download (no proxy needed or start date is after inception)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if verbose and not data.empty:
        print(f"  {ticker}: Fetched {len(data)} rows (direct, no proxy needed)")

    return data


def get_all_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple ETFs

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Print proxy usage information

    Returns:
        Dictionary mapping tickers to DataFrames
    """
    data = {}
    for ticker in tickers:
        if verbose:
            print(f"Fetching {ticker}...")
        df = get_etf_data(ticker, start_date, end_date, verbose=verbose)
        if not df.empty:
            data[ticker] = df
        else:
            if verbose:
                print(f"  Warning: No data retrieved for {ticker}")
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
        if df.empty:
            print(f"Warning: Empty data for {ticker}")
            continue

        # Handle both single-level and multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Adj Close' in df.columns:
            series = df['Adj Close']
        elif 'Close' in df.columns:
            series = df['Close']
        else:
            print(f"Warning: No close price found for {ticker}")
            continue

        # Only add non-empty series
        if not series.empty:
            prices[ticker] = series

    if not prices:
        return pd.DataFrame()

    return pd.DataFrame(prices)


def resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily prices to monthly using last available price

    Args:
        prices: Daily price DataFrame

    Returns:
        Monthly price DataFrame
    """
    return prices.resample('ME').last()


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