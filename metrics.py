"""
Financial metrics calculation module
Calculates CAGR, Max Drawdown, Sharpe Ratio, Sortino Ratio
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats
import config


def calculate_cagr(
    portfolio_values: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> float:
    """
    Calculate Compound Annual Growth Rate

    Args:
        portfolio_values: Time series of portfolio values
        start_date: Start date
        end_date: End date

    Returns:
        CAGR as decimal (e.g., 0.08 for 8%)
    """
    if len(portfolio_values) < 2:
        return 0.0

    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]

    # Calculate years
    years = (end_date - start_date).days / 365.25

    if years <= 0:
        return 0.0

    # CAGR = (End / Start)^(1/years) - 1
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr


def calculate_max_drawdown(portfolio_values: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate Maximum Drawdown

    Args:
        portfolio_values: Time series of portfolio values

    Returns:
        Tuple of (max_drawdown, start_date, end_date)
    """
    if len(portfolio_values) < 2:
        return 0.0, None, None

    # Calculate running maximum
    running_max = portfolio_values.cummax()

    # Calculate drawdown
    drawdown = (portfolio_values - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find the peak before the max drawdown
    peak_idx = running_max.loc[:max_dd_idx].idxmax()

    return max_dd, peak_idx, max_dd_idx


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = None,
    periods_per_year: int = 12
) -> float:
    """
    Calculate Sharpe Ratio

    Args:
        returns: Period returns (e.g., monthly returns)
        risk_free_rate: Annual risk-free rate (default from config)
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Sharpe ratio
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to period rate
    rf_per_period = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - rf_per_period

    # Sharpe = mean(excess returns) / std(returns) * sqrt(periods)
    mean_excess = excess_returns.mean()
    std_returns = returns.std()

    if std_returns == 0:
        return 0.0

    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = None,
    periods_per_year: int = 12
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of std)

    Args:
        returns: Period returns (e.g., monthly returns)
        risk_free_rate: Annual risk-free rate (default from config)
        periods_per_year: Number of periods per year (12 for monthly)

    Returns:
        Sortino ratio
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to period rate
    rf_per_period = risk_free_rate / periods_per_year

    # Excess returns
    excess_returns = returns - rf_per_period

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No negative returns

    downside_std = downside_returns.std()

    if downside_std == 0:
        return 0.0

    # Sortino = mean(excess returns) / downside_std * sqrt(periods)
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)
    return sortino


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 12
) -> float:
    """
    Calculate annualized volatility

    Args:
        returns: Period returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0

    return returns.std() * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float
) -> float:
    """
    Calculate Calmar Ratio (CAGR / abs(Max Drawdown))

    Args:
        cagr: Compound Annual Growth Rate
        max_drawdown: Maximum Drawdown (negative value)

    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float('inf') if cagr > 0 else 0.0

    return cagr / abs(max_drawdown)


def calculate_all_metrics(
    portfolio_values: pd.Series,
    returns: pd.Series = None
) -> dict:
    """
    Calculate all portfolio metrics

    Args:
        portfolio_values: Time series of portfolio values
        returns: Period returns (calculated if not provided)

    Returns:
        Dictionary of metrics
    """
    if len(portfolio_values) < 2:
        return {
            'cagr': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'volatility': 0,
            'calmar_ratio': 0,
            'total_return': 0
        }

    # Calculate returns if not provided
    if returns is None:
        returns = portfolio_values.pct_change().dropna()

    # Date range
    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]

    # Calculate metrics
    cagr = calculate_cagr(portfolio_values, start_date, end_date)
    max_dd, dd_peak, dd_trough = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    vol = calculate_volatility(returns)
    calmar = calculate_calmar_ratio(cagr, max_dd)

    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

    return {
        'cagr': cagr,
        'max_drawdown': max_dd,
        'max_dd_peak_date': dd_peak,
        'max_dd_trough_date': dd_trough,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'volatility': vol,
        'calmar_ratio': calmar,
        'total_return': total_return,
        'start_date': start_date,
        'end_date': end_date,
        'total_contributions': None,  # Will be filled by backtest
        'final_value': portfolio_values.iloc[-1]
    }


def calculate_annual_returns(portfolio_values: pd.Series) -> pd.DataFrame:
    """
    Calculate annual returns

    Args:
        portfolio_values: Time series of portfolio values

    Returns:
        DataFrame with year and annual return
    """
    if len(portfolio_values) < 2:
        return pd.DataFrame()

    # Resample to year-end values
    yearly = portfolio_values.resample('Y').last()

    # Calculate annual returns
    annual_returns = yearly.pct_change().dropna()

    # Create DataFrame
    df = pd.DataFrame({
        'Year': annual_returns.index.year,
        'Annual_Return': annual_returns.values
    })

    return df


def calculate_monthly_returns(portfolio_values: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns by year

    Args:
        portfolio_values: Time series of portfolio values

    Returns:
        DataFrame with monthly returns pivoted by year
    """
    if len(portfolio_values) < 2:
        return pd.DataFrame()

    # Calculate monthly returns
    monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()

    # Create DataFrame with year and month
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    # Pivot to wide format
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Add annual return
    yearly = portfolio_values.resample('Y').last()
    annual = yearly.pct_change().dropna()
    pivot['Year'] = annual.values

    return pivot


if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='M')
    values = pd.Series([100000 * (1.007 ** i) for i in range(len(dates))], index=dates)

    metrics = calculate_all_metrics(values)
    for key, value in metrics.items():
        print(f"{key}: {value}")