"""
Global Portfolio Backtest Configuration
"""

# Portfolio Allocation (weights must sum to 1.0)
PORTFOLIO_ALLOCATION = {
    'QQQ': 0.30,   # Nasdaq 100
    'SPY': 0.22,   # S&P 500
    'ASHR': 0.20,  # China A-shares
    'VIG': 0.12,   # Dividend
    'EWH': 0.08,   # Hong Kong
    'GLD': 0.08    # Gold
}

# ETF inception dates for reference (some ETFs didn't exist in 1998)
ETF_INCEPTION = {
    'QQQ': '1999-03-10',
    'SPY': '1993-01-29',
    'ASHR': '2014-04-02',  # Will need proxy data
    'VIG': '2006-04-21',
    'EWH': '1996-05-16',
    'GLD': '2004-11-18'
}

# Backtest Parameters
BACKTEST_CONFIG = {
    'start_date': '1998-01-01',
    'end_date': None,  # None means today
    'initial_capital': 30000,
    'monthly_contribution': 4500,
    'rebalance_frequency': 'semi-annual'  # 'monthly', 'quarterly', 'semi-annual', 'annual'
}

# Proxy ETFs for assets that didn't exist in early years
# These will be used before the actual ETF was available
# Format: 'ETF': 'PROXY' (proxy data used before ETF inception)
PROXY_MAPPING = {
    'QQQ': '^NDX',       # Nasdaq 100 index (available since 1985)
    'ASHR': 'FXI',       # China large-cap ETF (available since 2004) - best proxy for China A-shares
    'VIG': 'VTI',        # Total US stock market (available since 2001, close proxy for dividend stocks)
    'GLD': 'GC=F',       # Gold futures (available since 1970s)
}

# Risk-free rate for Sharpe/Sortino calculations (annualized)
RISK_FREE_RATE = 0.02  # 2% annual