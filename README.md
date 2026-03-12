# Global Portfolio Backtest System

A comprehensive portfolio backtest system with Streamlit dashboard for analyzing global portfolio performance over 30 years.

## Portfolio Allocation

| ETF | Weight | Description |
|-----|--------|-------------|
| QQQ | 30% | Nasdaq 100 |
| SPY | 22% | S&P 500 |
| ASHR | 20% | China A-shares |
| VIG | 12% | Dividend Appreciation |
| EWH | 8% | Hong Kong |
| GLD | 8% | Gold |

## Features

- **DCA Strategy**: Dollar-cost averaging with monthly contributions
- **Rebalancing**: Configurable rebalancing frequency (monthly/quarterly/semi-annual/annual)
- **Performance Metrics**: CAGR, Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Visualizations**: Portfolio growth charts, drawdown charts, annual/monthly returns
- **Interactive Dashboard**: Streamlit-based UI for easy configuration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

### Run Backtest Programmatically

```python
from backtest import PortfolioBacktest

# Create backtest instance
bt = PortfolioBacktest(
    allocation={'QQQ': 0.30, 'SPY': 0.22, 'ASHR': 0.20, 'VIG': 0.12, 'EWH': 0.08, 'GLD': 0.08},
    start_date='1998-01-01',
    initial_capital=30000,
    monthly_contribution=4500,
    rebalance_frequency='semi-annual'
)

# Run backtest
results = bt.run_backtest()

# Get metrics
metrics = bt.get_metrics()
print(f"CAGR: {metrics['cagr']*100:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Generate report
print(bt.generate_report())
```

## Project Structure

```
global-portfolio-backtest-v2/
├── app.py              # Streamlit dashboard
├── backtest.py         # Core backtest engine
├── config.py           # Configuration and parameters
├── data_fetcher.py     # yfinance data fetching
├── metrics.py          # Financial metrics calculations
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Data Notes

- Uses yfinance for historical ETF data
- Some ETFs (ASHR, VIG, GLD) have later inception dates than 1998
- Proxy data is used where available (e.g., FXI for early ASHR period)
- GLD data starts from 2004 (no suitable proxy available)

## Metrics Explained

- **CAGR**: Compound Annual Growth Rate - annualized return over the entire period
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return using standard deviation
- **Sortino Ratio**: Risk-adjusted return using downside deviation only
- **Calmar Ratio**: CAGR divided by absolute maximum drawdown

## License

MIT License