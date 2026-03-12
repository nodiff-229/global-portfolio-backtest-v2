"""
Core Backtest Engine for Global Portfolio
Implements DCA strategy with periodic rebalancing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import data_fetcher as df
import metrics
import config


class PortfolioBacktest:
    """
    Portfolio backtest engine implementing DCA strategy with rebalancing
    """

    def __init__(
        self,
        allocation: Dict[str, float] = None,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = None,
        monthly_contribution: float = None,
        rebalance_frequency: str = None
    ):
        """
        Initialize backtest engine

        Args:
            allocation: Portfolio allocation dict (ticker -> weight)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
            monthly_contribution: Monthly contribution amount
            rebalance_frequency: 'monthly', 'quarterly', 'semi-annual', 'annual'
        """
        # Use config defaults if not specified
        self.allocation = allocation or config.PORTFOLIO_ALLOCATION
        self.start_date = start_date or config.BACKTEST_CONFIG['start_date']
        self.end_date = end_date or config.BACKTEST_CONFIG['end_date'] or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital or config.BACKTEST_CONFIG['initial_capital']
        self.monthly_contribution = monthly_contribution or config.BACKTEST_CONFIG['monthly_contribution']
        self.rebalance_frequency = rebalance_frequency or config.BACKTEST_CONFIG['rebalance_frequency']

        # Validate allocation sums to 1
        total_weight = sum(self.allocation.values())
        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(f"Allocation weights must sum to 1.0, got {total_weight}")

        self.tickers = list(self.allocation.keys())
        self.weights = list(self.allocation.values())

        # Data storage
        self.prices = None
        self.monthly_prices = None
        self.results = None
        self.monthly_values = None

    def fetch_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Fetch price data for all tickers

        Args:
            verbose: Print progress messages

        Returns:
            DataFrame with adjusted close prices
        """
        if verbose:
            print(f"Fetching data from {self.start_date} to {self.end_date}...")

        # Fetch data for all tickers
        data = df.get_all_etf_data(self.tickers, self.start_date, self.end_date)

        # Extract adjusted close prices
        self.prices = df.get_adj_close_prices(data)

        # Resample to monthly
        self.monthly_prices = df.resample_monthly(self.prices)

        if verbose:
            print(f"Data range: {self.monthly_prices.index.min()} to {self.monthly_prices.index.max()}")
            print(f"Tickers available: {list(self.monthly_prices.columns)}")

        return self.monthly_prices

    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        Generate rebalance dates based on frequency

        Returns:
            List of rebalance dates
        """
        dates = self.monthly_prices.index

        if self.rebalance_frequency == 'monthly':
            return list(dates)
        elif self.rebalance_frequency == 'quarterly':
            # Rebalance at end of Mar, Jun, Sep, Dec
            return [d for d in dates if d.month in [3, 6, 9, 12]]
        elif self.rebalance_frequency == 'semi-annual':
            # Rebalance at end of Jun and Dec
            return [d for d in dates if d.month in [6, 12]]
        elif self.rebalance_frequency == 'annual':
            # Rebalance at end of year
            return [d for d in dates if d.month == 12]
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")

    def run_backtest(self, verbose: bool = True) -> pd.DataFrame:
        """
        Run the backtest

        Args:
            verbose: Print progress messages

        Returns:
            DataFrame with monthly portfolio values and contributions
        """
        if self.monthly_prices is None:
            self.fetch_data(verbose)

        if verbose:
            print(f"\nRunning backtest...")
            print(f"Initial capital: ${self.initial_capital:,.0f}")
            print(f"Monthly contribution: ${self.monthly_contribution:,.0f}")
            print(f"Rebalance frequency: {self.rebalance_frequency}")

        # Get rebalance dates
        rebalance_dates = set(self.get_rebalance_dates())

        # Initialize holdings (number of shares per ticker)
        holdings = {ticker: 0.0 for ticker in self.tickers}

        # Track portfolio values and contributions
        results = []
        total_contributions = 0.0

        # Process each month
        for i, date in enumerate(self.monthly_prices.index):
            # Get current prices
            current_prices = self.monthly_prices.loc[date]

            # Calculate current portfolio value
            portfolio_value = sum(
                holdings[ticker] * current_prices[ticker]
                for ticker in self.tickers
                if ticker in current_prices and not pd.isna(current_prices[ticker])
            )

            # Add contributions
            if i == 0:
                # Initial capital
                contribution = self.initial_capital
                total_contributions += contribution
            else:
                # Monthly contribution
                contribution = self.monthly_contribution
                total_contributions += contribution

            # Check if rebalance is needed
            if date in rebalance_dates or i == 0:
                # Total value to allocate
                total_value = portfolio_value + contribution

                # Calculate target allocations
                target_values = {ticker: total_value * self.allocation[ticker] for ticker in self.tickers}

                # Calculate new holdings
                holdings = {}
                for ticker in self.tickers:
                    if ticker in current_prices and not pd.isna(current_prices[ticker]):
                        holdings[ticker] = target_values[ticker] / current_prices[ticker]
                    else:
                        holdings[ticker] = 0.0
            else:
                # Just add contribution proportionally to target weights
                if contribution > 0:
                    for ticker in self.tickers:
                        if ticker in current_prices and not pd.isna(current_prices[ticker]):
                            shares_to_buy = (contribution * self.allocation[ticker]) / current_prices[ticker]
                            holdings[ticker] = holdings.get(ticker, 0) + shares_to_buy

            # Calculate final portfolio value for this month
            final_value = sum(
                holdings[ticker] * current_prices[ticker]
                for ticker in self.tickers
                if ticker in current_prices and not pd.isna(current_prices[ticker])
            )

            # Record results
            results.append({
                'date': date,
                'portfolio_value': final_value,
                'contribution': contribution,
                'total_contributions': total_contributions,
                'rebalanced': date in rebalance_dates or i == 0
            })

        self.results = pd.DataFrame(results)
        self.results.set_index('date', inplace=True)

        if verbose:
            print(f"\nBacktest complete!")
            print(f"Total contributions: ${total_contributions:,.0f}")
            print(f"Final portfolio value: ${final_value:,.0f}")

        return self.results

    def get_metrics(self) -> dict:
        """
        Calculate portfolio metrics

        Returns:
            Dictionary of metrics
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        portfolio_values = self.results['portfolio_value']
        returns = portfolio_values.pct_change().dropna()

        m = metrics.calculate_all_metrics(portfolio_values, returns)
        m['total_contributions'] = self.results['total_contributions'].iloc[-1]

        return m

    def get_annual_returns(self) -> pd.DataFrame:
        """
        Get annual returns table

        Returns:
            DataFrame with annual returns
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        return metrics.calculate_annual_returns(self.results['portfolio_value'])

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Get monthly returns table

        Returns:
            DataFrame with monthly returns by year
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        return metrics.calculate_monthly_returns(self.results['portfolio_value'])

    def get_annual_capital_changes(self) -> pd.DataFrame:
        """
        Get annual capital changes table showing:
        - Year
        - Portfolio value at year end
        - Cumulative contributions
        - Cumulative returns (value - contributions)

        Returns:
            DataFrame with annual capital changes
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        # Resample to year-end values
        yearly_values = self.results['portfolio_value'].resample('YE').last()
        yearly_contributions = self.results['total_contributions'].resample('YE').last()

        # Create DataFrame
        annual_changes = pd.DataFrame({
            'Year': yearly_values.index.year,
            'Portfolio_Value': yearly_values.values,
            'Cumulative_Contributions': yearly_contributions.values
        })

        # Calculate cumulative returns
        annual_changes['Cumulative_Returns'] = (
            annual_changes['Portfolio_Value'] - annual_changes['Cumulative_Contributions']
        )

        return annual_changes

    def get_holdings_over_time(self) -> pd.DataFrame:
        """
        Get portfolio holdings over time (value per asset)

        Returns:
            DataFrame with holdings value per asset over time
        """
        if self.results is None or self.monthly_prices is None:
            raise ValueError("Run backtest first")

        # This would require tracking holdings at each rebalance
        # For simplicity, return the weights
        holdings = pd.DataFrame(index=self.monthly_prices.index)
        for ticker in self.tickers:
            holdings[ticker] = self.allocation[ticker]

        return holdings

    def generate_report(self) -> str:
        """
        Generate a text report of backtest results

        Returns:
            Report string
        """
        if self.results is None:
            raise ValueError("Run backtest first")

        m = self.get_metrics()

        report = f"""
{'='*60}
GLOBAL PORTFOLIO BACKTEST REPORT
{'='*60}

PORTFOLIO ALLOCATION:
"""
        for ticker, weight in self.allocation.items():
            report += f"  {ticker}: {weight*100:.1f}%\n"

        report += f"""
BACKTEST PARAMETERS:
  Start Date: {self.start_date}
  End Date: {self.end_date}
  Initial Capital: ${self.initial_capital:,.0f}
  Monthly Contribution: ${self.monthly_contribution:,.0f}
  Rebalance Frequency: {self.rebalance_frequency}

PERFORMANCE METRICS:
  Total Return: {m['total_return']*100:.2f}%
  CAGR: {m['cagr']*100:.2f}%
  Max Drawdown: {m['max_drawdown']*100:.2f}%
  Volatility (Ann.): {m['volatility']*100:.2f}%
  Sharpe Ratio: {m['sharpe_ratio']:.2f}
  Sortino Ratio: {m['sortino_ratio']:.2f}
  Calmar Ratio: {m['calmar_ratio']:.2f}

FINAL RESULTS:
  Total Contributions: ${m['total_contributions']:,.0f}
  Final Portfolio Value: ${m['final_value']:,.0f}
  Total Gain: ${m['final_value'] - m['total_contributions']:,.0f}
{'='*60}
"""
        return report


def run_full_backtest(
    verbose: bool = True,
    **kwargs
) -> Tuple[PortfolioBacktest, pd.DataFrame, dict]:
    """
    Convenience function to run a full backtest

    Args:
        verbose: Print progress
        **kwargs: Override default config parameters

    Returns:
        Tuple of (backtest engine, results DataFrame, metrics dict)
    """
    bt = PortfolioBacktest(**kwargs)
    results = bt.run_backtest(verbose=verbose)
    m = bt.get_metrics()
    return bt, results, m


if __name__ == "__main__":
    # Run a test backtest
    bt = PortfolioBacktest()
    results = bt.run_backtest()
    print(bt.generate_report())