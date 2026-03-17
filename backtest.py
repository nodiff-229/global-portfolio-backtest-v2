"""
核心回测引擎 - 全球投资组合
实现定投策略与周期性再平衡

关键改进：
====================
1. 资产可用日历：预先定义每个资产的可用日期范围，避免前瞻偏差
2. 真实的再平衡逻辑：按当时市场价格买入卖出
3. XIRR集成：使用真正的内部收益率计算定投收益

回测流程：
====================
1. 获取所有资产的历史数据
2. 构建资产可用日历
3. 按月遍历时间线：
   - 检查当月哪些资产可用
   - 计算当前组合价值
   - 添加新投入资金
   - 判断是否需要再平衡
   - 执行再平衡或按比例买入
4. 计算绩效指标
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import data_fetcher as df
import metrics
import config


class AssetCalendar:
    """
    资产可用日历

    预先定义每个资产在回测期间的可用性，
    避免在回测过程中动态判断（造成前瞻偏差）。

    原理：
    - ETF成立后才能投资
    - 代理数据只能用于ETF成立前的近似
    - 投资者无法预知未来哪些ETF会成立

    使用：
    - 在回测开始时构建日历
    - 每月查询当月可用资产
    - 不可用资产的权重按比例分配给可用资产
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        proxy_mapping: Dict[str, str] = None
    ):
        """
        初始化资产可用日历

        Args:
            tickers: 资产代码列表
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            proxy_mapping: 代理映射 {ticker: proxy_ticker}
        """
        self.tickers = tickers
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.proxy_mapping = proxy_mapping or config.PROXY_MAPPING

        # ETF成立日期
        self.inception_dates = config.ETF_INCEPTION

        # 构建可用性日历
        self._build_availability_calendar()

    def _build_availability_calendar(self):
        """
        构建资产可用性日历

        对于每个资产，判断其在每个月是否可用：
        - 如果有代理数据，从回测开始就可用
        - 如果没有代理，从ETF成立日期开始可用
        """
        # 存储每个资产的可用开始日期
        self.availability_start = {}

        for ticker in self.tickers:
            # 检查是否有代理数据
            has_proxy = ticker in self.proxy_mapping

            if has_proxy:
                # 有代理数据，从回测开始日期就可用
                self.availability_start[ticker] = self.start_date
            else:
                # 无代理数据，从ETF成立日期开始可用
                inception = self.inception_dates.get(ticker, '1990-01-01')
                self.availability_start[ticker] = pd.Timestamp(inception)

    def get_available_tickers(self, date: pd.Timestamp) -> List[str]:
        """
        获取指定日期可用的资产列表

        Args:
            date: 查询日期

        Returns:
            可用资产代码列表
        """
        available = []
        for ticker in self.tickers:
            if date >= self.availability_start[ticker]:
                available.append(ticker)
        return available

    def get_adjusted_allocation(
        self,
        date: pd.Timestamp,
        target_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """
        获取调整后的资产配置权重

        当某些资产不可用时，将其权重按比例分配给可用资产。

        Args:
            date: 当前日期
            target_allocation: 目标配置权重 {ticker: weight}

        Returns:
            调整后的配置权重（仅包含可用资产）
        """
        available = self.get_available_tickers(date)
        missing = [t for t in self.tickers if t not in available]

        if not missing:
            return target_allocation.copy()

        # 计算可用资产的权重总和
        available_weight = sum(target_allocation[t] for t in available)

        if available_weight <= 0:
            # 所有资产都不可用，返回空字典
            return {}

        # 按比例重新分配权重
        adjusted = {}
        for ticker in available:
            adjusted[ticker] = target_allocation[ticker] / available_weight

        return adjusted

    def get_availability_info(self) -> pd.DataFrame:
        """
        获取资产可用性信息表

        Returns:
            DataFrame，包含每个资产的可用开始日期和代理信息
        """
        info = []
        for ticker in self.tickers:
            has_proxy = ticker in self.proxy_mapping
            proxy = self.proxy_mapping.get(ticker, None)
            inception = self.inception_dates.get(ticker, 'N/A')
            available_from = self.availability_start[ticker]

            info.append({
                'Ticker': ticker,
                'Inception Date': inception,
                'Has Proxy': has_proxy,
                'Proxy': proxy if has_proxy else None,
                'Available From': available_from.strftime('%Y-%m-%d')
            })

        return pd.DataFrame(info)


class PortfolioBacktest:
    """
    投资组合回测引擎

    实现定投策略与周期性再平衡。

    核心功能：
    - 支持自定义资产配置
    - 支持月度定投
    - 支持多种再平衡频率
    - 自动处理数据缺失情况
    - 计算完整的绩效指标
    """

    def __init__(
        self,
        allocation: Dict[str, float] = None,
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = None,
        monthly_contribution: float = None,
        rebalance_frequency: str = None,
        transaction_cost: float = 0.001,  # 交易成本（买卖价差+佣金），默认0.1%
        management_fee: float = None      # 年度管理费率，默认使用config
    ):
        """
        初始化回测引擎

        Args:
            allocation: 投资组合配置 {代码: 权重}
                        权重必须为正数且总和为1
            start_date: 回测开始日期 (YYYY-MM-DD)
            end_date: 回测结束日期 (YYYY-MM-DD)
            initial_capital: 初始资金
            monthly_contribution: 每月投入金额
            rebalance_frequency: 再平衡频率
                - 'monthly': 每月
                - 'quarterly': 每季度（3、6、9、12月）
                - 'semi-annual': 每半年（6、12月）
                - 'annual': 每年（12月）
            transaction_cost: 交易成本（单边），默认0.1%
                              例如：买入$10000，成本$10
            management_fee: 年度管理费率，默认0.65%（每月扣除年费率的1/12）
        """
        # 使用配置默认值
        self.allocation = allocation or config.PORTFOLIO_ALLOCATION.copy()
        self.start_date = start_date or config.BACKTEST_CONFIG['start_date']
        self.end_date = end_date or config.BACKTEST_CONFIG['end_date'] or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital or config.BACKTEST_CONFIG['initial_capital']
        self.monthly_contribution = monthly_contribution or config.BACKTEST_CONFIG['monthly_contribution']
        self.rebalance_frequency = rebalance_frequency or config.BACKTEST_CONFIG['rebalance_frequency']
        self.transaction_cost = transaction_cost
        self.management_fee = management_fee if management_fee is not None else config.MANAGEMENT_FEE

        # 验证权重总和
        total_weight = sum(self.allocation.values())
        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(f"配置权重总和必须为1.0，当前为 {total_weight:.4f}")

        # 资产列表
        self.tickers = list(self.allocation.keys())
        self.weights = list(self.allocation.values())

        # 数据存储
        self.prices = None              # 日频价格
        self.monthly_prices = None      # 月频价格
        self.asset_calendar = None      # 资产可用日历
        self.results = None             # 回测结果
        self.monthly_values = None      # 月度价值

        # 交易记录
        self.trade_log = []             # 交易日志

    def fetch_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        获取所有资产的价格数据

        Args:
            verbose: 是否打印进度信息

        Returns:
            DataFrame，包含调整后收盘价
        """
        if verbose:
            print(f"正在获取数据: {self.start_date} 至 {self.end_date}...")

        # 批量获取所有资产数据
        data = df.get_all_etf_data(self.tickers, self.start_date, self.end_date, verbose=verbose)

        # 提取调整后收盘价
        self.prices = df.get_adj_close_prices(data)

        # 检查是否有数据
        if self.prices.empty:
            raise ValueError(
                "无法获取任何资产数据。可能的原因：\n"
                "1. Yahoo Finance API 速率限制，请稍后重试\n"
                "2. 网络连接问题\n"
                "3. 资产代码无效"
            )

        # 重采样到月频
        self.monthly_prices = df.resample_monthly(self.prices)

        # 构建资产可用日历
        self.asset_calendar = AssetCalendar(
            self.tickers,
            self.start_date,
            self.end_date
        )

        if verbose:
            print(f"\n数据时间范围: {self.monthly_prices.index.min().strftime('%Y-%m-%d')} 至 {self.monthly_prices.index.max().strftime('%Y-%m-%d')}")
            print(f"可用资产: {list(self.monthly_prices.columns)}")

            # 显示资产可用性信息
            print("\n资产可用性信息:")
            availability_info = self.asset_calendar.get_availability_info()
            for _, row in availability_info.iterrows():
                proxy_info = f" (代理: {row['Proxy']})" if row['Has Proxy'] else ""
                print(f"  {row['Ticker']}: 成立于 {row['Inception Date']}, 可从 {row['Available From']}{proxy_info}")

        return self.monthly_prices

    def get_rebalance_dates(self) -> List[pd.Timestamp]:
        """
        获取再平衡日期列表

        根据再平衡频率生成需要执行再平衡的日期。

        Returns:
            再平衡日期列表
        """
        dates = self.monthly_prices.index

        if self.rebalance_frequency == 'monthly':
            # 每月末再平衡
            return list(dates)
        elif self.rebalance_frequency == 'quarterly':
            # 季度末再平衡（3、6、9、12月）
            return [d for d in dates if d.month in [3, 6, 9, 12]]
        elif self.rebalance_frequency == 'semi-annual':
            # 半年末再平衡（6、12月）
            return [d for d in dates if d.month in [6, 12]]
        elif self.rebalance_frequency == 'annual':
            # 年末再平衡（12月）
            return [d for d in dates if d.month == 12]
        else:
            raise ValueError(f"未知的再平衡频率: {self.rebalance_frequency}")

    def run_backtest(self, verbose: bool = True) -> pd.DataFrame:
        """
        执行回测

        核心回测逻辑：
        1. 遍历每个月
        2. 计算当前组合价值
        3. 添加新投入资金
        4. 判断是否需要再平衡
        5. 执行买入/卖出操作
        6. 记录结果

        Args:
            verbose: 是否打印进度信息

        Returns:
            DataFrame，包含每月的组合价值和投入记录
        """
        # 如果数据未获取，先获取
        if self.monthly_prices is None:
            self.fetch_data(verbose)

        if verbose:
            print(f"\n开始回测...")
            print(f"初始资金: ${self.initial_capital:,.0f}")
            print(f"每月投入: ${self.monthly_contribution:,.0f}")
            print(f"再平衡频率: {self.rebalance_frequency}")
            print(f"交易成本: {self.transaction_cost*100:.2f}%")
            print(f"年度管理费: {self.management_fee*100:.2f}% (每月扣除 {(self.management_fee/12)*100:.4f}%)")

        # 获取再平衡日期
        rebalance_dates = set(self.get_rebalance_dates())

        # 初始化持仓（每只资产的份额）
        holdings = {ticker: 0.0 for ticker in self.tickers}

        # 结果记录
        results = []
        total_contributions = 0.0
        total_management_fees = 0.0  # 累计管理费
        self.trade_log = []  # 清空交易日志

        # 月度管理费率（年费率的1/12）
        monthly_fee_rate = self.management_fee / 12

        # ================================================================
        # 遍历每个月
        # ================================================================
        for i, date in enumerate(self.monthly_prices.index):
            # 获取当月价格
            current_prices = self.monthly_prices.loc[date]

            # ============================================================
            # 步骤1：计算当前组合价值
            # ============================================================
            portfolio_value = 0.0
            for ticker in self.tickers:
                if ticker in current_prices and pd.notna(current_prices[ticker]):
                    portfolio_value += holdings[ticker] * current_prices[ticker]

            # ============================================================
            # 步骤2：添加投入资金
            # ============================================================
            if i == 0:
                # 第一个月：初始资金
                contribution = self.initial_capital
            else:
                # 后续月份：月度投入
                contribution = self.monthly_contribution

            total_contributions += contribution

            # ============================================================
            # 步骤3：获取当月可用资产（使用资产日历，避免前瞻偏差）
            # ============================================================
            available_tickers = self.asset_calendar.get_available_tickers(date)

            # 过滤出有价格数据的可用资产
            available_with_data = [
                t for t in available_tickers
                if t in current_prices and pd.notna(current_prices[t])
            ]

            # 获取调整后的配置权重
            adjusted_allocation = self.asset_calendar.get_adjusted_allocation(date, self.allocation)

            # ============================================================
            # 步骤4：判断是否需要再平衡
            # ============================================================
            is_rebalance_date = date in rebalance_dates or i == 0

            if is_rebalance_date:
                # ========================================================
                # 再平衡：卖出所有持仓，按目标权重重新买入
                # ========================================================

                # 总价值 = 当前组合价值 + 新投入
                total_value = portfolio_value + contribution

                # 计算交易成本
                # 卖出所有现有持仓的成本
                sell_cost = portfolio_value * self.transaction_cost
                total_value -= sell_cost

                # 计算目标持仓价值
                target_values = {}
                for ticker in available_with_data:
                    weight = adjusted_allocation.get(ticker, 0)
                    target_values[ticker] = total_value * weight

                # 计算买入成本
                buy_cost = sum(target_values.values()) * self.transaction_cost
                total_value -= buy_cost

                # 重新计算目标价值（扣除成本后）
                if total_value > 0:
                    for ticker in available_with_data:
                        weight = adjusted_allocation.get(ticker, 0)
                        target_values[ticker] = total_value * weight

                # 计算新的持仓份额
                new_holdings = {}
                for ticker in self.tickers:
                    if ticker in available_with_data and ticker in target_values:
                        new_holdings[ticker] = target_values[ticker] / current_prices[ticker]
                    else:
                        new_holdings[ticker] = 0.0

                # 记录交易
                if verbose and i == 0:
                    print(f"\n{date.strftime('%Y-%m')}: 初始建仓")
                    for ticker in available_with_data:
                        if ticker in target_values:
                            print(f"  {ticker}: ${target_values[ticker]:,.0f} ({adjusted_allocation.get(ticker, 0)*100:.1f}%)")

                # 更新持仓
                holdings = new_holdings

            else:
                # ========================================================
                # 非再平衡月份：按比例买入新投入资金
                # ========================================================
                if contribution > 0 and available_with_data:
                    for ticker in available_with_data:
                        weight = adjusted_allocation.get(ticker, 0)
                        # 计算买入金额（扣除交易成本）
                        buy_amount = contribution * weight
                        buy_cost = buy_amount * self.transaction_cost
                        net_buy_amount = buy_amount - buy_cost

                        if net_buy_amount > 0:
                            shares = net_buy_amount / current_prices[ticker]
                            holdings[ticker] = holdings.get(ticker, 0) + shares

            # ============================================================
            # 步骤5：扣除月度管理费
            # ============================================================
            # 计算当前组合价值用于扣除管理费
            current_portfolio_value = 0.0
            for ticker in self.tickers:
                if ticker in current_prices and pd.notna(current_prices[ticker]):
                    current_portfolio_value += holdings[ticker] * current_prices[ticker]

            # 扣除月度管理费（按资产净值比例）
            if current_portfolio_value > 0 and monthly_fee_rate > 0:
                management_fee = current_portfolio_value * monthly_fee_rate
                total_management_fees += management_fee

                # 按比例减少各资产持仓份额
                fee_ratio = 1 - monthly_fee_rate
                for ticker in self.tickers:
                    holdings[ticker] = holdings[ticker] * fee_ratio

            # ============================================================
            # 步骤6：计算期末组合价值
            # ============================================================
            final_value = 0.0
            for ticker in self.tickers:
                if ticker in current_prices and pd.notna(current_prices[ticker]):
                    final_value += holdings[ticker] * current_prices[ticker]

            # ============================================================
            # 步骤7：记录结果
            # ============================================================
            results.append({
                'date': date,
                'portfolio_value': final_value,
                'contribution': contribution,
                'total_contributions': total_contributions,
                'total_management_fees': total_management_fees,
                'rebalanced': is_rebalance_date
            })

        # 转换为DataFrame
        self.results = pd.DataFrame(results)
        self.results.set_index('date', inplace=True)

        if verbose:
            print(f"\n回测完成！")
            print(f"总投入: ${total_contributions:,.0f}")
            print(f"累计管理费: ${total_management_fees:,.0f}")
            print(f"期末价值: ${final_value:,.0f}")
            print(f"总收益: ${final_value - total_contributions:,.0f}")

        return self.results

    def get_metrics(self) -> dict:
        """
        计算投资组合绩效指标

        使用XIRR方法计算定投策略的真实年化收益率。

        Returns:
            字典，包含各项绩效指标
        """
        if self.results is None:
            raise ValueError("请先执行回测 (run_backtest)")

        portfolio_values = self.results['portfolio_value']
        returns = portfolio_values.pct_change().dropna()
        contributions = self.results['contribution']

        # 计算所有指标（包括XIRR）
        m = metrics.calculate_all_metrics(portfolio_values, returns, contributions)
        m['total_contributions'] = self.results['total_contributions'].iloc[-1]
        m['total_management_fees'] = self.results['total_management_fees'].iloc[-1]

        return m

    def get_annual_returns(self) -> pd.DataFrame:
        """
        获取年度收益表

        Returns:
            DataFrame，包含每年的收益率
        """
        if self.results is None:
            raise ValueError("请先执行回测")

        return metrics.calculate_annual_returns(self.results['portfolio_value'])

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        获取月度收益表

        Returns:
            DataFrame，行=年份，列=月份，值=收益率
        """
        if self.results is None:
            raise ValueError("请先执行回测")

        return metrics.calculate_monthly_returns(self.results['portfolio_value'])

    def get_annual_capital_changes(self) -> pd.DataFrame:
        """
        获取年度资金变化表

        展示每年年末的组合价值、累计投入和累计收益。

        Returns:
            DataFrame，包含：
            - Year: 年份
            - Portfolio_Value: 组合价值
            - Cumulative_Contributions: 累计投入
            - Cumulative_Returns: 累计收益
        """
        if self.results is None:
            raise ValueError("请先执行回测")

        # 重采样到年末
        yearly_values = self.results['portfolio_value'].resample('YE').last()
        yearly_contributions = self.results['total_contributions'].resample('YE').last()

        # 构建结果
        annual_changes = pd.DataFrame({
            'Year': yearly_values.index.year,
            'Portfolio_Value': yearly_values.values,
            'Cumulative_Contributions': yearly_contributions.values
        })

        # 计算累计收益 = 组合价值 - 累计投入
        annual_changes['Cumulative_Returns'] = (
            annual_changes['Portfolio_Value'] - annual_changes['Cumulative_Contributions']
        )

        return annual_changes

    def get_holdings_over_time(self) -> pd.DataFrame:
        """
        获取持仓权重时间序列

        Returns:
            DataFrame，包含各资产的配置权重
        """
        if self.results is None or self.monthly_prices is None:
            raise ValueError("请先执行回测")

        holdings = pd.DataFrame(index=self.monthly_prices.index)
        for ticker in self.tickers:
            holdings[ticker] = self.allocation[ticker]

        return holdings

    def generate_report(self) -> str:
        """
        生成文本格式的回测报告

        Returns:
            报告字符串
        """
        if self.results is None:
            raise ValueError("请先执行回测")

        m = self.get_metrics()

        report = f"""
{'='*60}
全球投资组合回测报告
{'='*60}

投资组合配置:
"""
        for ticker, weight in self.allocation.items():
            report += f"  {ticker}: {weight*100:.1f}%\n"

        report += f"""
回测参数:
  开始日期: {self.start_date}
  结束日期: {self.end_date}
  初始资金: ${self.initial_capital:,.0f}
  每月投入: ${self.monthly_contribution:,.0f}
  再平衡频率: {self.rebalance_frequency}
  交易成本: {self.transaction_cost*100:.2f}%
  年度管理费: {self.management_fee*100:.2f}%

绩效指标:
  总收益率: {m['total_return']*100:.2f}%
  年化收益率 (XIRR): {m['cagr']*100:.2f}%
  最大回撤: {m['max_drawdown']*100:.2f}%
  年化波动率: {m['volatility']*100:.2f}%
  夏普比率: {m['sharpe_ratio']:.2f}
  索提诺比率: {m['sortino_ratio']:.2f}
  卡玛比率: {m['calmar_ratio']:.2f}

最终结果:
  累计投入: ${m['total_contributions']:,.0f}
  期末价值: ${m['final_value']:,.0f}
  总收益: ${m['final_value'] - m['total_contributions']:,.0f}
{'='*60}
"""
        return report


def run_full_backtest(
    verbose: bool = True,
    **kwargs
) -> Tuple[PortfolioBacktest, pd.DataFrame, dict]:
    """
    便捷函数：执行完整回测

    Args:
        verbose: 是否打印进度
        **kwargs: 覆盖默认参数
            - allocation: 投资组合配置
            - start_date: 开始日期
            - end_date: 结束日期
            - initial_capital: 初始资金
            - monthly_contribution: 每月投入
            - rebalance_frequency: 再平衡频率

    Returns:
        元组 (回测引擎, 结果DataFrame, 指标字典)
    """
    bt = PortfolioBacktest(**kwargs)
    results = bt.run_backtest(verbose=verbose)
    m = bt.get_metrics()
    return bt, results, m


if __name__ == "__main__":
    # 运行测试回测
    print("=" * 60)
    print("投资组合回测测试")
    print("=" * 60)

    bt = PortfolioBacktest()
    results = bt.run_backtest()
    print(bt.generate_report())