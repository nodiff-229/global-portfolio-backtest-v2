"""
财务指标计算模块
计算 CAGR、最大回撤、夏普比率、索提诺比率等

关键改进：
====================
1. XIRR计算：使用真正的内部收益率方法计算定投策略的年化收益
2. 精确的CAGR：区分一次性投入和定期投入的计算方式

XIRR说明：
====================
XIRR（Extended Internal Rate of Return）是处理不规则现金流和日期的内部收益率计算方法。

为什么定投策略必须用XIRR：
- 定投是每月投入资金，不是一次性投入
- 每笔投入的时间不同，其时间价值也不同
- 简单CAGR假设所有资金在开始时一次性投入，会低估真实收益率

XIRR计算原理：
- 找到使NPV（净现值）= 0 的折现率
- NPV = Σ(现金流 / (1+折现率)^((日期-起始日期)/365.25))
- 使用数值方法（牛顿法或布伦特法）求解
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.optimize import newton, brentq
import config


def calculate_xirr(
    dates: pd.Series,
    amounts: pd.Series,
    guess: float = 0.1
) -> float:
    """
    计算XIRR（扩展内部收益率）

    XIRR是处理不规则现金流和日期的内部收益率计算方法，
    是计算定投策略收益率的正确方法。

    核心原理：
    - 找到使净现值（NPV）等于0的年化折现率
    - NPV = Σ(现金流 / (1+折现率)^((日期-起始日期)/365.25))

    现金流符号约定：
    - 负数：现金流出（投入资金）
    - 正数：现金流入（赎回资金）

    Args:
        dates: 日期序列（pd.Series，类型为datetime）
        amounts: 现金流金额序列（pd.Series）
                 投入为负数，赎回为正数
        guess: 初始猜测值，默认0.1（即10%）
               用于数值求解的起始点

    Returns:
        XIRR年化收益率（小数形式，如0.08表示8%）

    Raises:
        ValueError: 如果无法收敛到解

    Example:
        dates = pd.Series([
            pd.Timestamp('2020-01-01'),  # 投入$1000
            pd.Timestamp('2020-02-01'),  # 投入$1000
            pd.Timestamp('2020-03-01'),  # 赎回$2100
        ])
        amounts = pd.Series([-1000, -1000, 2100])
        xirr = calculate_xirr(dates, amounts)  # 约 0.10 (10%)

    Note:
        - 投入资金应为负值（现金流出）
        - 最后一个值通常是正值（当前价值或赎回金额）
        - 如果求解失败，返回0.0
    """
    # 参数验证
    if len(dates) != len(amounts):
        raise ValueError("日期序列和金额序列长度必须相同")

    if len(dates) < 2:
        return 0.0

    # 确保日期是datetime类型
    dates = pd.to_datetime(dates)

    # 转换为列表以便计算
    dates_list = dates.tolist()
    amounts_list = amounts.tolist()

    # 计算起始日期（用于计算天数差）
    start_date = min(dates_list)

    def xnpv(rate: float) -> float:
        """
        计算净现值（NPV）

        NPV = Σ(现金流 / (1+折现率)^((日期-起始日期)/365.25))

        Args:
            rate: 折现率（年化）

        Returns:
            净现值
        """
        total = 0.0
        for date, amount in zip(dates_list, amounts_list):
            # 计算从起始日期到当前日期的年数
            days = (date - start_date).days
            years = days / 365.25

            # 折现到现值
            if years == 0:
                # 起始日期的现金流不折现
                total += amount
            else:
                total += amount / ((1 + rate) ** years)

        return total

    # 使用牛顿法求解 xnpv(rate) = 0
    try:
        # 首先尝试牛顿法（通常更快）
        xirr = newton(xnpv, guess, maxiter=100)
        return xirr
    except (RuntimeError, ValueError):
        # 牛顿法失败，尝试布伦特法
        try:
            # 在合理范围内搜索
            # 收益率通常在 -100% 到 +1000% 之间
            xirr = brentq(xnpv, -0.99, 10.0, maxiter=100)
            return xirr
        except (RuntimeError, ValueError):
            # 两种方法都失败，返回0
            print("警告: XIRR计算未能收敛，返回0")
            return 0.0


def calculate_cagr(
    portfolio_values: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    contributions: pd.Series = None,
    contribution_dates: pd.Series = None,
    use_xirr: bool = True
) -> float:
    """
    计算复合年化增长率（CAGR）

    对于定投策略，推荐使用XIRR方法计算，因为：
    - XIRR考虑了每笔投入的时间价值
    - XIRR能准确反映定投策略的真实收益率
    - 简单CAGR假设一次性投入，会低估定投收益

    Args:
        portfolio_values: 时间序列的投资组合价值
                          索引为日期，值为组合总价值
        start_date: 回测开始日期
        end_date: 回测结束日期
        contributions: 每月投入金额的时间序列（用于XIRR计算）
                       索引为日期，值为投入金额
        contribution_dates: 投入日期序列（可选）
        use_xirr: 是否使用XIRR计算（默认True）
                  对于定投策略，应设为True

    Returns:
        CAGR年化收益率（小数形式，如0.08表示8%）

    计算方法说明：
    ===============
    1. 定投策略（use_xirr=True）：
       - 使用XIRR计算，考虑每笔投入的时间
       - 每月投入视为负现金流，期末价值视为正现金流
       - 计算使NPV=0的内部收益率

    2. 一次性投入（use_xirr=False或无contributions）：
       - 使用简单CAGR公式
       - CAGR = (期末价值/期初价值)^(1/年数) - 1
    """
    if len(portfolio_values) < 2:
        return 0.0

    # 获取起始和结束价值
    start_value = portfolio_values.iloc[0]
    end_value = portfolio_values.iloc[-1]

    # 计算年数
    years = (end_date - start_date).days / 365.25

    if years <= 0:
        return 0.0

    # 如果有投入数据且使用XIRR，使用XIRR计算
    if use_xirr and contributions is not None and len(contributions) > 0:
        # 构建现金流序列
        # 投入为负数（现金流出），期末价值为正数（现金流入）

        # 获取投入日期和金额
        if contribution_dates is not None:
            dates = contribution_dates.copy()
        else:
            dates = contributions.index.to_series()

        amounts = contributions.copy()

        # 投入金额为负（现金流出）
        amounts = -amounts

        # 添加期末价值（现金流入）
        final_date = pd.Series([end_date])
        final_amount = pd.Series([end_value])

        dates = pd.concat([dates.reset_index(drop=True), final_date.reset_index(drop=True)], ignore_index=True)
        amounts = pd.concat([amounts.reset_index(drop=True), final_amount.reset_index(drop=True)], ignore_index=True)

        # 计算XIRR
        xirr = calculate_xirr(dates, amounts)

        return xirr

    else:
        # 简单CAGR公式（一次性投入）
        if start_value <= 0:
            return 0.0

        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr


def calculate_max_drawdown(
    portfolio_values: pd.Series
) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    计算最大回撤（Maximum Drawdown）

    最大回撤是从峰值到谷值的最大跌幅，
    是衡量投资风险的重要指标。

    计算方法：
    1. 计算每个时间点的历史最高值（running max）
    2. 计算每个时间点的回撤 = (当前值 - 历史最高值) / 历史最高值
    3. 找到最大的负值即为最大回撤

    Args:
        portfolio_values: 时间序列的投资组合价值
                          索引为日期，值为组合总价值

    Returns:
        元组 (最大回撤, 回撤开始日期, 回撤结束日期)
        - 最大回撤：负数，如 -0.20 表示最大回撤20%
        - 回撤开始日期：峰值日期（回撤开始的那个高点）
        - 回撤结束日期：谷值日期（最大回撤时的低点）

    Example:
        values = pd.Series([100, 110, 105, 95, 100, 90],
                          index=pd.date_range('2020-01-01', periods=6))
        max_dd, peak, trough = calculate_max_drawdown(values)
        # max_dd = -0.1818 (从110跌到90，跌幅18.18%)
        # peak = 2020-01-02 (值110)
        # trough = 2020-01-06 (值90)
    """
    if len(portfolio_values) < 2:
        return 0.0, None, None

    # 计算历史最高值（running maximum）
    running_max = portfolio_values.cummax()

    # 计算回撤 = (当前值 - 历史最高值) / 历史最高值
    drawdown = (portfolio_values - running_max) / running_max

    # 找到最大回撤（最小值，即最负的值）
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # 找到峰值日期（最大回撤开始前的最高点）
    peak_idx = running_max.loc[:max_dd_idx].idxmax()

    return max_dd, peak_idx, max_dd_idx


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = None,
    periods_per_year: int = 12
) -> float:
    """
    计算夏普比率（Sharpe Ratio）

    夏普比率衡量每单位风险获得的超额收益。
    是最常用的风险调整收益指标。

    公式：
    夏普比率 = (平均收益 - 无风险利率) / 收益率标准差 * sqrt(年化因子)

    Args:
        returns: 周期收益率序列（如月度收益率）
        risk_free_rate: 年化无风险利率（默认从config读取）
                        如 0.02 表示2%年化
        periods_per_year: 每年的周期数
                          月度数据=12，季度数据=4，日度数据≈252

    Returns:
        夏普比率（无量纲）
        - > 1: 较好
        - > 2: 很好
        - > 3: 优秀

    Note:
        - 使用几何平均计算超额收益
        - 标准差年化 = 月度标准差 * sqrt(12)
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    if len(returns) < 2:
        return 0.0

    # 将年化无风险利率转换为周期利率
    rf_per_period = risk_free_rate / periods_per_year

    # 计算超额收益
    excess_returns = returns - rf_per_period

    # 计算平均值和标准差
    mean_excess = excess_returns.mean()
    std_returns = returns.std()

    if std_returns == 0 or pd.isna(std_returns):
        return 0.0

    # 夏普比率 = 平均超额收益 / 标准差 * sqrt(年化因子)
    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = None,
    periods_per_year: int = 12
) -> float:
    """
    计算索提诺比率（Sortino Ratio）

    索提诺比率是夏普比率的改进版本，
    只考虑下行风险（负收益的标准差），而不是全部波动。

    为什么用索提诺比率：
    - 夏普比率惩罚所有波动，包括上涨波动
    - 索提诺比率只惩罚下行波动
    - 对于投资者来说，下行风险才是真正的风险

    公式：
    索提诺比率 = (平均收益 - 无风险利率) / 下行标准差 * sqrt(年化因子)

    Args:
        returns: 周期收益率序列（如月度收益率）
        risk_free_rate: 年化无风险利率（默认从config读取）
        periods_per_year: 每年的周期数

    Returns:
        索提诺比率（无量纲）
        通常高于夏普比率，因为分母较小

    Note:
        - 如果没有负收益，返回无穷大
        - 下行标准差只计算收益率为负的数据点
    """
    if risk_free_rate is None:
        risk_free_rate = config.RISK_FREE_RATE

    if len(returns) < 2:
        return 0.0

    # 将年化无风险利率转换为周期利率
    rf_per_period = risk_free_rate / periods_per_year

    # 计算超额收益
    excess_returns = returns - rf_per_period

    # 下行标准差：只计算负收益的标准差
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # 没有负收益，理论上无下行风险

    downside_std = downside_returns.std()

    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    # 索提诺比率 = 平均超额收益 / 下行标准差 * sqrt(年化因子)
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 12
) -> float:
    """
    计算年化波动率（Volatility）

    波动率衡量收益率的变化程度，是风险的基本度量。

    公式：
    年化波动率 = 周期标准差 * sqrt(年化因子)

    Args:
        returns: 周期收益率序列（如月度收益率）
        periods_per_year: 每年的周期数
                          月度数据=12，日度数据≈252

    Returns:
        年化波动率（小数形式，如0.15表示15%）

    Example:
        月度标准差 0.04，年化波动率 = 0.04 * sqrt(12) ≈ 0.138
    """
    if len(returns) < 2:
        return 0.0

    return returns.std() * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown: float
) -> float:
    """
    计算卡玛比率（Calmar Ratio）

    卡玛比率衡量每单位最大回撤获得的年化收益。

    公式：
    卡玛比率 = CAGR / |最大回撤|

    Args:
        cagr: 复合年化增长率（小数形式）
        max_drawdown: 最大回撤（负数，如-0.20表示20%回撤）

    Returns:
        卡玛比率（无量纲）
        - > 1: 较好（年化收益大于最大回撤）
        - > 3: 优秀

    Note:
        - 最大回撤为0时返回无穷大（理论上不可能发生）
        - 这是衡量风险调整收益的另一个角度
    """
    if max_drawdown == 0:
        return float('inf') if cagr > 0 else 0.0

    return cagr / abs(max_drawdown)


def calculate_all_metrics(
    portfolio_values: pd.Series,
    returns: pd.Series = None,
    contributions: pd.Series = None
) -> dict:
    """
    计算所有投资组合指标

    这是主函数，汇总计算所有关键指标。

    Args:
        portfolio_values: 时间序列的投资组合价值
        returns: 周期收益率（如未提供则自动计算）
        contributions: 每月投入金额的时间序列（用于XIRR计算）

    Returns:
        字典，包含以下指标：
        - cagr: 复合年化增长率（使用XIRR计算）
        - max_drawdown: 最大回撤
        - max_dd_peak_date: 回撤峰值日期
        - max_dd_trough_date: 回撤谷值日期
        - sharpe_ratio: 夏普比率
        - sortino_ratio: 索提诺比率
        - volatility: 年化波动率
        - calmar_ratio: 卡玛比率
        - total_return: 总收益率
        - start_date: 开始日期
        - end_date: 结束日期
        - total_contributions: 总投入金额
        - final_value: 期末价值
    """
    if len(portfolio_values) < 2:
        return {
            'cagr': 0,
            'max_drawdown': 0,
            'max_dd_peak_date': None,
            'max_dd_trough_date': None,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'volatility': 0,
            'calmar_ratio': 0,
            'total_return': 0,
            'start_date': None,
            'end_date': None,
            'total_contributions': 0,
            'final_value': 0
        }

    # 如果未提供收益率，自动计算
    if returns is None:
        returns = portfolio_values.pct_change().dropna()

    # 获取日期范围
    start_date = portfolio_values.index[0]
    end_date = portfolio_values.index[-1]

    # 计算各项指标
    cagr = calculate_cagr(portfolio_values, start_date, end_date, contributions)
    max_dd, dd_peak, dd_trough = calculate_max_drawdown(portfolio_values)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    vol = calculate_volatility(returns)
    calmar = calculate_calmar_ratio(cagr, max_dd)

    # 计算总收益率
    if contributions is not None:
        # 对于定投策略：总收益 = (期末价值 - 总投入) / 总投入
        total_invested = contributions.sum() + portfolio_values.iloc[0]
        total_return = (portfolio_values.iloc[-1] - total_invested) / total_invested
    else:
        # 对于一次性投入：总收益 = 期末价值 / 期初价值 - 1
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
        'total_contributions': contributions.sum() if contributions is not None else None,
        'final_value': portfolio_values.iloc[-1]
    }


def calculate_annual_returns(portfolio_values: pd.Series) -> pd.DataFrame:
    """
    计算年度收益率

    将投资组合价值重采样到年度，计算每年的收益率。

    Args:
        portfolio_values: 时间序列的投资组合价值

    Returns:
        DataFrame，包含两列：
        - Year: 年份
        - Annual_Return: 年度收益率（小数形式）

    Example:
            Year  Annual_Return
        0   2020       0.15
        1   2021       0.28
        2   2022      -0.12
    """
    if len(portfolio_values) < 2:
        return pd.DataFrame()

    # 重采样到年末值
    yearly = portfolio_values.resample('YE').last()

    # 计算年度收益率
    annual_returns = yearly.pct_change().dropna()

    # 创建结果DataFrame
    df = pd.DataFrame({
        'Year': annual_returns.index.year,
        'Annual_Return': annual_returns.values
    })

    return df


def calculate_monthly_returns(portfolio_values: pd.Series) -> pd.DataFrame:
    """
    计算月度收益率（按年份和月份分组）

    将收益率按年份（行）和月份（列）组织，便于热力图展示。

    Args:
        portfolio_values: 时间序列的投资组合价值

    Returns:
        DataFrame，行为年份，列为月份（Jan-Dec）和年度收益（Year）
        例如：
               Jan    Feb    Mar  ...   Dec    Year
        2020  0.02  -0.01   0.05  ... 0.03   0.15
        2021  0.01   0.02   0.03  ... 0.02   0.28

    Note:
        - 月份列显示月度收益率
        - Year列显示年度收益率
    """
    if len(portfolio_values) < 2:
        return pd.DataFrame()

    # 计算月度收益率
    monthly_returns = portfolio_values.resample('ME').last().pct_change().dropna()

    # 创建DataFrame，包含年份和月份
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    # 转换为宽表格式（行=年份，列=月份）
    pivot = df.pivot(index='Year', columns='Month', values='Return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 添加年度收益率
    yearly = portfolio_values.resample('YE').last()
    annual = yearly.pct_change()

    # 对齐年份（annual第一年为NaN）
    annual_aligned = annual.iloc[1:]

    # 确保长度匹配
    if len(annual_aligned) > len(pivot):
        annual_aligned = annual_aligned.iloc[:len(pivot)]
    elif len(annual_aligned) < len(pivot):
        pivot = pivot.iloc[:len(annual_aligned)]

    pivot['Year'] = annual_aligned.values

    return pivot


if __name__ == "__main__":
    # 测试指标计算
    print("=" * 60)
    print("财务指标计算模块测试")
    print("=" * 60)

    # 创建模拟数据
    dates = pd.date_range('2010-01-01', '2020-12-31', freq='ME')
    values = pd.Series([100000 * (1.007 ** i) for i in range(len(dates))], index=dates)

    # 创建模拟投入数据
    contributions = pd.Series([1000] * len(dates), index=dates)
    contributions.iloc[0] = 10000  # 初始投入

    # 计算指标
    metrics = calculate_all_metrics(values, contributions=contributions)

    print("\n计算结果：")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 测试XIRR
    print("\n" + "=" * 60)
    print("XIRR 测试")
    print("=" * 60)

    # 简单测试案例
    test_dates = pd.Series([
        pd.Timestamp('2020-01-01'),
        pd.Timestamp('2020-02-01'),
        pd.Timestamp('2020-03-01'),
    ])
    test_amounts = pd.Series([-1000, -1000, 2100])

    xirr = calculate_xirr(test_dates, test_amounts)
    print(f"XIRR: {xirr*100:.2f}% (预期约10%)")