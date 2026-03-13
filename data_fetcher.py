"""
数据获取模块 - 投资组合回测系统
处理 yfinance 数据获取和缓存

价格缩放方法说明：
====================
问题背景：
- 代理数据（如黄金期货 GC=F）和实际ETF（如 GLD）价格水平不同
- 直接拼接会导致价格跳变（如 $400 -> $44 的虚假暴跌）
- 这会严重扭曲回测结果

解决方案：
- 计算重叠期价格比率，将代理价格缩放到ETF价格水平
- 使用中位数比率提高稳定性，避免极端值影响
- 确保价格序列连续，反映真实的收益率变化

示例：
- GC=F 价格 $400, GLD 价格 $44
- 价格比率 = 44/400 = 0.11
- 缩放后：$400 * 0.11 = $44，与GLD价格一致
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
    获取ETF历史价格数据，支持代理数据拼接

    对于成立较晚的ETF，使用代理数据进行回溯填充。
    采用价格缩放方法确保价格序列连续，避免虚假跳变。

    价格缩放原理：
    1. 代理数据（如黄金期货 GC=F）和实际ETF（如 GLD）价格水平不同
    2. 直接拼接会导致价格跳变（如 $400 -> $44 的虚假暴跌）
    3. 解决方案：计算重叠期价格比率，将代理价格缩放到ETF价格水平

    缩放步骤：
    1. 获取代理数据（从回测开始日期到ETF成立后一段时间）
    2. 获取实际ETF数据（从ETF成立日期开始）
    3. 找到重叠期，计算价格比率（ETF价格 / 代理价格）
    4. 将代理数据的OHLC价格全部乘以价格比率
    5. 拼接缩放后的代理数据和实际ETF数据

    Args:
        ticker: ETF代码（如 'GLD', 'QQQ'）
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
        use_proxy: 是否使用代理数据进行回溯填充
        verbose: 是否打印详细的处理信息

    Returns:
        DataFrame包含OHLCV数据，价格已缩放到ETF水平
        列包括：Open, High, Low, Close, Adj Close, Volume

    Raises:
        无直接抛出异常，数据获取失败时返回空DataFrame
    """
    # 如果未指定结束日期，使用今天
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 获取ETF成立日期
    inception = config.ETF_INCEPTION.get(ticker, '1990-01-01')

    # 检查是否需要代理数据（仅当开始日期早于ETF成立日期时）
    if use_proxy and ticker in config.PROXY_MAPPING and start_date < inception:
        proxy_ticker = config.PROXY_MAPPING[ticker]

        if verbose:
            print(f"  {ticker}: 检测到ETF成立日期 {inception} 晚于回测开始日期 {start_date}")
            print(f"  {ticker}: 使用代理数据 {proxy_ticker} 进行回溯填充")

        # ============================================================
        # 步骤1：获取代理数据
        # 需要获取到ETF成立后一段时间的数据，用于计算价格比率
        # ============================================================
        proxy_end_date = (datetime.strptime(inception, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
        proxy_data = yf.download(
            proxy_ticker,
            start=start_date,
            end=proxy_end_date,
            progress=False
        )

        # 处理 yfinance 返回的 MultiIndex 列名
        if isinstance(proxy_data.columns, pd.MultiIndex):
            proxy_data.columns = proxy_data.columns.get_level_values(0)

        # ============================================================
        # 步骤2：获取实际ETF数据（从ETF成立日期到结束日期）
        # ============================================================
        actual_data = yf.download(
            ticker,
            start=inception,
            end=end_date,
            progress=False
        )

        # 处理 MultiIndex 列名
        if isinstance(actual_data.columns, pd.MultiIndex):
            actual_data.columns = actual_data.columns.get_level_values(0)

        # ============================================================
        # 检查数据有效性
        # ============================================================
        if proxy_data.empty and actual_data.empty:
            if verbose:
                print(f"  {ticker}: 警告 - 代理数据和实际ETF数据均为空")
            return pd.DataFrame()

        if proxy_data.empty:
            if verbose:
                print(f"  {ticker}: 代理数据为空，仅使用实际ETF数据 ({len(actual_data)} 行)")
            return actual_data

        if actual_data.empty:
            if verbose:
                print(f"  {ticker}: 实际ETF数据为空，仅使用代理数据 ({len(proxy_data)} 行)")
            return proxy_data

        # ============================================================
        # 步骤3：找到重叠期，计算价格比率
        # 重叠期：代理数据和实际ETF数据都有数据的时期
        # ============================================================
        overlap_start = actual_data.index.min()
        overlap_end = proxy_data.index.max()

        # 检查是否有重叠
        if overlap_start > overlap_end:
            if verbose:
                print(f"  {ticker}: 警告 - 代理数据与实际ETF数据无重叠期，无法计算价格比率")
                print(f"  {ticker}: 代理数据结束于 {proxy_data.index.max().strftime('%Y-%m-%d')}，ETF数据开始于 {actual_data.index.min().strftime('%Y-%m-%d')}")
            # 直接拼接，但会有价格跳变风险（这是最后的备选方案）
            combined = pd.concat([proxy_data, actual_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            return combined

        # 找到重叠期内的数据
        overlap_proxy = proxy_data.loc[overlap_start:overlap_end]
        overlap_actual = actual_data.loc[overlap_start:overlap_end]

        # 如果重叠期数据不足，尝试扩展代理数据的获取范围
        if len(overlap_proxy) == 0 or len(overlap_actual) == 0:
            # 尝试使用ETF成立后第一个月的数据作为重叠期
            first_month_end = (datetime.strptime(inception, '%Y-%m-%d') + timedelta(days=30)).strftime('%Y-%m-%d')
            overlap_actual = actual_data.loc[inception:first_month_end]
            overlap_proxy = proxy_data.loc[inception:first_month_end]

        # ============================================================
        # 计算价格比率（使用收盘价的中位数比率，避免极端值影响）
        # ============================================================
        if len(overlap_proxy) > 0 and len(overlap_actual) > 0:
            # 找到两个数据集都有的日期
            common_dates = overlap_proxy.index.intersection(overlap_actual.index)

            if len(common_dates) > 0:
                # 计算多个日期的价格比率，取中位数以提高稳定性
                price_ratios = []
                for date in common_dates:
                    proxy_close = overlap_proxy.loc[date, 'Close']
                    actual_close = overlap_actual.loc[date, 'Close']
                    # 确保价格有效（大于0且不为NaN）
                    if pd.notna(proxy_close) and pd.notna(actual_close) and proxy_close > 0 and actual_close > 0:
                        price_ratios.append(actual_close / proxy_close)

                if len(price_ratios) > 0:
                    price_ratio = pd.Series(price_ratios).median()
                    if verbose:
                        print(f"  {ticker}: 找到 {len(common_dates)} 个重叠日期")
                        print(f"  {ticker}: 价格缩放比率 (ETF价格/代理价格) = {price_ratio:.6f}")
                        # 显示示例
                        sample_date = common_dates[0]
                        print(f"  {ticker}: 示例 - 代理价格 ${overlap_proxy.loc[sample_date, 'Close']:.2f} -> 缩放后 ${overlap_proxy.loc[sample_date, 'Close'] * price_ratio:.2f} (实际ETF价格 ${overlap_actual.loc[sample_date, 'Close']:.2f})")
                else:
                    price_ratio = 1.0
                    if verbose:
                        print(f"  {ticker}: 警告 - 无法计算有效价格比率，使用默认比率 1.0")
            else:
                price_ratio = 1.0
                if verbose:
                    print(f"  {ticker}: 警告 - 无共同日期，使用默认价格比率 1.0")
        else:
            price_ratio = 1.0
            if verbose:
                print(f"  {ticker}: 警告 - 重叠期数据不足，使用默认价格比率 1.0")

        # ============================================================
        # 步骤4：缩放代理数据的所有OHLCV价格列
        # 注意：不缩放 Volume，因为成交量不应该随价格变化
        # ============================================================
        ohlc_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaled_proxy = proxy_data.copy()

        for col in ohlc_columns:
            if col in scaled_proxy.columns:
                scaled_proxy[col] = scaled_proxy[col] * price_ratio

        # ============================================================
        # 步骤5：拼接缩放后的代理数据和实际ETF数据
        # 只保留ETF成立日期之前的代理数据（避免重复）
        # ============================================================
        scaled_proxy_before_inception = scaled_proxy[scaled_proxy.index < overlap_start]

        # 拼接数据
        combined = pd.concat([scaled_proxy_before_inception, actual_data])
        combined = combined[~combined.index.duplicated(keep='last')]

        if verbose:
            print(f"  {ticker}: 数据拼接完成 - 代理数据 {len(scaled_proxy_before_inception)} 行 + 实际ETF数据 {len(actual_data)} 行 = 总计 {len(combined)} 行")

        return combined

    # ================================================================
    # 直接下载（不需要代理，或开始日期在ETF成立日期之后）
    # ================================================================
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # 处理 MultiIndex 列名
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if verbose and not data.empty:
        print(f"  {ticker}: 直接获取数据 {len(data)} 行（无需代理）")

    return data


def get_all_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    批量获取多个ETF的历史价格数据

    遍历所有ETF代码，分别获取数据，支持代理数据回溯填充。

    Args:
        tickers: ETF代码列表（如 ['QQQ', 'SPY', 'GLD']）
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD，默认为今天
        verbose: 是否打印详细的处理信息

    Returns:
        字典，键为ETF代码，值为对应的DataFrame
        例如：{'QQQ': DataFrame, 'SPY': DataFrame, ...}

    Note:
        获取失败的ETF不会出现在返回的字典中
    """
    data = {}

    for ticker in tickers:
        if verbose:
            print(f"正在获取 {ticker} 数据...")

        df = get_etf_data(ticker, start_date, end_date, verbose=verbose)

        if not df.empty:
            data[ticker] = df
        else:
            if verbose:
                print(f"  警告: 无法获取 {ticker} 的数据")

    return data


def get_adj_close_prices(
    ticker_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    从多个ETF数据中提取调整后收盘价

    调整后收盘价（Adj Close）考虑了分红、拆股等企业行为，
    是计算收益率的正确价格数据。

    Args:
        ticker_data: 字典，键为ETF代码，值为DataFrame
                     DataFrame应包含 'Adj Close' 或 'Close' 列

    Returns:
        DataFrame，包含所有ETF的调整后收盘价
        行索引为日期，列为各ETF代码
        例如：
                    QQQ    SPY    GLD
        2020-01-31  100.5  320.2  150.3
        2020-02-29  98.7   315.8  155.2

    Note:
        - 优先使用 'Adj Close' 列，若无则使用 'Close' 列
        - 数据为空的ETF会被跳过
    """
    prices = {}

    # 如果输入为空，返回空DataFrame
    if not ticker_data:
        return pd.DataFrame()

    for ticker, df in ticker_data.items():
        if df.empty:
            print(f"警告: {ticker} 数据为空")
            continue

        # 处理 MultiIndex 列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 优先使用调整后收盘价
        if 'Adj Close' in df.columns:
            series = df['Adj Close']
        elif 'Close' in df.columns:
            series = df['Close']
        else:
            print(f"警告: {ticker} 未找到收盘价列")
            continue

        # 只添加非空序列
        if not series.empty:
            prices[ticker] = series

    if not prices:
        return pd.DataFrame()

    result = pd.DataFrame(prices)
    # 确保索引是DatetimeIndex
    if not isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index = pd.to_datetime(result.index)
        except Exception as e:
            print(f"警告: 无法将索引转换为日期类型: {e}")

    return result


def resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    将日频价格数据重采样为月频

    使用月末最后一个可用的价格。
    这是回测中最常用的重采样方式，因为：
    1. 月度频率足够捕捉主要趋势
    2. 减少数据量和计算复杂度
    3. 与月度定投策略匹配

    Args:
        prices: 日频价格DataFrame
                行索引为日期（DatetimeIndex）

    Returns:
        月频价格DataFrame
        每月只保留一个数据点（月末最后一天）

    Example:
        输入：日频数据 2020-01-02, 2020-01-03, ..., 2020-01-31, 2020-02-03, ...
        输出：月频数据 2020-01-31, 2020-02-28, ...
    """
    return prices.resample('ME').last()


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    从价格数据计算月度收益率

    收益率 = (本期价格 - 上期价格) / 上期价格

    Args:
        prices: 价格DataFrame（通常是月频）

    Returns:
        收益率DataFrame（百分比变化）
        第一行为NaN（无上期价格），已自动删除

    Note:
        返回的是简单收益率，不是对数收益率
    """
    return prices.pct_change().dropna()


if __name__ == "__main__":
    # 测试数据获取
    print("=" * 60)
    print("数据获取模块测试")
    print("=" * 60)

    tickers = list(config.PORTFOLIO_ALLOCATION.keys())
    print(f"\n测试ETF列表: {tickers}")
    print(f"回测开始日期: 1998-01-01\n")

    data = get_all_etf_data(tickers, '1998-01-01')
    prices = get_adj_close_prices(data)

    print("\n" + "=" * 60)
    print("数据获取结果")
    print("=" * 60)
    print(f"\n前5行数据:")
    print(prices.head())
    print(f"\n后5行数据:")
    print(prices.tail())
    print(f"\n日期范围: {prices.index.min().strftime('%Y-%m-%d')} 到 {prices.index.max().strftime('%Y-%m-%d')}")
    print(f"ETF数量: {len(prices.columns)}")