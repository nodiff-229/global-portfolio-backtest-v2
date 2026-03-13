"""
全球投资组合回测仪表盘
Streamlit 应用程序，用于可视化回测结果

功能特点：
====================
1. 交互式参数配置
2. 资产可用性日历展示
3. 多维度绩效可视化
4. XIRR年化收益率显示

使用方法：
====================
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import backtest as bt
import config

# 页面配置
st.set_page_config(
    page_title="全球投资组合回测系统",
    page_icon="📈",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def run_backtest(allocation, start_date, initial_capital, monthly_contribution, rebalance_freq, transaction_cost):
    """
    执行回测（带缓存）

    使用 Streamlit 的缓存机制，避免重复执行相同的回测。

    Args:
        allocation: 投资组合配置 {代码: 权重}
        start_date: 开始日期
        initial_capital: 初始资金
        monthly_contribution: 每月投入
        rebalance_freq: 再平衡频率
        transaction_cost: 交易成本

    Returns:
        (回测引擎实例, 结果DataFrame)
    """
    backtest = bt.PortfolioBacktest(
        allocation=allocation,
        start_date=start_date,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        rebalance_frequency=rebalance_freq,
        transaction_cost=transaction_cost
    )
    results = backtest.run_backtest(verbose=False)
    return backtest, results


def format_currency(value):
    """
    格式化为货币显示

    Args:
        value: 数值

    Returns:
        格式化字符串，如 "$1,234,567"
    """
    return f"${value:,.0f}"


def format_percent(value):
    """
    格式化为百分比显示

    Args:
        value: 小数形式的百分比

    Returns:
        格式化字符串，如 "12.34%"
    """
    return f"{value*100:.2f}%"


def format_ratio(value):
    """
    格式化比率（保留两位小数）

    Args:
        value: 数值

    Returns:
        格式化字符串，如 "1.23"
    """
    return f"{value:.2f}"


def create_portfolio_growth_chart(results):
    """
    创建投资组合增长曲线图

    展示投资组合价值与累计投入的对比。

    Args:
        results: 回测结果DataFrame

    Returns:
        Plotly Figure对象
    """
    fig = go.Figure()

    # 投资组合价值曲线
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['portfolio_value'],
        mode='lines',
        name='投资组合价值',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    # 累计投入曲线
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['total_contributions'],
        mode='lines',
        name='累计投入',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='投资组合增长曲线',
        xaxis_title='日期',
        yaxis_title='价值 ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )

    return fig


def create_drawdown_chart(results):
    """
    创建回撤图表

    展示投资组合从峰值的最大跌幅。

    Args:
        results: 回测结果DataFrame

    Returns:
        Plotly Figure对象
    """
    portfolio_values = results['portfolio_value']
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results.index,
        y=drawdown,
        mode='lines',
        name='回撤',
        fill='tozeroy',
        line=dict(color='#d62728', width=1),
        hovertemplate='%{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title='投资组合回撤',
        xaxis_title='日期',
        yaxis_title='回撤 (%)',
        hovermode='x unified',
        height=300
    )

    return fig


def create_annual_returns_chart(annual_returns):
    """
    创建年度收益柱状图

    正收益显示绿色，负收益显示红色。

    Args:
        annual_returns: 年度收益DataFrame

    Returns:
        Plotly Figure对象
    """
    fig = go.Figure()

    # 根据正负决定颜色
    colors = ['#28a745' if r >= 0 else '#dc3545' for r in annual_returns['Annual_Return']]

    fig.add_trace(go.Bar(
        x=annual_returns['Year'],
        y=annual_returns['Annual_Return'] * 100,
        marker_color=colors,
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='年度收益率',
        xaxis_title='年份',
        yaxis_title='收益率 (%)',
        height=400
    )

    return fig


def create_annual_capital_changes_chart(annual_changes):
    """
    创建年度资金构成堆叠柱状图

    展示每年年末的资金构成：累计投入 vs 累计收益。

    Args:
        annual_changes: 年度资金变化DataFrame

    Returns:
        Plotly Figure对象
    """
    fig = go.Figure()

    # 累计投入（绿色）
    fig.add_trace(go.Bar(
        x=annual_changes['Year'],
        y=annual_changes['Cumulative_Contributions'],
        name='累计投入',
        marker_color='#2ca02c',
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    # 累计收益（蓝色）
    fig.add_trace(go.Bar(
        x=annual_changes['Year'],
        y=annual_changes['Cumulative_Returns'],
        name='累计收益',
        marker_color='#1f77b4',
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='年度资金构成',
        xaxis_title='年份',
        yaxis_title='价值 ($)',
        barmode='stack',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )

    return fig


def create_monthly_returns_heatmap(monthly_returns):
    """
    创建月度收益热力图

    按年份和月份展示收益率，颜色深浅表示正负。

    Args:
        monthly_returns: 月度收益DataFrame

    Returns:
        Plotly Figure对象
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns[months].values * 100,
        x=months,
        y=monthly_returns.index,
        colorscale=[
            [0, '#dc3545'],      # 负收益：红色
            [0.5, '#ffffff'],    # 零收益：白色
            [1, '#28a745']       # 正收益：绿色
        ],
        zmid=0,
        hovertemplate='%{y}: %{x} = %{z:.2f}%<extra></extra>',
        colorbar=dict(title='收益率 (%)')
    ))

    fig.update_layout(
        title='月度收益率热力图 (%)',
        xaxis_title='月份',
        yaxis_title='年份',
        height=max(400, len(monthly_returns) * 25)
    )

    return fig


def create_allocation_pie_chart(allocation):
    """
    创建投资组合配置饼图

    Args:
        allocation: 配置字典 {代码: 权重}

    Returns:
        Plotly Figure对象
    """
    fig = go.Figure(data=[go.Pie(
        labels=list(allocation.keys()),
        values=[v * 100 for v in allocation.values()],
        hole=0.3,
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='投资组合配置',
        height=400
    )

    return fig


def create_asset_availability_table(backtest):
    """
    创建资产可用性信息表

    展示每个ETF的成立日期和代理数据使用情况。

    Args:
        backtest: 回测引擎实例

    Returns:
        pandas DataFrame
    """
    if backtest.asset_calendar is None:
        return pd.DataFrame()

    return backtest.asset_calendar.get_availability_info()


def create_metrics_table(metrics):
    """
    创建绩效指标表格

    展示所有关键绩效指标。

    Args:
        metrics: 指标字典

    Returns:
        pandas DataFrame
    """
    metrics_data = [
        ['年化收益率 (XIRR)', f"{metrics['cagr']*100:.2f}%"],
        ['总收益率', f"{metrics['total_return']*100:.2f}%"],
        ['最大回撤', f"{metrics['max_drawdown']*100:.2f}%"],
        ['年化波动率', f"{metrics['volatility']*100:.2f}%"],
        ['夏普比率', f"{metrics['sharpe_ratio']:.2f}"],
        ['索提诺比率', f"{metrics['sortino_ratio']:.2f}"],
        ['卡玛比率', f"{metrics['calmar_ratio']:.2f}"],
        ['累计投入', f"${metrics['total_contributions']:,.0f}"],
        ['期末价值', f"${metrics['final_value']:,.0f}"],
        ['总收益', f"${metrics['final_value'] - metrics['total_contributions']:,.0f}"],
    ]

    df = pd.DataFrame(metrics_data, columns=['指标', '数值'])
    return df


def main():
    """
    主函数：构建Streamlit应用界面
    """
    st.title("📈 全球投资组合回测系统")
    st.markdown("---")

    # ================================================================
    # 侧边栏：配置参数
    # ================================================================
    st.sidebar.header("⚙️ 参数配置")

    # 投资组合配置
    st.sidebar.subheader("投资组合配置")

    default_allocation = config.PORTFOLIO_ALLOCATION.copy()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        qqq = st.number_input("QQQ (%)", 0, 100, int(default_allocation['QQQ'] * 100), 5)
        ashr = st.number_input("ASHR (%)", 0, 100, int(default_allocation['ASHR'] * 100), 5)
        ewh = st.number_input("EWH (%)", 0, 100, int(default_allocation['EWH'] * 100), 5)

    with col2:
        spy = st.number_input("SPY (%)", 0, 100, int(default_allocation['SPY'] * 100), 5)
        vig = st.number_input("VIG (%)", 0, 100, int(default_allocation['VIG'] * 100), 5)
        gld = st.number_input("GLD (%)", 0, 100, int(default_allocation['GLD'] * 100), 5)

    allocation = {
        'QQQ': qqq / 100,
        'SPY': spy / 100,
        'ASHR': ashr / 100,
        'VIG': vig / 100,
        'EWH': ewh / 100,
        'GLD': gld / 100
    }

    # 验证权重总和
    total_alloc = sum(allocation.values())
    if abs(total_alloc - 1.0) > 0.01:
        st.sidebar.error(f"权重总和必须为100%。当前: {total_alloc*100:.0f}%")
        st.stop()

    # 回测参数
    st.sidebar.subheader("回测参数")

    start_date = st.sidebar.date_input(
        "开始日期",
        value=datetime.strptime(config.BACKTEST_CONFIG['start_date'], '%Y-%m-%d').date(),
        max_value=datetime.now().date()
    )

    initial_capital = st.sidebar.number_input(
        "初始资金 ($)",
        min_value=0,
        value=config.BACKTEST_CONFIG['initial_capital'],
        step=1000
    )

    monthly_contribution = st.sidebar.number_input(
        "每月投入 ($)",
        min_value=0,
        value=config.BACKTEST_CONFIG['monthly_contribution'],
        step=100
    )

    rebalance_freq = st.sidebar.selectbox(
        "再平衡频率",
        options=['monthly', 'quarterly', 'semi-annual', 'annual'],
        index=['monthly', 'quarterly', 'semi-annual', 'annual'].index(
            config.BACKTEST_CONFIG['rebalance_frequency']
        )
    )

    # 交易成本
    transaction_cost = st.sidebar.number_input(
        "交易成本 (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.05,
        format="%.2f"
    ) / 100  # 转换为小数

    # 运行按钮
    run_backtest_btn = st.sidebar.button("🚀 开始回测", type="primary")

    # ================================================================
    # 主内容区域
    # ================================================================
    if run_backtest_btn:
        with st.spinner("正在获取数据并执行回测..."):
            try:
                backtest, results = run_backtest(
                    allocation=allocation,
                    start_date=str(start_date),
                    initial_capital=initial_capital,
                    monthly_contribution=monthly_contribution,
                    rebalance_freq=rebalance_freq,
                    transaction_cost=transaction_cost
                )
                st.session_state['backtest'] = backtest
                st.session_state['results'] = results
                st.success("回测完成！")
            except Exception as e:
                st.error(f"回测执行错误: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    # ================================================================
    # 显示结果
    # ================================================================
    if 'backtest' in st.session_state and 'results' in st.session_state:
        backtest = st.session_state['backtest']
        results = st.session_state['results']
        metrics = backtest.get_metrics()

        # 关键指标概览
        st.header("📊 绩效概览")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("年化收益率 (XIRR)", f"{metrics['cagr']*100:.2f}%")
        with col2:
            st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("最大回撤", f"{metrics['max_drawdown']*100:.2f}%")
        with col4:
            st.metric("期末价值", format_currency(metrics['final_value']))

        st.markdown("---")

        # ============================================================
        # 资产可用性信息
        # ============================================================
        st.header("📋 资产可用性信息")
        st.markdown("""
        <div class="info-box">
        <b>说明：</b>由于部分ETF成立时间晚于回测开始日期，系统使用代理数据进行回溯填充。
        代理数据的价格已缩放到ETF价格水平，确保价格序列连续无跳变。
        </div>
        """, unsafe_allow_html=True)

        availability_df = create_asset_availability_table(backtest)
        if not availability_df.empty:
            st.dataframe(availability_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ============================================================
        # 图表标签页
        # ============================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 组合增长",
            "💰 年度资金",
            "📊 年度收益",
            "🗓️ 月度收益",
            "📋 详细数据"
        ])

        with tab1:
            # 投资组合增长曲线
            st.plotly_chart(create_portfolio_growth_chart(results), use_container_width=True)

            # 回撤图表
            st.plotly_chart(create_drawdown_chart(results), use_container_width=True)

        with tab2:
            # 年度资金构成
            annual_changes = backtest.get_annual_capital_changes()
            st.plotly_chart(create_annual_capital_changes_chart(annual_changes), use_container_width=True)

            # 年度资金变化表
            st.subheader("年度资金变化表")
            display_df = annual_changes.copy()
            display_df['Portfolio_Value'] = display_df['Portfolio_Value'].apply(format_currency)
            display_df['Cumulative_Contributions'] = display_df['Cumulative_Contributions'].apply(format_currency)
            display_df['Cumulative_Returns'] = display_df['Cumulative_Returns'].apply(format_currency)

            st.dataframe(
                display_df.rename(columns={
                    'Year': '年份',
                    'Portfolio_Value': '组合价值',
                    'Cumulative_Contributions': '累计投入',
                    'Cumulative_Returns': '累计收益'
                }),
                use_container_width=True,
                hide_index=True
            )

        with tab3:
            # 年度收益
            annual_returns = backtest.get_annual_returns()
            st.plotly_chart(create_annual_returns_chart(annual_returns), use_container_width=True)

            # 年度收益表
            annual_returns['Annual_Return_Pct'] = annual_returns['Annual_Return'].apply(
                lambda x: f"{x*100:.2f}%"
            )
            st.dataframe(
                annual_returns[['Year', 'Annual_Return_Pct']].rename(
                    columns={'Year': '年份', 'Annual_Return_Pct': '年度收益率'}
                ),
                use_container_width=True,
                hide_index=True
            )

        with tab4:
            # 月度收益热力图
            monthly_returns = backtest.get_monthly_returns()
            st.plotly_chart(create_monthly_returns_heatmap(monthly_returns), use_container_width=True)

            # 月度收益表
            st.dataframe(
                monthly_returns.style.format('{:.2%}'),
                use_container_width=True
            )

        with tab5:
            # 指标表
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("投资组合配置")
                st.plotly_chart(create_allocation_pie_chart(allocation), use_container_width=True)

            with col2:
                st.subheader("全部指标")
                st.dataframe(create_metrics_table(metrics), use_container_width=True, hide_index=True)

            # 月度数据表
            st.subheader("月度组合价值")
            display_df = results.copy()
            display_df['portfolio_value_fmt'] = display_df['portfolio_value'].apply(format_currency)
            display_df['contribution_fmt'] = display_df['contribution'].apply(format_currency)
            display_df['total_contributions_fmt'] = display_df['total_contributions'].apply(format_currency)

            st.dataframe(
                display_df[['portfolio_value_fmt', 'contribution_fmt', 'total_contributions_fmt', 'rebalanced']].rename(
                    columns={
                        'portfolio_value_fmt': '组合价值',
                        'contribution_fmt': '当月投入',
                        'total_contributions_fmt': '累计投入',
                        'rebalanced': '是否再平衡'
                    }
                ),
                use_container_width=True
            )

    else:
        # ================================================================
        # 初始状态：显示说明
        # ================================================================
        st.info('👈 在侧边栏配置投资组合参数，然后点击"开始回测"运行回测。')

        # 显示默认配置
        st.header("默认投资组合配置")
        st.plotly_chart(
            create_allocation_pie_chart(config.PORTFOLIO_ALLOCATION),
            use_container_width=True
        )

        # 显示ETF说明
        st.header("ETF说明")
        etf_info = {
            'QQQ': {
                'name': 'Invesco QQQ Trust',
                'desc': '跟踪纳斯达克100指数，投资于纳斯达克交易所上市的100家最大非金融公司。',
                'proxy': '^NDX (纳斯达克100指数)'
            },
            'SPY': {
                'name': 'SPDR S&P 500 ETF Trust',
                'desc': '跟踪标普500指数，投资于美国500家大型上市公司。',
                'proxy': '无'
            },
            'ASHR': {
                'name': 'Xtrackers Harvest CSI 300 China A-Shares ETF',
                'desc': '跟踪沪深300指数，投资于中国A股市场。',
                'proxy': 'FXI (中国大型股ETF)'
            },
            'VIG': {
                'name': 'Vanguard Dividend Appreciation ETF',
                'desc': '投资于有持续增加分红历史的公司。',
                'proxy': 'VTI (美国全市场ETF)'
            },
            'EWH': {
                'name': 'iShares MSCI Hong Kong ETF',
                'desc': '跟踪MSCI香港指数，投资于香港股票市场。',
                'proxy': '无'
            },
            'GLD': {
                'name': 'SPDR Gold Shares',
                'desc': '跟踪黄金现货价格，提供黄金商品敞口。',
                'proxy': 'GC=F (黄金期货)'
            }
        }

        for ticker, info in etf_info.items():
            st.markdown(f"""
            **{ticker}** - {info['name']}

            {info['desc']}

            *代理数据: {info['proxy']}*

            ---
            """)

        # 显示方法说明
        st.header("方法论说明")

        st.markdown("""
        ### 价格缩放方法

        当ETF成立时间晚于回测开始日期时，系统使用代理数据进行回溯填充。
        为了避免价格跳变，系统采用**价格缩放方法**：

        1. 获取代理数据和实际ETF数据
        2. 在重叠期计算价格比率（ETF价格 / 代理价格）
        3. 将代理数据所有OHLC价格乘以价格比率
        4. 拼接缩放后的代理数据和实际ETF数据

        ### XIRR年化收益率

        对于定投策略，使用**XIRR（扩展内部收益率）**计算年化收益：
        - 考虑每笔投入的时间价值
        - 是计算定投策略收益率的正确方法
        - 比简单CAGR更准确

        ### 资产可用日历

        系统预先构建资产可用日历，避免前瞻偏差：
        - 有代理数据的资产从回测开始就可用
        - 无代理数据的资产从ETF成立日期开始可用
        - 不可用资产的权重按比例分配给可用资产
        """)


if __name__ == "__main__":
    main()