"""
Global Portfolio Backtest Dashboard
Streamlit application for visualizing backtest results
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

# Page config
st.set_page_config(
    page_title="Global Portfolio Backtest",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def run_backtest(allocation, start_date, initial_capital, monthly_contribution, rebalance_freq):
    """Run backtest with caching"""
    backtest = bt.PortfolioBacktest(
        allocation=allocation,
        start_date=start_date,
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        rebalance_frequency=rebalance_freq
    )
    results = backtest.run_backtest(verbose=False)
    return backtest, results


def format_currency(value):
    """Format number as currency"""
    return f"${value:,.0f}"


def format_percent(value):
    """Format number as percentage"""
    return f"{value*100:.2f}%"


def format_ratio(value):
    """Format ratio with 2 decimals"""
    return f"{value:.2f}"


def create_portfolio_growth_chart(results):
    """Create portfolio growth chart"""
    fig = go.Figure()

    # Portfolio value
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    # Total contributions
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['total_contributions'],
        mode='lines',
        name='Total Contributions',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Growth Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)',
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
    """Create drawdown chart"""
    portfolio_values = results['portfolio_value']
    running_max = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='#d62728', width=1),
        hovertemplate='%{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        height=300
    )

    return fig


def create_annual_returns_chart(annual_returns):
    """Create annual returns bar chart"""
    fig = go.Figure()

    colors = ['#28a745' if r >= 0 else '#dc3545' for r in annual_returns['Annual_Return']]

    fig.add_trace(go.Bar(
        x=annual_returns['Year'],
        y=annual_returns['Annual_Return'] * 100,
        marker_color=colors,
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Annual Returns',
        xaxis_title='Year',
        yaxis_title='Return (%)',
        height=400
    )

    return fig


def create_annual_capital_changes_chart(annual_changes):
    """Create annual capital changes stacked bar chart"""
    fig = go.Figure()

    # Cumulative Contributions
    fig.add_trace(go.Bar(
        x=annual_changes['Year'],
        y=annual_changes['Cumulative_Contributions'],
        name='Cumulative Contributions',
        marker_color='#2ca02c',
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    # Cumulative Returns
    fig.add_trace(go.Bar(
        x=annual_changes['Year'],
        y=annual_changes['Cumulative_Returns'],
        name='Cumulative Returns',
        marker_color='#1f77b4',
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Annual Capital Composition',
        xaxis_title='Year',
        yaxis_title='Value ($)',
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
    """Create monthly returns heatmap"""
    # Prepare data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns[months].values * 100,
        x=months,
        y=monthly_returns.index,
        colorscale=[
            [0, '#dc3545'],
            [0.5, '#ffffff'],
            [1, '#28a745']
        ],
        zmid=0,
        hovertemplate='%{y}: %{x} = %{z:.2f}%<extra></extra>',
        colorbar=dict(title='Return (%)')
    ))

    fig.update_layout(
        title='Monthly Returns Heatmap (%)',
        xaxis_title='Month',
        yaxis_title='Year',
        height=max(400, len(monthly_returns) * 25)
    )

    return fig


def create_allocation_pie_chart(allocation):
    """Create portfolio allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=list(allocation.keys()),
        values=[v * 100 for v in allocation.values()],
        hole=0.3,
        hovertemplate='%{label}: %{value:.1f}%<extra></extra>',
        textinfo='label+percent',
        textposition='outside'
    )])

    fig.update_layout(
        title='Portfolio Allocation',
        height=400
    )

    return fig


def create_metrics_table(metrics):
    """Create metrics display table"""
    metrics_data = [
        ['CAGR', f"{metrics['cagr']*100:.2f}%"],
        ['Total Return', f"{metrics['total_return']*100:.2f}%"],
        ['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"],
        ['Volatility (Ann.)', f"{metrics['volatility']*100:.2f}%"],
        ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
        ['Sortino Ratio', f"{metrics['sortino_ratio']:.2f}"],
        ['Calmar Ratio', f"{metrics['calmar_ratio']:.2f}"],
        ['Total Contributions', f"${metrics['total_contributions']:,.0f}"],
        ['Final Value', f"${metrics['final_value']:,.0f}"],
        ['Total Gain', f"${metrics['final_value'] - metrics['total_contributions']:,.0f}"],
    ]

    df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
    return df


def main():
    st.title("📈 Global Portfolio Backtest System")
    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Portfolio allocation input
    st.sidebar.subheader("Portfolio Allocation")

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

    # Validate allocation
    total_alloc = sum(allocation.values())
    if abs(total_alloc - 1.0) > 0.01:
        st.sidebar.error(f"Allocation must sum to 100%. Current: {total_alloc*100:.0f}%")
        st.stop()

    # Other parameters
    st.sidebar.subheader("Backtest Parameters")

    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.strptime(config.BACKTEST_CONFIG['start_date'], '%Y-%m-%d').date(),
        max_value=datetime.now().date()
    )

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=0,
        value=config.BACKTEST_CONFIG['initial_capital'],
        step=1000
    )

    monthly_contribution = st.sidebar.number_input(
        "Monthly Contribution ($)",
        min_value=0,
        value=config.BACKTEST_CONFIG['monthly_contribution'],
        step=100
    )

    rebalance_freq = st.sidebar.selectbox(
        "Rebalance Frequency",
        options=['monthly', 'quarterly', 'semi-annual', 'annual'],
        index=['monthly', 'quarterly', 'semi-annual', 'annual'].index(
            config.BACKTEST_CONFIG['rebalance_frequency']
        )
    )

    # Run button
    run_backtest_btn = st.sidebar.button("🚀 Run Backtest", type="primary")

    # Main content area
    if run_backtest_btn:
        with st.spinner("Fetching data and running backtest..."):
            try:
                backtest, results = run_backtest(
                    allocation=allocation,
                    start_date=str(start_date),
                    initial_capital=initial_capital,
                    monthly_contribution=monthly_contribution,
                    rebalance_freq=rebalance_freq
                )
                st.session_state['backtest'] = backtest
                st.session_state['results'] = results
                st.success("Backtest completed!")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.stop()

    # Display results if available
    if 'backtest' in st.session_state and 'results' in st.session_state:
        backtest = st.session_state['backtest']
        results = st.session_state['results']
        metrics = backtest.get_metrics()

        # Key metrics at top
        st.header("📊 Performance Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CAGR", f"{metrics['cagr']*100:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        with col4:
            st.metric("Final Value", format_currency(metrics['final_value']))

        st.markdown("---")

        # Charts
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Portfolio Growth",
            "💰 Annual Capital",
            "📊 Annual Returns",
            "🗓️ Monthly Returns",
            "📋 Details"
        ])

        with tab1:
            # Portfolio growth chart
            st.plotly_chart(create_portfolio_growth_chart(results), use_container_width=True)

            # Drawdown chart
            st.plotly_chart(create_drawdown_chart(results), use_container_width=True)

        with tab2:
            # Annual capital changes
            annual_changes = backtest.get_annual_capital_changes()
            st.plotly_chart(create_annual_capital_changes_chart(annual_changes), use_container_width=True)

            # Annual capital changes table
            st.subheader("Annual Capital Changes Table")
            display_df = annual_changes.copy()
            display_df['Portfolio_Value'] = display_df['Portfolio_Value'].apply(format_currency)
            display_df['Cumulative_Contributions'] = display_df['Cumulative_Contributions'].apply(format_currency)
            display_df['Cumulative_Returns'] = display_df['Cumulative_Returns'].apply(format_currency)

            st.dataframe(
                display_df.rename(columns={
                    'Year': 'Year',
                    'Portfolio_Value': 'Portfolio Value',
                    'Cumulative_Contributions': 'Cumulative Contributions',
                    'Cumulative_Returns': 'Cumulative Returns'
                }),
                use_container_width=True,
                hide_index=True
            )

        with tab3:
            # Annual returns
            annual_returns = backtest.get_annual_returns()
            st.plotly_chart(create_annual_returns_chart(annual_returns), use_container_width=True)

            # Annual returns table
            annual_returns['Annual_Return_Pct'] = annual_returns['Annual_Return'].apply(
                lambda x: f"{x*100:.2f}%"
            )
            st.dataframe(
                annual_returns[['Year', 'Annual_Return_Pct']].rename(
                    columns={'Annual_Return_Pct': 'Annual Return'}
                ),
                use_container_width=True,
                hide_index=True
            )

        with tab4:
            # Monthly returns heatmap
            monthly_returns = backtest.get_monthly_returns()
            st.plotly_chart(create_monthly_returns_heatmap(monthly_returns), use_container_width=True)

            # Monthly returns table
            st.dataframe(
                monthly_returns.style.format('{:.2%}'),
                use_container_width=True
            )

        with tab5:
            # Metrics table
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Portfolio Allocation")
                st.plotly_chart(create_allocation_pie_chart(allocation), use_container_width=True)

            with col2:
                st.subheader("All Metrics")
                st.dataframe(create_metrics_table(metrics), use_container_width=True, hide_index=True)

            # Monthly data table
            st.subheader("Monthly Portfolio Values")
            display_df = results.copy()
            display_df['portfolio_value_fmt'] = display_df['portfolio_value'].apply(format_currency)
            display_df['contribution_fmt'] = display_df['contribution'].apply(format_currency)
            display_df['total_contributions_fmt'] = display_df['total_contributions'].apply(format_currency)

            st.dataframe(
                display_df[['portfolio_value_fmt', 'contribution_fmt', 'total_contributions_fmt', 'rebalanced']].rename(
                    columns={
                        'portfolio_value_fmt': 'Portfolio Value',
                        'contribution_fmt': 'Contribution',
                        'total_contributions_fmt': 'Total Contributions',
                        'rebalanced': 'Rebalanced'
                    }
                ),
                use_container_width=True
            )

    else:
        # Initial state - show instructions
        st.info("👈 Configure your portfolio in the sidebar and click 'Run Backtest' to start.")

        # Show default allocation
        st.header("Default Portfolio Allocation")
        st.plotly_chart(
            create_allocation_pie_chart(config.PORTFOLIO_ALLOCATION),
            use_container_width=True
        )

        # Show ETF descriptions
        st.header("ETF Descriptions")
        etf_info = {
            'QQQ': 'Invesco QQQ Trust - Tracks the Nasdaq-100 Index, providing exposure to 100 of the largest non-financial companies listed on Nasdaq.',
            'SPY': 'SPDR S&P 500 ETF Trust - Tracks the S&P 500 Index, providing exposure to 500 large-cap U.S. companies.',
            'ASHR': 'Xtrackers Harvest CSI 300 China A-Shares ETF - Tracks the CSI 300 Index, providing exposure to China A-shares.',
            'VIG': 'Vanguard Dividend Appreciation ETF - Tracks companies with a history of increasing dividends.',
            'EWH': 'iShares MSCI Hong Kong ETF - Tracks the MSCI Hong Kong Index, providing exposure to Hong Kong equities.',
            'GLD': 'SPDR Gold Shares - Tracks the price of gold bullion, providing exposure to gold as a commodity.'
        }

        for ticker, desc in etf_info.items():
            st.markdown(f"**{ticker}**: {desc}")


if __name__ == "__main__":
    main()