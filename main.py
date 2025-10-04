import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="P&L & Forecasting Dashboard", layout="wide", page_icon="📊")

# Title
st.title("📊 Automated P&L & Forecasting Dashboard")
st.markdown("**Upload your sales, costs, and inventory data to generate P&L statements and forecasts**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("1. Upload Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    st.markdown("---")
    
    # Forecast settings
    st.subheader("2. Forecast Settings")
    forecast_periods = st.slider("Forecast periods (months)", 1, 12, 6)
    confidence_level = st.slider("Confidence level (%)", 80, 99, 95)
    
    st.markdown("---")
    
    # Scenario settings
    st.subheader("3. Scenario Planning")
    growth_base = st.number_input("Base case growth (%)", -50, 100, 0)
    growth_best = st.number_input("Best case growth (%)", -50, 200, 20)
    growth_worst = st.number_input("Worst case growth (%)", -100, 100, -20)

# Helper functions
def validate_data(df):
    """Validate uploaded data"""
    required_cols = ['date', 'revenue', 'cogs']
    missing = [col for col in required_cols if col not in df.columns.str.lower()]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "Data validated successfully"

def prepare_data(df):
    """Prepare and clean data"""
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Convert numeric columns
    numeric_cols = ['revenue', 'cogs', 'operating_expenses', 'inventory']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values for optional columns
    if 'operating_expenses' not in df.columns:
        df['operating_expenses'] = 0
    if 'inventory' not in df.columns:
        df['inventory'] = 0
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['date', 'revenue', 'cogs'])
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def calculate_pl(df):
    """Calculate P&L metrics"""
    pl = df.copy()
    
    # Calculate key metrics
    pl['gross_profit'] = pl['revenue'] - pl['cogs']
    pl['gross_margin_pct'] = (pl['gross_profit'] / pl['revenue'] * 100).round(2)
    pl['operating_profit'] = pl['gross_profit'] - pl['operating_expenses']
    pl['operating_margin_pct'] = (pl['operating_profit'] / pl['revenue'] * 100).round(2)
    
    return pl

def forecast_sales(df, periods=6, method='exponential'):
    """Forecast sales using time series models"""
    # Prepare time series data
    ts_data = df.set_index('date')['revenue'].resample('M').sum()
    
    if len(ts_data) < 12:
        # Use simple exponential smoothing for short series
        model = ExponentialSmoothing(ts_data, seasonal=None, trend='add')
        fitted = model.fit()
        forecast = fitted.forecast(periods)
    else:
        # Use ARIMA for longer series
        try:
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(periods)
        except:
            # Fallback to exponential smoothing
            model = ExponentialSmoothing(ts_data, seasonal='add', trend='add', seasonal_periods=12)
            fitted = model.fit()
            forecast = fitted.forecast(periods)
    
    # Create forecast dataframe
    last_date = ts_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='M')
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast.values
    })
    
    return forecast_df, fitted

def calculate_scenarios(base_forecast, growth_rates):
    """Calculate scenario forecasts"""
    scenarios = {}
    
    for scenario, growth in growth_rates.items():
        multiplier = 1 + (growth / 100)
        scenarios[scenario] = base_forecast * multiplier
    
    return scenarios

def create_pl_summary(pl_df):
    """Create summary P&L table"""
    summary = {
        'Total Revenue': pl_df['revenue'].sum(),
        'Total COGS': pl_df['cogs'].sum(),
        'Gross Profit': pl_df['gross_profit'].sum(),
        'Avg Gross Margin %': pl_df['gross_margin_pct'].mean(),
        'Total Operating Expenses': pl_df['operating_expenses'].sum(),
        'Operating Profit': pl_df['operating_profit'].sum(),
        'Avg Operating Margin %': pl_df['operating_margin_pct'].mean()
    }
    
    return summary

# Main app logic
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Validate
        is_valid, msg = validate_data(df)
        
        if not is_valid:
            st.error(f"❌ {msg}")
            st.info("**Required columns:** date, revenue, cogs\n**Optional columns:** operating_expenses, inventory")
            st.stop()
        
        # Prepare data
        df = prepare_data(df)
        
        st.success(f"✅ Data loaded successfully! {len(df)} records processed.")
        
        # Calculate P&L
        pl_df = calculate_pl(df)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 P&L Statement", "🔮 Forecasts", "🎯 Scenarios", "📊 Data View"])
        
        with tab1:
            st.header("Profit & Loss Statement")
            
            # Summary metrics
            summary = create_pl_summary(pl_df)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Revenue", f"${summary['Total Revenue']:,.0f}")
            col2.metric("Gross Profit", f"${summary['Gross Profit']:,.0f}")
            col3.metric("Operating Profit", f"${summary['Operating Profit']:,.0f}")
            col4.metric("Gross Margin", f"{summary['Avg Gross Margin %']:.1f}%")
            
            st.markdown("---")
            
            # P&L Chart
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Revenue vs Costs Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pl_df['date'], y=pl_df['revenue'], 
                                        name='Revenue', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=pl_df['date'], y=pl_df['cogs'], 
                                        name='COGS', line=dict(color='red', width=2)))
                fig.add_trace(go.Scatter(x=pl_df['date'], y=pl_df['operating_expenses'], 
                                        name='Operating Expenses', line=dict(color='orange', width=2)))
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Margin Analysis")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pl_df['date'], y=pl_df['gross_margin_pct'], 
                                        name='Gross Margin %', fill='tonexty', 
                                        line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=pl_df['date'], y=pl_df['operating_margin_pct'], 
                                        name='Operating Margin %', fill='tonexty',
                                        line=dict(color='purple', width=2)))
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed P&L table
            st.subheader("Monthly P&L Breakdown")
            monthly_pl = pl_df.set_index('date').resample('M').agg({
                'revenue': 'sum',
                'cogs': 'sum',
                'gross_profit': 'sum',
                'operating_expenses': 'sum',
                'operating_profit': 'sum'
            }).round(2)
            monthly_pl.index = monthly_pl.index.strftime('%Y-%m')
            st.dataframe(monthly_pl.style.format('${:,.2f}'), use_container_width=True)
        
        with tab2:
            st.header("Sales & Cost Forecasts")
            
            # Generate forecasts
            revenue_forecast, revenue_model = forecast_sales(pl_df, forecast_periods, 'arima')
            
            # Calculate COGS forecast (maintain average margin)
            avg_cogs_ratio = (pl_df['cogs'] / pl_df['revenue']).mean()
            revenue_forecast['cogs_forecast'] = revenue_forecast['forecast'] * avg_cogs_ratio
            revenue_forecast['gross_profit_forecast'] = revenue_forecast['forecast'] - revenue_forecast['cogs_forecast']
            
            # Display forecast metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Forecasted Revenue (Total)", f"${revenue_forecast['forecast'].sum():,.0f}")
            col2.metric("Forecasted COGS (Total)", f"${revenue_forecast['cogs_forecast'].sum():,.0f}")
            col3.metric("Forecasted Gross Profit", f"${revenue_forecast['gross_profit_forecast'].sum():,.0f}")
            
            st.markdown("---")
            
            # Forecast chart
            st.subheader("Revenue Forecast")
            
            historical_monthly = pl_df.set_index('date')['revenue'].resample('M').sum()
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_monthly.index, 
                y=historical_monthly.values,
                name='Historical Revenue',
                line=dict(color='green', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=revenue_forecast['date'], 
                y=revenue_forecast['forecast'],
                name='Forecasted Revenue',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig.update_layout(height=500, hovermode='x unified', 
                            xaxis_title='Date', yaxis_title='Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.subheader("Detailed Forecast")
            forecast_display = revenue_forecast.copy()
            forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m')
            forecast_display.columns = ['Month', 'Revenue', 'COGS', 'Gross Profit']
            st.dataframe(forecast_display.style.format({'Revenue': '${:,.2f}', 
                                                        'COGS': '${:,.2f}',
                                                        'Gross Profit': '${:,.2f}'}), 
                        use_container_width=True)
        
        with tab3:
            st.header("Scenario Planning")
            
            # Calculate scenarios
            growth_rates = {
                'Base Case': growth_base,
                'Best Case': growth_best,
                'Worst Case': growth_worst
            }
            
            scenarios = calculate_scenarios(revenue_forecast['forecast'], growth_rates)
            
            # Display scenario metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Case Total", f"${scenarios['Best Case'].sum():,.0f}", 
                       f"+{growth_best}%", delta_color="normal")
            col2.metric("Base Case Total", f"${scenarios['Base Case'].sum():,.0f}",
                       f"{growth_base:+.0f}%", delta_color="off")
            col3.metric("Worst Case Total", f"${scenarios['Worst Case'].sum():,.0f}",
                       f"{growth_worst}%", delta_color="inverse")
            
            st.markdown("---")
            
            # Scenario chart
            st.subheader("Scenario Comparison")
            
            fig = go.Figure()
            
            colors = {'Best Case': 'green', 'Base Case': 'blue', 'Worst Case': 'red'}
            
            for scenario, values in scenarios.items():
                fig.add_trace(go.Scatter(
                    x=revenue_forecast['date'],
                    y=values,
                    name=scenario,
                    line=dict(color=colors[scenario], width=3)
                ))
            
            fig.update_layout(height=500, hovermode='x unified',
                            xaxis_title='Date', yaxis_title='Revenue ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario table
            st.subheader("Scenario Details")
            scenario_df = pd.DataFrame({
                'Month': revenue_forecast['date'].dt.strftime('%Y-%m'),
                'Best Case': scenarios['Best Case'].values,
                'Base Case': scenarios['Base Case'].values,
                'Worst Case': scenarios['Worst Case'].values
            })
            st.dataframe(scenario_df.style.format({
                'Best Case': '${:,.2f}',
                'Base Case': '${:,.2f}',
                'Worst Case': '${:,.2f}'
            }), use_container_width=True)
        
        with tab4:
            st.header("Raw Data View")
            
            # Display options
            col1, col2 = st.columns(2)
            with col1:
                show_records = st.number_input("Number of records to display", 10, len(pl_df), 50)
            with col2:
                view_type = st.selectbox("View type", ["All Data", "Monthly Aggregation"])
            
            if view_type == "All Data":
                st.dataframe(pl_df.head(show_records), use_container_width=True)
                
                # Download button
                csv = pl_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download P&L Data",
                    data=csv,
                    file_name=f"pl_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                monthly = pl_df.set_index('date').resample('M').agg({
                    'revenue': 'sum',
                    'cogs': 'sum',
                    'gross_profit': 'sum',
                    'gross_margin_pct': 'mean',
                    'operating_expenses': 'sum',
                    'operating_profit': 'sum',
                    'operating_margin_pct': 'mean'
                }).round(2)
                monthly.index = monthly.index.strftime('%Y-%m')
                st.dataframe(monthly, use_container_width=True)
                
                # Download button
                csv = monthly.to_csv()
                st.download_button(
                    label="📥 Download Monthly P&L",
                    data=csv,
                    file_name=f"monthly_pl_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV has the correct format and try again.")

else:
    # Show instructions
    st.info("👈 Upload a CSV file to get started!")
    
    st.markdown("---")
    
    st.subheader("📋 Data Format Requirements")
    
    st.markdown("""
    Your CSV file should contain the following columns:
    
    **Required:**
    - `date` - Date of the transaction (YYYY-MM-DD format)
    - `revenue` - Sales revenue for that period
    - `cogs` - Cost of Goods Sold
    
    **Optional:**
    - `operating_expenses` - Operating expenses (will default to 0)
    - `inventory` - Inventory levels (for future features)
    
    **Example CSV structure:**
    """)
    
    # Create example dataframe
    example_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=6, freq='M'),
        'revenue': [50000, 55000, 52000, 58000, 61000, 59000],
        'cogs': [30000, 33000, 31000, 35000, 36000, 35000],
        'operating_expenses': [8000, 8000, 8500, 8500, 9000, 9000]
    })
    
    st.dataframe(example_data, use_container_width=True)
    
    # Download example
    example_csv = example_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Example CSV",
        data=example_csv,
        file_name="example_pl_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    st.subheader("✨ Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **P&L Analysis:**
        - Automatic gross profit & margin calculation
        - Operating profit analysis
        - Monthly breakdown and trends
        - Visual charts and graphs
        """)
    
    with col2:
        st.markdown("""
        **Forecasting:**
        - Time series forecasting (ARIMA/Exponential Smoothing)
        - 1-12 month projections
        - Scenario planning (best/base/worst case)
        - Confidence intervals
        """)

# Footer
st.markdown("---")
st.markdown("**💡 Pro Tip:** Regularly update your data to improve forecast accuracy!")
