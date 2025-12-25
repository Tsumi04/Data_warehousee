"""
Retail Analytics Dashboard using Streamlit
A comprehensive business intelligence dashboard for retail data analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
from sqlalchemy import text
import numpy as np
from datetime import datetime, timedelta

from config import DATABASE_URL, DASHBOARD_TITLE, CHART_COLORS

# Page Configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


# ==================== DATABASE CONNECTION ====================

@st.cache_resource
def get_engine():
    """Get database engine with caching."""
    return create_engine(DATABASE_URL)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(query, params=None):
    """
    Load data from data warehouse.
    
    Args:
        query: SQL query string
        params: Optional query parameters
    
    Returns:
        pandas.DataFrame with query results
    """
    engine = get_engine()
    try:
        df = pd.read_sql(text(query), engine, params=params)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ==================== DASHBOARD TITLE ====================

st.title("📊 Retail Analytics Dashboard")
st.markdown("---")

# ==================== SIDEBAR FILTERS ====================

st.sidebar.header("🎛️ Filters")

# Date range filter
st.sidebar.subheader("Date Range")

try:
    min_date = load_data("SELECT MIN(FullDate) as MinDate FROM DimDate")['MinDate'].values[0]
    max_date = load_data("SELECT MAX(FullDate) as MaxDate FROM DimDate")['MaxDate'].values[0]
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range"
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date
except:
    start_date, end_date = None, None
    st.sidebar.warning("Unable to load date range")

# Top N selection
st.sidebar.subheader("Display Settings")
top_n = st.sidebar.slider("Top N Items", min_value=5, max_value=50, value=10)

# Chart type selection
st.sidebar.subheader("Visualization Settings")
chart_theme = st.sidebar.selectbox(
    "Chart Theme",
    ["plotly", "plotly_white", "plotly_dark", "ggplot2", "presentation"]
)


# ==================== KPI METRICS ====================

st.header("📈 Key Performance Indicators (KPIs)")

try:
    kpi_query = """
    SELECT 
        SUM(TotalRevenue) as TotalRevenue,
        COUNT(DISTINCT CustomerKey) as UniqueCustomers,
        SUM(Quantity) as TotalQuantity,
        COUNT(DISTINCT ProductKey) as UniqueProducts,
        COUNT(DISTINCT InvoiceNo) as TotalOrders,
        ROUND(AVG(TotalRevenue), 2) as AvgTransactionValue
    FROM FactSales
    """
    
    kpi_data = load_data(kpi_query)
    
    if not kpi_data.empty:
        total_revenue = kpi_data['TotalRevenue'].values[0] if 'TotalRevenue' in kpi_data.columns else 0
        unique_customers = kpi_data['UniqueCustomers'].values[0] if 'UniqueCustomers' in kpi_data.columns else 0
        total_quantity = kpi_data['TotalQuantity'].values[0] if 'TotalQuantity' in kpi_data.columns else 0
        unique_products = kpi_data['UniqueProducts'].values[0] if 'UniqueProducts' in kpi_data.columns else 0
        total_orders = kpi_data['TotalOrders'].values[0] if 'TotalOrders' in kpi_data.columns else 0
        avg_transaction = kpi_data['AvgTransactionValue'].values[0] if 'AvgTransactionValue' in kpi_data.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Total Revenue",
            f"${total_revenue:,.2f}" if total_revenue else "$0.00"
        )
        col2.metric(
            "Unique Customers",
            f"{int(unique_customers):,}" if unique_customers else "0"
        )
        col3.metric(
            "Total Products Sold",
            f"{int(total_quantity):,}" if total_quantity else "0"
        )
        col4.metric(
            "Total Orders",
            f"{int(total_orders):,}" if total_orders else "0"
        )
        
        col5, col6 = st.columns(2)
        col5.metric("Unique Products", f"{int(unique_products):,}" if unique_products else "0")
        col6.metric("Avg Transaction Value", f"${avg_transaction:.2f}" if avg_transaction else "$0.00")
    else:
        st.warning("No data available for KPIs")
        
except Exception as e:
    st.error(f"Error loading KPI data: {e}")


st.markdown("---")

# ==================== REVENUE TREND ANALYSIS ====================

st.header("📊 Revenue Trend Analysis")

try:
    revenue_query = """
    SELECT 
        d.Year,
        d.Month,
        d.MonthName,
        SUM(fs.TotalRevenue) as MonthlyRevenue,
        COUNT(DISTINCT fs.InvoiceNo) as NumOrders,
        SUM(fs.Quantity) as TotalQuantity,
        COUNT(DISTINCT fs.CustomerKey) as NumCustomers
    FROM FactSales fs
    JOIN DimDate d ON fs.DateKey = d.DateKey
    GROUP BY d.Year, d.Month, d.MonthName
    ORDER BY d.Year, d.Month
    """
    
    revenue_data = load_data(revenue_query)
    
    if not revenue_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly Revenue Trend
            fig_revenue = px.line(
                revenue_data,
                x='MonthName',
                y='MonthlyRevenue',
                color='Year',
                title='Monthly Revenue Trend',
                labels={'MonthlyRevenue': 'Revenue ($)', 'MonthName': 'Month'},
                template=chart_theme,
                markers=True
            )
            fig_revenue.update_layout(height=400)
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Monthly Orders
            fig_orders = px.bar(
                revenue_data,
                x='MonthName',
                y='NumOrders',
                color='Year',
                title='Number of Orders by Month',
                labels={'NumOrders': 'Number of Orders', 'MonthName': 'Month'},
                template=chart_theme
            )
            fig_orders.update_layout(height=400)
            st.plotly_chart(fig_orders, use_container_width=True)
    else:
        st.warning("No revenue trend data available")
        
except Exception as e:
    st.error(f"Error loading revenue data: {e}")


# ==================== GEOGRAPHIC ANALYSIS ====================

st.header("🌍 Geographic Analysis")

try:
    geography_query = """
    SELECT 
        l.Country,
        SUM(fs.TotalRevenue) as TotalRevenue,
        COUNT(DISTINCT fs.CustomerKey) as NumCustomers,
        COUNT(DISTINCT fs.ProductKey) as NumProducts,
        SUM(fs.Quantity) as TotalQuantity
    FROM FactSales fs
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    GROUP BY l.Country
    ORDER BY TotalRevenue DESC
    """
    
    geo_data = load_data(geography_query)
    
    if not geo_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Country (Map)")
            try:
                fig_map = px.choropleth(
                    geo_data,
                    locations="Country",
                    locationmode='country names',
                    color="TotalRevenue",
                    hover_name="Country",
                    hover_data=["NumCustomers", "TotalQuantity"],
                    color_continuous_scale="Plasma",
                    title="Global Revenue Distribution"
                )
                fig_map.update_layout(height=500)
                st.plotly_chart(fig_map, use_container_width=True)
            except:
                st.info("Choropleth map requires valid country names. Using bar chart instead.")
                top_countries = geo_data.head(top_n).sort_values('TotalRevenue')
                fig_bar = px.bar(
                    top_countries,
                    x='TotalRevenue',
                    y='Country',
                    orientation='h',
                    title=f'Top {top_n} Countries by Revenue',
                    labels={'TotalRevenue': 'Revenue ($)', 'Country': 'Country'},
                    template=chart_theme
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.subheader(f"Top {top_n} Countries by Revenue")
            top_geo = geo_data.head(top_n)
            
            fig_country = px.bar(
                top_geo.sort_values('TotalRevenue', ascending=True),
                x='TotalRevenue',
                y='Country',
                orientation='h',
                color='TotalRevenue',
                color_continuous_scale='Viridis',
                title=f'Revenue by Country',
                labels={'TotalRevenue': 'Revenue ($)', 'Country': 'Country'},
                template=chart_theme
            )
            fig_country.update_layout(height=500)
            st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.warning("No geographic data available")
        
except Exception as e:
    st.error(f"Error loading geographic data: {e}")


# ==================== PRODUCT ANALYSIS ====================

st.header("📦 Product Performance Analysis")

try:
    # Top Products by Revenue
    top_products_query = """
    SELECT 
        p.Description,
        SUM(fs.TotalRevenue) as TotalRevenue,
        SUM(fs.Quantity) as TotalQuantity,
        COUNT(DISTINCT fs.InvoiceNo) as NumOrders,
        AVG(fs.UnitPrice) as AvgPrice
    FROM FactSales fs
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    WHERE p.Description IS NOT NULL AND p.Description != ''
    GROUP BY p.Description
    ORDER BY TotalRevenue DESC
    LIMIT 20
    """
    
    product_data = load_data(top_products_query)
    
    if not product_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Products by Revenue
            st.subheader(f"Top {top_n} Products by Revenue")
            top_products = product_data.head(top_n)
            
            fig_product_rev = px.bar(
                top_products.sort_values('TotalRevenue', ascending=True),
                x='TotalRevenue',
                y='Description',
                orientation='h',
                title=f'Top {top_n} Products by Revenue',
                labels={'TotalRevenue': 'Revenue ($)', 'Description': 'Product'},
                template=chart_theme,
                text='TotalRevenue'
            )
            fig_product_rev.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig_product_rev, use_container_width=True)
        
        with col2:
            # Top Products by Quantity
            st.subheader(f"Top {top_n} Products by Quantity")
            top_products_qty = product_data.nlargest(top_n, 'TotalQuantity')
            
            fig_product_qty = px.bar(
                top_products_qty.sort_values('TotalQuantity', ascending=True),
                x='TotalQuantity',
                y='Description',
                orientation='h',
                title=f'Top {top_n} Products by Quantity Sold',
                labels={'TotalQuantity': 'Quantity Sold', 'Description': 'Product'},
                template=chart_theme,
                color='AvgPrice',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_product_qty, use_container_width=True)
    else:
        st.warning("No product data available")
        
except Exception as e:
    st.error(f"Error loading product data: {e}")


# ==================== CUSTOMER ANALYSIS ====================

st.header("👥 Customer Analysis")

try:
    customer_query = """
    SELECT 
        c.Country,
        COUNT(DISTINCT c.CustomerKey) as NumCustomers,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.TotalRevenue) as AvgRevenuePerCustomer
    FROM FactSales fs
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    WHERE c.CustomerID != 'UNKNOWN'
    GROUP BY c.Country
    ORDER BY NumCustomers DESC
    """
    
    customer_data = load_data(customer_query)
    
    if not customer_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Distribution by Country")
            top_customers = customer_data.head(top_n)
            
            fig_customer = px.pie(
                top_customers,
                values='NumCustomers',
                names='Country',
                title=f'Customer Distribution (Top {min(top_n, len(top_customers))} Countries)',
                template=chart_theme
            )
            st.plotly_chart(fig_customer, use_container_width=True)
        
        with col2:
            st.subheader("Revenue per Customer by Country")
            fig_rev_customer = px.bar(
                top_customers.sort_values('AvgRevenuePerCustomer', ascending=True),
                x='AvgRevenuePerCustomer',
                y='Country',
                orientation='h',
                title='Average Revenue per Customer',
                labels={'AvgRevenuePerCustomer': 'Avg Revenue per Customer ($)', 'Country': 'Country'},
                template=chart_theme
            )
            st.plotly_chart(fig_rev_customer, use_container_width=True)
    else:
        st.warning("No customer data available")
        
except Exception as e:
    st.error(f"Error loading customer data: {e}")


# ==================== TIME-BASED ANALYSIS ====================

st.header("⏰ Time-Based Analysis")

try:
    time_query = """
    SELECT 
        d.DayName,
        d.IsWeekend,
        SUM(fs.TotalRevenue) as TotalRevenue,
        COUNT(DISTINCT fs.InvoiceNo) as NumOrders
    FROM FactSales fs
    JOIN DimDate d ON fs.DateKey = d.DateKey
    GROUP BY d.DayName, d.IsWeekend
    ORDER BY 
        CASE d.DayName
            WHEN 'Mon' THEN 1
            WHEN 'Tue' THEN 2
            WHEN 'Wed' THEN 3
            WHEN 'Thu' THEN 4
            WHEN 'Fri' THEN 5
            WHEN 'Sat' THEN 6
            WHEN 'Sun' THEN 7
        END
    """
    
    time_data = load_data(time_query)
    
    if not time_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue by Day of Week")
            fig_dow = px.bar(
                time_data,
                x='DayName',
                y='TotalRevenue',
                color='IsWeekend',
                title='Revenue by Day of Week',
                labels={'TotalRevenue': 'Revenue ($)', 'DayName': 'Day of Week'},
                template=chart_theme
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        
        with col2:
            st.subheader("Orders by Day of Week")
            fig_dow_orders = px.bar(
                time_data,
                x='DayName',
                y='NumOrders',
                color='IsWeekend',
                title='Number of Orders by Day of Week',
                labels={'NumOrders': 'Number of Orders', 'DayName': 'Day of Week'},
                template=chart_theme
            )
            st.plotly_chart(fig_dow_orders, use_container_width=True)
    else:
        st.warning("No time-based data available")
        
except Exception as e:
    st.error(f"Error loading time-based data: {e}")


# ==================== DATA QUALITY & SUMMARY ====================

st.markdown("---")
st.header("📋 Data Warehouse Summary")

try:
    summary_query = """
    SELECT 
        'FactSales' as TableName,
        COUNT(*) as RowCount,
        MIN(LoadTimestamp) as FirstLoad,
        MAX(LoadTimestamp) as LastLoad
    FROM FactSales
    
    UNION ALL
    
    SELECT 
        'DimDate' as TableName,
        COUNT(*) as RowCount,
        NULL as FirstLoad,
        NULL as LastLoad
    FROM DimDate
    
    UNION ALL
    
    SELECT 
        'DimCustomer' as TableName,
        COUNT(*) as RowCount,
        NULL as FirstLoad,
        NULL as LastLoad
    FROM DimCustomer
    
    UNION ALL
    
    SELECT 
        'DimProduct' as TableName,
        COUNT(*) as RowCount,
        NULL as FirstLoad,
        NULL as LastLoad
    FROM DimProduct
    
    UNION ALL
    
    SELECT 
        'DimLocation' as TableName,
        COUNT(*) as RowCount,
        NULL as FirstLoad,
        NULL as LastLoad
    FROM DimLocation
    """
    
    summary_data = load_data(summary_query)
    
    if not summary_data.empty:
        st.dataframe(summary_data, use_container_width=True)
    else:
        st.info("No summary data available")
        
except Exception as e:
    st.error(f"Error loading summary data: {e}")


# ==================== FOOTER ====================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem;'>
        <p>Data Warehouse & Business Intelligence Dashboard | Built with ❤️ using Streamlit & Plotly</p>
        <p>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)



