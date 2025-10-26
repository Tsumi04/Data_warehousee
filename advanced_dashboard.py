#!/usr/bin/env python3
"""
Advanced Retail Analytics Dashboard
Comprehensive business intelligence dashboard with advanced analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

from config import DATABASE_URL, CHART_COLORS
from advanced_analytics import AdvancedAnalytics

# Page configuration
st.set_page_config(
    page_title="Advanced Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
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
def get_engine():
    """Get database engine"""
    return create_engine(DATABASE_URL)

@st.cache_resource
def get_analytics():
    """Get analytics engine"""
    return AdvancedAnalytics()

@st.cache_data(ttl=3600, max_entries=100)
def load_data(query, params=None):
    """Load data from database with enhanced caching"""
    engine = get_engine()
    try:
        # Add query optimization hints
        if 'LIMIT' not in query.upper():
            # Add LIMIT for large queries to prevent memory issues
            if 'ORDER BY' in query.upper():
                query = query + ' LIMIT 10000'
        
        df = pd.read_sql(text(query), engine, params=params)
        
        # Optimize memory usage
        if not df.empty:
            # Convert object columns to category if they have low cardinality
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=7200, max_entries=50)
def load_metadata():
    """Load metadata for filters with longer cache"""
    engine = get_engine()
    metadata = {}
    
    try:
        # Load countries
        countries_query = "SELECT DISTINCT Country FROM DimLocation ORDER BY Country"
        metadata['countries'] = pd.read_sql(text(countries_query), engine)['Country'].tolist()
        
        # Load segments
        segments_query = "SELECT DISTINCT CustomerSegment FROM DimCustomer WHERE CustomerSegment IS NOT NULL ORDER BY CustomerSegment"
        metadata['segments'] = pd.read_sql(text(segments_query), engine)['CustomerSegment'].tolist()
        
        # Load categories
        categories_query = "SELECT DISTINCT ProductCategory FROM DimProduct WHERE ProductCategory IS NOT NULL ORDER BY ProductCategory"
        metadata['categories'] = pd.read_sql(text(categories_query), engine)['ProductCategory'].tolist()
        
        return metadata
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return {'countries': [], 'segments': [], 'categories': []}

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Advanced Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2010, 12, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2011, 12, 9))
    
    # Load metadata once for all filters
    metadata = load_metadata()
    
    # Country filter
    st.sidebar.subheader("🌍 Country Filter")
    selected_countries = st.sidebar.multiselect(
        "Select Countries", 
        metadata['countries'],
        default=metadata['countries'][:5] if metadata['countries'] else []
    )
    
    # Customer segment filter
    st.sidebar.subheader("👥 Customer Segment")
    selected_segments = st.sidebar.multiselect(
        "Select Segments",
        metadata['segments'],
        default=metadata['segments']
    )
    
    # Product category filter
    st.sidebar.subheader("📦 Product Category")
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        metadata['categories'],
        default=metadata['categories']
    )
    
    # Build WHERE clause for filters
    where_conditions = []
    if selected_countries:
        country_list = "','".join(selected_countries)
        where_conditions.append(f"l.Country IN ('{country_list}')")
    if selected_segments:
        segment_list = "','".join(selected_segments)
        where_conditions.append(f"c.CustomerSegment IN ('{segment_list}')")
    if selected_categories:
        category_list = "','".join(selected_categories)
        where_conditions.append(f"p.ProductCategory IN ('{category_list}')")
    
    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "👥 Customer Analytics", "📦 Product Analytics", 
        "🌍 Geographic Analytics", "📊 Advanced Analytics", "🔍 Data Quality"
    ])
    
    with tab1:
        show_overview_tab(where_clause)
    
    with tab2:
        show_customer_analytics_tab(where_clause)
    
    with tab3:
        show_product_analytics_tab(where_clause)
    
    with tab4:
        show_geographic_analytics_tab(where_clause)
    
    with tab5:
        show_advanced_analytics_tab()
    
    with tab6:
        show_data_quality_tab()

def show_overview_tab(where_clause):
    """Show overview tab with key metrics"""
    st.header("📈 Business Overview")
    
    # Key Metrics
    metrics_query = f"""
    SELECT 
        COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
        COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers,
        COUNT(DISTINCT fs.ProductKey) as UniqueProducts,
        SUM(fs.Quantity) as TotalQuantity,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.TotalRevenue) as AvgOrderValue
    FROM FactSales fs
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause}
    """
    
    metrics_df = load_data(metrics_query)
    
    if not metrics_df.empty:
        metrics = metrics_df.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Orders", f"{metrics['TotalOrders']:,}")
            st.metric("Unique Customers", f"{metrics['UniqueCustomers']:,}")
        
        with col2:
            st.metric("Total Revenue", f"${metrics['TotalRevenue']:,.2f}")
            st.metric("Average Order Value", f"${metrics['AvgOrderValue']:,.2f}")
        
        with col3:
            st.metric("Total Quantity", f"{metrics['TotalQuantity']:,}")
            st.metric("Unique Products", f"{metrics['UniqueProducts']:,}")
    
    # Revenue Trend
    st.subheader("📈 Revenue Trend")
    trend_query = f"""
    SELECT 
        d.Year,
        d.Month,
        d.MonthName,
        SUM(fs.TotalRevenue) as MonthlyRevenue,
        COUNT(DISTINCT fs.InvoiceNo) as MonthlyOrders
    FROM FactSales fs
    JOIN DimDate d ON fs.DateKey = d.DateKey
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause}
    GROUP BY d.Year, d.Month, d.MonthName
    ORDER BY d.Year, d.Month
    """
    
    trend_df = load_data(trend_query)
    
    if not trend_df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Revenue', 'Monthly Orders'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trend_df['MonthName'] + ' ' + trend_df['Year'].astype(str),
                y=trend_df['MonthlyRevenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color=CHART_COLORS['primary'])
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trend_df['MonthName'] + ' ' + trend_df['Year'].astype(str),
                y=trend_df['MonthlyOrders'],
                mode='lines+markers',
                name='Orders',
                line=dict(color=CHART_COLORS['secondary'])
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_customer_analytics_tab(where_clause):
    """Show customer analytics tab"""
    st.header("👥 Customer Analytics")
    
    # Customer Segments
    st.subheader("🎯 Customer Segments")
    segment_query = f"""
    SELECT 
        c.CustomerSegment,
        COUNT(DISTINCT c.CustomerKey) as CustomerCount,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.TotalRevenue) as AvgRevenue
    FROM FactSales fs
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause} AND c.CustomerSegment IS NOT NULL
    GROUP BY c.CustomerSegment
    ORDER BY TotalRevenue DESC
    """
    
    segment_df = load_data(segment_query)
    
    if not segment_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                segment_df, 
                values='CustomerCount', 
                names='CustomerSegment',
                title='Customer Distribution by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                segment_df,
                x='CustomerSegment',
                y='TotalRevenue',
                title='Revenue by Customer Segment',
                color='TotalRevenue',
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Top Customers
    st.subheader("🏆 Top Customers")
    top_customers_query = f"""
    SELECT 
        c.CustomerID,
        c.Country,
        c.CustomerSegment,
        COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.TotalRevenue) as AvgOrderValue
    FROM FactSales fs
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause} AND c.CustomerID != 'UNKNOWN'
    GROUP BY c.CustomerKey, c.CustomerID, c.Country, c.CustomerSegment
    ORDER BY TotalRevenue DESC
    LIMIT 20
    """
    
    top_customers_df = load_data(top_customers_query)
    
    if not top_customers_df.empty:
        st.dataframe(
            top_customers_df,
            use_container_width=True,
            hide_index=True
        )

def show_product_analytics_tab(where_clause):
    """Show product analytics tab"""
    st.header("📦 Product Analytics")
    
    # Product Performance
    st.subheader("📊 Product Performance")
    product_query = f"""
    SELECT 
        p.StockCode,
        p.Description,
        p.ProductCategory,
        COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
        SUM(fs.Quantity) as TotalQuantity,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.UnitPrice) as AvgUnitPrice
    FROM FactSales fs
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause}
    GROUP BY p.ProductKey, p.StockCode, p.Description, p.ProductCategory
    ORDER BY TotalRevenue DESC
    LIMIT 20
    """
    
    product_df = load_data(product_query)
    
    if not product_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                product_df.head(10),
                x='TotalRevenue',
                y='Description',
                orientation='h',
                title='Top 10 Products by Revenue',
                color='TotalRevenue',
                color_continuous_scale='Greens'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                product_df,
                x='TotalQuantity',
                y='TotalRevenue',
                size='TotalOrders',
                color='ProductCategory',
                hover_data=['Description'],
                title='Product Performance Matrix'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Product Categories
    st.subheader("📂 Product Categories")
    category_query = f"""
    SELECT 
        p.ProductCategory,
        COUNT(DISTINCT p.ProductKey) as ProductCount,
        SUM(fs.TotalRevenue) as TotalRevenue,
        AVG(fs.TotalRevenue) as AvgRevenue
    FROM FactSales fs
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    WHERE {where_clause} AND p.ProductCategory IS NOT NULL
    GROUP BY p.ProductCategory
    ORDER BY TotalRevenue DESC
    """
    
    category_df = load_data(category_query)
    
    if not category_df.empty:
        fig = px.treemap(
            category_df,
            path=['ProductCategory'],
            values='TotalRevenue',
            title='Revenue by Product Category'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_geographic_analytics_tab(where_clause):
    """Show geographic analytics tab"""
    st.header("🌍 Geographic Analytics")
    
    # Country Performance
    st.subheader("🗺️ Country Performance")
    country_query = f"""
    SELECT 
        l.Country,
        l.Region,
        l.Continent,
        COUNT(DISTINCT fs.InvoiceNo) as TotalOrders,
        SUM(fs.TotalRevenue) as TotalRevenue,
        COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers
    FROM FactSales fs
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    WHERE {where_clause}
    GROUP BY l.Country, l.Region, l.Continent
    ORDER BY TotalRevenue DESC
    """
    
    country_df = load_data(country_query)
    
    if not country_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.choropleth(
                country_df,
                locations="Country",
                locationmode='country names',
                color="TotalRevenue",
                hover_name="Country",
                color_continuous_scale="Blues",
                title="Revenue by Country"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                country_df.head(15),
                x='Country',
                y='TotalRevenue',
                title='Top 15 Countries by Revenue',
                color='TotalRevenue',
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Regional Analysis
    st.subheader("🌏 Regional Analysis")
    region_query = f"""
    SELECT 
        l.Region,
        l.Continent,
        COUNT(DISTINCT l.Country) as CountryCount,
        SUM(fs.TotalRevenue) as TotalRevenue,
        COUNT(DISTINCT fs.CustomerKey) as UniqueCustomers
    FROM FactSales fs
    JOIN DimLocation l ON fs.LocationKey = l.LocationKey
    JOIN DimCustomer c ON fs.CustomerKey = c.CustomerKey
    JOIN DimProduct p ON fs.ProductKey = p.ProductKey
    WHERE {where_clause} AND l.Region IS NOT NULL
    GROUP BY l.Region, l.Continent
    ORDER BY TotalRevenue DESC
    """
    
    region_df = load_data(region_query)
    
    if not region_df.empty:
        fig = px.sunburst(
            region_df,
            path=['Continent', 'Region'],
            values='TotalRevenue',
            title='Revenue by Region and Continent'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_advanced_analytics_tab():
    """Show advanced analytics tab"""
    st.header("📊 Advanced Analytics")
    
    analytics = get_analytics()
    
    # RFM Analysis
    st.subheader("🎯 RFM Analysis")
    if st.button("Generate RFM Analysis"):
        with st.spinner("Analyzing customer behavior..."):
            rfm_df = analytics.get_rfm_analysis()
            
            if not rfm_df.empty:
                # Remove any duplicate columns to prevent plotly errors
                rfm_df = rfm_df.loc[:, ~rfm_df.columns.duplicated()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        rfm_df,
                        values='RFMScore',
                        names='RFMSegment',
                        title='Customer RFM Segments'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        rfm_df,
                        x='Frequency',
                        y='Monetary',
                        size='RecencyScore',
                        color='RFMSegment',
                        hover_data=['CustomerID', 'Country'],
                        title='RFM Scatter Plot'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # RFM Summary
                st.subheader("📋 RFM Summary")
                rfm_summary = rfm_df.groupby('RFMSegment').agg({
                    'CustomerKey': 'count',
                    'Monetary': 'mean',
                    'Frequency': 'mean'
                }).round(2)
                rfm_summary.columns = ['Customer Count', 'Avg Monetary', 'Avg Frequency']
                st.dataframe(rfm_summary, use_container_width=True)
    
    # Customer Lifetime Value
    st.subheader("💰 Customer Lifetime Value")
    if st.button("Generate CLV Analysis"):
        with st.spinner("Calculating customer lifetime value..."):
            clv_df = analytics.get_customer_lifetime_value()
            
            if not clv_df.empty:
                # Remove any duplicate columns to prevent plotly errors
                clv_df = clv_df.loc[:, ~clv_df.columns.duplicated()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        clv_df,
                        x='TotalRevenue',
                        nbins=20,
                        title='Customer Lifetime Value Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        clv_df,
                        x='CustomerTier',
                        y='TotalRevenue',
                        title='CLV by Customer Tier'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # CLV Summary
                st.subheader("📋 CLV Summary")
                clv_summary = clv_df.groupby('CustomerTier').agg({
                    'CustomerKey': 'count',
                    'TotalRevenue': ['mean', 'sum'],
                    'MonthlyOrderFrequency': 'mean'
                }).round(2)
                st.dataframe(clv_summary, use_container_width=True)
    
    # Product Performance Analysis
    st.subheader("📦 Product Performance Analysis")
    if st.button("Generate Product Analysis"):
        with st.spinner("Analyzing product performance..."):
            product_df = analytics.get_product_performance_analysis()
            
            if not product_df.empty:
                # Remove any duplicate columns to prevent plotly errors
                product_df = product_df.loc[:, ~product_df.columns.duplicated()]
                
                # Filter out products with very low values for better visualization
                # Only show products with meaningful sales data
                product_df_filtered = product_df[
                    (product_df['MonthlyQuantity'] > 1) & 
                    (product_df['MonthlyRevenue'] > 1)
                ].copy()
                
                if not product_df_filtered.empty:
                    # Use log scale for better visualization of wide-ranging data
                    fig = px.scatter(
                        product_df_filtered,
                        x='MonthlyQuantity',
                        y='MonthlyRevenue',
                        size='UniqueCustomers',
                        color='ProductCategory',
                        hover_data=['Description', 'TotalOrders', 'LifecycleStage', 'DailyRevenue', 'TotalRevenue'],
                        title='Product Performance Matrix (Log Scale)',
                        log_x=True,
                        log_y=True
                    )
                    
                    # Update layout for better readability
                    fig.update_layout(
                        xaxis_title="Monthly Quantity (Log Scale)",
                        yaxis_title="Monthly Revenue (Log Scale)",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add alternative visualization for better understanding
                    st.subheader("📊 Alternative View (Linear Scale - Top Products)")
                    top_products = product_df_filtered.nlargest(50, 'MonthlyRevenue')
                    
                    fig2 = px.scatter(
                        top_products,
                        x='MonthlyQuantity',
                        y='MonthlyRevenue',
                        size='UniqueCustomers',
                        color='ProductCategory',
                        hover_data=['Description', 'TotalOrders', 'LifecycleStage'],
                        title='Top 50 Products Performance (Linear Scale)'
                    )
                    
                    fig2.update_layout(
                        xaxis_title="Monthly Quantity",
                        yaxis_title="Monthly Revenue",
                        height=500
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Show data summary
                    st.subheader("📊 Data Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Products", f"{len(product_df):,}")
                    with col2:
                        st.metric("Filtered Products", f"{len(product_df_filtered):,}")
                    with col3:
                        st.metric("Star Products", f"{len(product_df[product_df['ProductCategory'] == 'Star']):,}")
                    with col4:
                        st.metric("Avg Monthly Revenue", f"${product_df['MonthlyRevenue'].mean():,.0f}")
                    
                    # Show category breakdown
                    st.subheader("📦 Product Category Breakdown")
                    category_summary = product_df.groupby('ProductCategory').agg({
                        'ProductKey': 'count',
                        'TotalRevenue': 'sum',
                        'MonthlyRevenue': 'mean'
                    }).round(2)
                    category_summary.columns = ['Count', 'Total Revenue', 'Avg Monthly Revenue']
                    st.dataframe(category_summary, use_container_width=True)
                    
                else:
                    st.warning("No products found with meaningful sales data for visualization.")
                    st.info("Try adjusting the filters or check if there are any products with sales data.")
                
                # Product Summary
                st.subheader("📋 Product Performance Summary")
                product_summary = product_df.groupby('ProductCategory').agg({
                    'ProductKey': 'count',
                    'TotalRevenue': ['mean', 'sum'],
                    'MonthlyQuantity': 'mean'
                }).round(2)
                st.dataframe(product_summary, use_container_width=True)

def show_data_quality_tab():
    """Show data quality tab"""
    st.header("🔍 Data Quality Dashboard")
    
    # Data Quality Summary
    st.subheader("📊 Data Quality Summary")
    quality_query = """
    SELECT 
        check_type,
        severity,
        COUNT(*) as CheckCount,
        SUM(records_failed) as TotalFailedRecords,
        AVG(failure_rate) as AvgFailureRate
    FROM DataQualityLog
    GROUP BY check_type, severity
    ORDER BY check_type, severity
    """
    
    quality_df = load_data(quality_query)
    
    if not quality_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                quality_df,
                x='check_type',
                y='CheckCount',
                color='severity',
                title='Data Quality Checks by Type and Severity'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                quality_df,
                x='check_type',
                y='AvgFailureRate',
                color='severity',
                title='Average Failure Rate by Check Type'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data Quality Details
        st.subheader("📋 Data Quality Details")
        st.dataframe(quality_df, use_container_width=True)
    
    # Recent Quality Issues
    st.subheader("⚠️ Recent Quality Issues")
    recent_issues_query = """
    SELECT 
        check_name,
        table_name,
        severity,
        records_failed,
        failure_rate,
        error_message,
        check_timestamp
    FROM DataQualityLog
    WHERE records_failed > 0
    ORDER BY check_timestamp DESC
    LIMIT 20
    """
    
    issues_df = load_data(recent_issues_query)
    
    if not issues_df.empty:
        st.dataframe(issues_df, use_container_width=True)
    else:
        st.success("No recent quality issues found!")

if __name__ == "__main__":
    main()