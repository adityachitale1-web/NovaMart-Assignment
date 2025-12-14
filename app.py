import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data(uploaded_file=None):
    """Load and prepare marketing campaign data"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Try to load from default location
            df = pd.read_csv('marketing_campaign_data.csv')
        
        # Convert date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.rename(columns={'date': 'Date'}, inplace=True)
        
        # Calculate derived metrics
        if 'Revenue' in df.columns and 'Cost' in df.columns:
            df['ROI'] = ((df['Revenue'] - df['Cost']) / df['Cost'] * 100).round(2)
            df['Profit'] = df['Revenue'] - df['Cost']
        
        if 'Conversions' in df.columns and 'Clicks' in df.columns:
            df['Conversion_Rate'] = (df['Conversions'] / df['Clicks'] * 100).round(2)
            df['Conversion_Rate'] = df['Conversion_Rate'].replace([np.inf, -np.inf], 0)
        
        if 'Cost' in df.columns and 'Clicks' in df.columns:
            df['CPC'] = (df['Cost'] / df['Clicks']).round(2)
            df['CPC'] = df['CPC'].replace([np.inf, -np.inf], 0)
        
        if 'Cost' in df.columns and 'Conversions' in df.columns:
            df['CPA'] = (df['Cost'] / df['Conversions']).round(2)
            df['CPA'] = df['CPA'].replace([np.inf, -np.inf], 0)
        
        if 'Impressions' in df.columns and 'Clicks' in df.columns:
            df['CTR'] = (df['Clicks'] / df['Impressions'] * 100).round(2)
            df['CTR'] = df['CTR'].replace([np.inf, -np.inf], 0)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Filter data function
def filter_data(df, date_range, campaigns, channels, regions):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if date_range:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
            (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Campaign filter
    if campaigns and 'Campaign_Name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Campaign_Name'].isin(campaigns)]
    
    # Channel filter
    if channels and 'Channel' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Channel'].isin(channels)]
    
    # Region filter
    if regions and 'Region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    return filtered_df

# KPI Cards
def display_kpi_cards(df):
    """Display key performance indicators"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
        st.metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{len(df)} campaigns"
        )
    
    with col2:
        total_cost = df['Cost'].sum() if 'Cost' in df.columns else 0
        st.metric(
            label="💸 Total Cost",
            value=f"${total_cost:,.0f}",
            delta=f"{(total_cost/total_revenue*100):.1f}% of revenue" if total_revenue > 0 else "0%"
        )
    
    with col3:
        avg_roi = df['ROI'].mean() if 'ROI' in df.columns else 0
        st.metric(
            label="📈 Average ROI",
            value=f"{avg_roi:.1f}%",
            delta="Return on Investment"
        )
    
    with col4:
        total_conversions = df['Conversions'].sum() if 'Conversions' in df.columns else 0
        st.metric(
            label="🎯 Total Conversions",
            value=f"{total_conversions:,.0f}",
            delta=f"Avg: {df['Conversions'].mean():.0f}" if 'Conversions' in df.columns else "0"
        )
    
    with col5:
        avg_conversion_rate = df['Conversion_Rate'].mean() if 'Conversion_Rate' in df.columns else 0
        st.metric(
            label="📊 Avg Conversion Rate",
            value=f"{avg_conversion_rate:.2f}%",
            delta="Click to conversion"
        )

# Overview Page
def overview_page(df):
    """Display overview dashboard"""
    st.title("📊 Marketing Campaign Dashboard - Overview")
    
    # KPI Cards
    display_kpi_cards(df)
    
    st.markdown("---")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Revenue Trend Over Time")
        if 'Date' in df.columns and 'Revenue' in df.columns:
            daily_revenue = df.groupby('Date')['Revenue'].sum().reset_index()
            fig = px.line(
                daily_revenue,
                x='Date',
                y='Revenue',
                title='Daily Revenue Trend',
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Revenue ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📺 Revenue by Channel")
        if 'Channel' in df.columns and 'Revenue' in df.columns:
            channel_revenue = df.groupby('Channel')['Revenue'].sum().reset_index()
            fig = px.pie(
                channel_revenue,
                values='Revenue',
                names='Channel',
                title='Revenue Distribution by Channel',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🎯 Top 10 Campaigns by ROI")
        if 'Campaign_Name' in df.columns and 'ROI' in df.columns:
            top_campaigns = df.nlargest(10, 'ROI')[['Campaign_Name', 'ROI', 'Revenue', 'Cost']]
            fig = px.bar(
                top_campaigns,
                x='ROI',
                y='Campaign_Name',
                orientation='h',
                title='Top Performing Campaigns',
                color='ROI',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        st.subheader("🌍 Performance by Region")
        if 'Region' in df.columns and 'Revenue' in df.columns:
            region_metrics = df.groupby('Region').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Conversions': 'sum'
            }).reset_index()
            region_metrics['ROI'] = ((region_metrics['Revenue'] - region_metrics['Cost']) / region_metrics['Cost'] * 100).round(2)
            
            fig = px.bar(
                region_metrics,
                x='Region',
                y=['Revenue', 'Cost'],
                title='Revenue vs Cost by Region',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

# Campaign Performance Page
def campaign_performance_page(df):
    """Display detailed campaign performance analysis"""
    st.title("🎯 Campaign Performance Analysis")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_campaigns = df['Campaign_Name'].nunique() if 'Campaign_Name' in df.columns else 0
        st.metric("Total Campaigns", total_campaigns)
    
    with col2:
        avg_cpa = df['CPA'].mean() if 'CPA' in df.columns else 0
        st.metric("Avg Cost Per Acquisition", f"${avg_cpa:.2f}")
    
    with col3:
        avg_cpc = df['CPC'].mean() if 'CPC' in df.columns else 0
        st.metric("Avg Cost Per Click", f"${avg_cpc:.2f}")
    
    st.markdown("---")
    
    # Campaign comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Campaign Revenue vs Cost")
        if 'Campaign_Name' in df.columns:
            campaign_metrics = df.groupby('Campaign_Name').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Conversions': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Revenue',
                x=campaign_metrics['Campaign_Name'],
                y=campaign_metrics['Revenue'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Cost',
                x=campaign_metrics['Campaign_Name'],
                y=campaign_metrics['Cost'],
                marker_color='coral'
            ))
            fig.update_layout(
                barmode='group',
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Conversion Funnel")
        if all(col in df.columns for col in ['Impressions', 'Clicks', 'Conversions']):
            total_impressions = df['Impressions'].sum()
            total_clicks = df['Clicks'].sum()
            total_conversions = df['Conversions'].sum()
            
            fig = go.Figure(go.Funnel(
                y=['Impressions', 'Clicks', 'Conversions'],
                x=[total_impressions, total_clicks, total_conversions],
                textinfo="value+percent initial",
                marker={"color": ["lightblue", "lightyellow", "lightgreen"]}
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed campaign table
    st.subheader("📋 Detailed Campaign Metrics")
    
    if 'Campaign_Name' in df.columns:
        campaign_summary = df.groupby('Campaign_Name').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean',
            'CTR': 'mean',
            'Conversion_Rate': 'mean'
        }).round(2).reset_index()
        
        campaign_summary.columns = ['Campaign', 'Revenue', 'Cost', 'Impressions', 
                                    'Clicks', 'Conversions', 'Avg ROI %', 'Avg CTR %', 'Avg Conv Rate %']
        
        st.dataframe(
            campaign_summary.style.format({
                'Revenue': '${:,.0f}',
                'Cost': '${:,.0f}',
                'Impressions': '{:,.0f}',
                'Clicks': '{:,.0f}',
                'Conversions': '{:,.0f}',
                'Avg ROI %': '{:.2f}%',
                'Avg CTR %': '{:.2f}%',
                'Avg Conv Rate %': '{:.2f}%'
            }).background_gradient(subset=['Avg ROI %'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = campaign_summary.to_csv(index=False)
        st.download_button(
            label="📥 Download Campaign Data as CSV",
            data=csv,
            file_name="campaign_performance.csv",
            mime="text/csv"
        )

# Customer Segmentation Page
def customer_segmentation_page(df):
    """Display customer segmentation analysis"""
    st.title("👥 Customer Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Regional Distribution")
        if 'Region' in df.columns:
            region_data = df.groupby('Region').agg({
                'Conversions': 'sum',
                'Revenue': 'sum'
            }).reset_index()
            
            fig = px.scatter(
                region_data,
                x='Conversions',
                y='Revenue',
                size='Revenue',
                color='Region',
                title='Revenue vs Conversions by Region',
                hover_data=['Region']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📺 Channel Performance")
        if 'Channel' in df.columns:
            channel_data = df.groupby('Channel').agg({
                'Revenue': 'sum',
                'Cost': 'sum',
                'Conversions': 'sum'
            }).reset_index()
            channel_data['ROI'] = ((channel_data['Revenue'] - channel_data['Cost']) / channel_data['Cost'] * 100).round(2)
            
            fig = px.bar(
                channel_data,
                x='Channel',
                y='ROI',
                color='ROI',
                title='ROI by Marketing Channel',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Channel vs Region Performance Heatmap")
    if 'Channel' in df.columns and 'Region' in df.columns:
        heatmap_data = df.pivot_table(
            values='Revenue',
            index='Channel',
            columns='Region',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Region", y="Channel", color="Revenue"),
            title="Revenue Heatmap: Channel vs Region",
            color_continuous_scale='Blues',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment summary table
    st.subheader("📊 Segment Performance Summary")
    if 'Region' in df.columns and 'Channel' in df.columns:
        segment_summary = df.groupby(['Region', 'Channel']).agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Conversions': 'sum',
            'ROI': 'mean'
        }).round(2).reset_index()
        
        st.dataframe(
            segment_summary.style.format({
                'Revenue': '${:,.0f}',
                'Cost': '${:,.0f}',
                'Conversions': '{:,.0f}',
                'ROI': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )

# ROI Analysis Page
def roi_analysis_page(df):
    """Display ROI analysis and insights"""
    st.title("💹 ROI Analysis & Insights")
    
    # ROI Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 ROI Distribution")
        if 'ROI' in df.columns:
            fig = px.histogram(
                df,
                x='ROI',
                nbins=30,
                title='Distribution of Campaign ROI',
                labels={'ROI': 'ROI (%)', 'count': 'Number of Campaigns'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=df['ROI'].mean(), line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {df['ROI'].mean():.1f}%")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Profit Analysis")
        if 'Profit' in df.columns and 'Campaign_Name' in df.columns:
            profit_by_campaign = df.groupby('Campaign_Name')['Profit'].sum().reset_index()
            profit_by_campaign = profit_by_campaign.nlargest(10, 'Profit')
            
            fig = px.bar(
                profit_by_campaign,
                x='Profit',
                y='Campaign_Name',
                orientation='h',
                title='Top 10 Campaigns by Profit',
                color='Profit',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # ROI by Channel and Region
    st.subheader("📈 ROI Comparison")
    
    tab1, tab2, tab3 = st.tabs(["By Channel", "By Region", "By Time"])
    
    with tab1:
        if 'Channel' in df.columns and 'ROI' in df.columns:
            channel_roi = df.groupby('Channel').agg({
                'ROI': 'mean',
                'Revenue': 'sum',
                'Cost': 'sum'
            }).reset_index()
            
            fig = px.bar(
                channel_roi,
                x='Channel',
                y='ROI',
                title='Average ROI by Channel',
                color='ROI',
                color_continuous_scale='RdYlGn',
                text='ROI'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Region' in df.columns and 'ROI' in df.columns:
            region_roi = df.groupby('Region').agg({
                'ROI': 'mean',
                'Revenue': 'sum',
                'Cost': 'sum'
            }).reset_index()
            
            fig = px.bar(
                region_roi,
                x='Region',
                y='ROI',
                title='Average ROI by Region',
                color='ROI',
                color_continuous_scale='RdYlGn',
                text='ROI'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'Date' in df.columns and 'ROI' in df.columns:
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            monthly_roi = df.groupby('Month')['ROI'].mean().reset_index()
            
            fig = px.line(
                monthly_roi,
                x='Month',
                y='ROI',
                title='ROI Trend Over Time',
                markers=True
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # ROI Insights
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'ROI' in df.columns:
            best_roi = df.loc[df['ROI'].idxmax()]
            st.success(f"**Best ROI Campaign**\n\n{best_roi.get('Campaign_Name', 'N/A')}\n\nROI: {best_roi['ROI']:.2f}%")
    
    with col2:
        if 'Channel' in df.columns and 'ROI' in df.columns:
            best_channel = df.groupby('Channel')['ROI'].mean().idxmax()
            best_channel_roi = df.groupby('Channel')['ROI'].mean().max()
            st.info(f"**Best Performing Channel**\n\n{best_channel}\n\nAvg ROI: {best_channel_roi:.2f}%")
    
    with col3:
        if 'Region' in df.columns and 'Revenue' in df.columns:
            best_region = df.groupby('Region')['Revenue'].sum().idxmax()
            best_region_revenue = df.groupby('Region')['Revenue'].sum().max()
            st.warning(f"**Top Revenue Region**\n\n{best_region}\n\nRevenue: ${best_region_revenue:,.0f}")

# Data Explorer Page
def data_explorer_page(df):
    """Display raw data with filtering options"""
    st.title("🔍 Data Explorer")
    
    st.write("Explore and download the raw campaign data")
    
    # Column selector
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to display",
        all_columns,
        default=all_columns[:10] if len(all_columns) > 10 else all_columns
    )
    
    if selected_columns:
        # Display data
        st.dataframe(df[selected_columns], use_container_width=True, height=500)
        
        # Statistics
        st.subheader("📊 Data Statistics")
        st.write(df[selected_columns].describe())
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df[selected_columns].to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="marketing_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Data info
            st.info(f"**Total Records:** {len(df)}\n\n**Columns:** {len(selected_columns)}")

# Main App
def main():
    """Main application function"""
    
    # Sidebar
    st.sidebar.title("🎯 Marketing Dashboard")
    st.sidebar.markdown("---")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Marketing Data (CSV)",
        type=['csv'],
        help="Upload your marketing campaign data in CSV format"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("⚠️ Please upload a valid CSV file to get started!")
        st.info("👆 Use the file uploader in the sidebar to upload your marketing campaign data.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.markdown("### 🔧 Filters")
    
    # Date range filter
    if 'Date' in df.columns:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
    
    # Campaign filter
    if 'Campaign_Name' in df.columns:
        campaigns = st.sidebar.multiselect(
            "Select Campaigns",
            options=df['Campaign_Name'].unique().tolist(),
            default=None
        )
    else:
        campaigns = None
    
    # Channel filter
    if 'Channel' in df.columns:
        channels = st.sidebar.multiselect(
            "Select Channels",
            options=df['Channel'].unique().tolist(),
            default=None
        )
    else:
        channels = None
    
    # Region filter
    if 'Region' in df.columns:
        regions = st.sidebar.multiselect(
            "Select Regions",
            options=df['Region'].unique().tolist(),
            default=None
        )
    else:
        regions = None
    
    # Apply filters
    filtered_df = filter_data(df, date_range, campaigns, channels, regions)
    
    # Display filter info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)} / {len(df)}")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📑 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Overview", "🎯 Campaign Performance", "👥 Customer Segmentation", 
         "💹 ROI Analysis", "🔍 Data Explorer"]
    )
    
    # Page routing
    if page == "📊 Overview":
        overview_page(filtered_df)
    elif page == "🎯 Campaign Performance":
        campaign_performance_page(filtered_df)
    elif page == "👥 Customer Segmentation":
        customer_segmentation_page(filtered_df)
    elif page == "💹 ROI Analysis":
        roi_analysis_page(filtered_df)
    elif page == "🔍 Data Explorer":
        data_explorer_page(filtered_df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This dashboard provides comprehensive analysis of marketing campaign performance, "
        "including ROI tracking, customer segmentation, and detailed metrics visualization."
    )
    st.sidebar.markdown("**Version:** 1.0.0")
    st.sidebar.markdown("**Last Updated:** 2024")

if __name__ == "__main__":
    main()
