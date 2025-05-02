import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import numpy as np  # Add this at the top if not already imported

# Page config
st.set_page_config(
    page_title="Machu Pouches Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Machu Pouches Dashboard")

# Add this function near the top of the file, after the imports
def get_green_gradient(value, min_val, max_val):
    """Create a green gradient color based on value, min, and max."""
    # Normalize the value between 0 and 1
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    # Create a gradient from light green (#e6ffe6) to dark green (#006400)
    r = int(230 - normalized * 230)  # Start at 230, decrease to 0
    g = int(255 - normalized * 191)  # Start at 255, decrease to 64
    b = int(230 - normalized * 230)  # Start at 230, decrease to 0
    
    # Ensure text is visible (use white text for darker backgrounds)
    text_color = 'white' if normalized > 0.7 else 'black'
    
    return f'background-color: rgb({r},{g},{b}); color: {text_color}'

# Google Sheets setup
def connect_to_sheets():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("Could not find gcp_service_account in Streamlit secrets.")
            return None
            
        credentials_dict = dict(st.secrets["gcp_service_account"])
        if not credentials_dict:
            st.error("Empty credentials found in Streamlit secrets.")
            return None
            
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {str(e)}")
        st.error("Please check your Streamlit secrets configuration.")
        return None

# Function to load data
def load_data():
    try:
        client = connect_to_sheets()
        if client:
            # Replace with your actual spreadsheet key
            spreadsheet = client.open_by_key('1WLn7DH3F1Sm5ZSEHgWVEILWvvjFRsrE0b9xKrYU43Hw')
            worksheet = spreadsheet.get_worksheet(0)  # Get the first worksheet
            data = worksheet.get_all_records()
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main content
st.write("Loading data...")
df = load_data()

# Create a new column 'Payment Amount (Numeric)' for analytics
if df is not None and 'Payment Amount' in df.columns:
    df['Payment Amount (Numeric)'] = (
        df['Payment Amount']
        .astype(str)
        .str.replace('S/.', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.extract(r'([\d\.]+)')[0]
        .astype(float)
    )

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")

    # --- Chronological Time Frame Filter ---
    time_frame = st.sidebar.radio(
        "Select Time Frame",
        ("Today", "This Week", "This Month", "Last 6 Months", "Last Year", "All Time", "Custom Range"),
        index=5
    )

    # Date conversion and filtering
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        today = pd.Timestamp.today().normalize()
        
        # Add custom date range picker
        if time_frame == "Custom Range":
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
        elif time_frame == "Today":
            mask = df['Date'] == today
        elif time_frame == "This Week":
            start = today - pd.Timedelta(days=today.weekday())
            end = start + pd.Timedelta(days=6)
            mask = (df['Date'] >= start) & (df['Date'] <= end)
        elif time_frame == "This Month":
            start = today.replace(day=1)
            end = (start + pd.offsets.MonthEnd(1)).normalize()
            mask = (df['Date'] >= start) & (df['Date'] <= end)
        elif time_frame == "Last 6 Months":
            start = (today - pd.DateOffset(months=6)).replace(day=1)
            mask = (df['Date'] >= start) & (df['Date'] <= today)
        elif time_frame == "Last Year":
            start = today - pd.DateOffset(years=1)
            mask = (df['Date'] >= start) & (df['Date'] <= today)
        else:  # All Time
            mask = pd.Series([True] * len(df))
        df = df[mask]

    # Product filter
    if 'Product' in df.columns:
        products = df['Product'].unique()
        selected_products = st.sidebar.multiselect(
            "Select Products",
            options=products,
            default=products
        )
        df = df[df['Product'].isin(selected_products)]

    # --- TOP KPIs & INSIGHTS ---
    st.markdown("## 📈 Key Metrics & Insights")
    # Metrics row
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        if 'Payment Amount (Numeric)' in df.columns:
            total_sales = df['Payment Amount (Numeric)'].sum()
            st.metric("Total Sales Amount", f"S/. {total_sales:,.2f}")
    with metric_col2:
        unique_customers = df['Client ID'].nunique() if 'Client ID' in df.columns else 0
        st.metric("Unique Customers", f"{unique_customers}")
    with metric_col3:
        unique_orders = df['Order ID'].nunique() if 'Order ID' in df.columns else 0
        st.metric("Unique Orders", f"{unique_orders}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 10 tables row
    table_col1, table_col2 = st.columns(2)
    with table_col1:
        if 'Product' in df.columns and 'Payment Amount (Numeric)' in df.columns:
            top_products = df.groupby('Product')['Payment Amount (Numeric)'].sum().sort_values(ascending=False).head(10)
            percent = (top_products / total_sales * 100).round(2)
            product_df = pd.DataFrame({
                'Product': top_products.index,
                'Sales': [f"S/. {amt:,.2f}" for amt in top_products.values],
                '% of Total': [f"{p}%" for p in percent.values]
            })
            product_df.index = product_df.index + 1
            st.markdown("**Top 10 Products Sold**")
            st.table(product_df)
    with table_col2:
        if 'Buyer' in df.columns and 'Payment Amount (Numeric)' in df.columns:
            top_buyers = df.groupby('Buyer')['Payment Amount (Numeric)'].sum().sort_values(ascending=False).head(10)
            percent = (top_buyers / total_sales * 100).round(2)
            buyer_df = pd.DataFrame({
                'Buyer': top_buyers.index,
                'Sales': [f"S/. {amt:,.2f}" for amt in top_buyers.values],
                '% of Total': [f"{p}%" for p in percent.values]
            })
            buyer_df.index = buyer_df.index + 1
            st.markdown("**Top 10 Buyers**")
            st.table(buyer_df)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sales by product pie chart
    if 'Product' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        product_sales = df.groupby('Product')['Payment Amount (Numeric)'].sum().reset_index()
        # Filter out zero sales
        product_sales = product_sales[product_sales['Payment Amount (Numeric)'] > 0]
        # Calculate percent for each slice
        product_sales['percent'] = product_sales['Payment Amount (Numeric)'] / product_sales['Payment Amount (Numeric)'].sum() * 100
        # Set textposition: inside for >=3%, outside for <3%
        textpositions = ['inside' if p >= 3 else 'outside' for p in product_sales['percent']]
        fig = px.pie(product_sales, names='Product', values='Payment Amount (Numeric)',
                     title='Sales by Product', hole=0.3, width=600, height=600)
        fig.update_traces(textfont_size=22, textinfo='percent', textposition=textpositions, insidetextorientation='radial')
        fig.update_layout(
            legend=dict(
                font=dict(size=18, color='white'),
                bgcolor='rgba(0,0,0,0)',
            ),
            title_font=dict(size=22),
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Sales by Nicotine Level ---
        nicotine_levels = ['1.5mg', '3mg', '6mg', '9mg', '11mg']
        nicotine_sales = {level: 0 for level in nicotine_levels}
        for _, row in product_sales.iterrows():
            for level in nicotine_levels:
                if level in row['Product']:
                    nicotine_sales[level] += row['Payment Amount (Numeric)']
        nicotine_keys = nicotine_levels
        nicotine_vals = [nicotine_sales[level] for level in nicotine_levels]
        nicotine_perc = [f"{(v/sum(nicotine_vals)*100):.2f}%" if sum(nicotine_vals) else '0%' for v in nicotine_vals]
        fig_nic = px.bar(x=nicotine_keys, y=nicotine_vals,
                        labels={'x': 'Nicotine Level', 'y': 'Sales'},
                        title='Sales by Nicotine Level',
                        text=[f"S/. {v:,.2f} ({p})" for v, p in zip(nicotine_vals, nicotine_perc)])
        fig_nic.update_traces(textposition='outside')
        st.plotly_chart(fig_nic, use_container_width=True)

        # --- Sales by Flavour ---
        flavours = ['Cool Mint', 'Apple Mint', 'Spearmint', 'Cool Frost', 'Fresh Mint', 'Citrus', 'Exotic Mango']
        flavour_sales = {flavour: 0 for flavour in flavours}
        for _, row in product_sales.iterrows():
            for flavour in flavours:
                if flavour in row['Product']:
                    flavour_sales[flavour] += row['Payment Amount (Numeric)']
        # Sort by value descending
        flavour_items = sorted(flavour_sales.items(), key=lambda x: x[1], reverse=True)
        flavour_keys = [k for k, v in flavour_items]
        flavour_vals = [v for k, v in flavour_items]
        flavour_perc = [f"{(v/sum(flavour_vals)*100):.2f}%" if sum(flavour_vals) else '0%' for v in flavour_vals]
        fig_flav = px.bar(x=flavour_keys, y=flavour_vals,
                        labels={'x': 'Flavour', 'y': 'Sales'},
                        title='Sales by Flavour',
                        text=[f"S/. {v:,.2f} ({p})" for v, p in zip(flavour_vals, flavour_perc)])
        fig_flav.update_traces(textposition='outside')
        st.plotly_chart(fig_flav, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Sales Trend Analysis ---
    if 'Date' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        st.markdown("## 📈 Sales Trend Analysis")
        
        # Add view selection with better styling
        trend_view = st.radio(
            "Select Sales Trend View",
            ("Daily", "Weekly", "Monthly"),
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Prepare data based on selected view
        if trend_view == "Daily":
            sales_trend = df.groupby('Date')['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Daily Sales Trend'
            freq = 'D'
        elif trend_view == "Weekly":
            sales_trend = df.groupby(pd.Grouper(key='Date', freq='W-MON'))['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Weekly Sales Trend'
            freq = 'W'
        else:  # Monthly
            sales_trend = df.groupby(pd.Grouper(key='Date', freq='M'))['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Monthly Sales Trend'
            freq = 'M'
        
        # Create the plot with enhanced styling
        fig = go.Figure()
        
        # Add the main sales line
        fig.add_trace(go.Scatter(
            x=sales_trend['Date'],
            y=sales_trend['Payment Amount (Numeric)'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6, color='#1f77b4')
        ))
        
        # Add moving average based on the selected frequency
        if freq == 'D':
            window = 7  # 7-day moving average for daily view
        elif freq == 'W':
            window = 4  # 4-week moving average for weekly view
        else:
            window = 3  # 3-month moving average for monthly view
            
        sales_trend['Moving Average'] = sales_trend['Payment Amount (Numeric)'].rolling(window=window).mean()
        
        fig.add_trace(go.Scatter(
            x=sales_trend['Date'],
            y=sales_trend['Moving Average'],
            mode='lines',
            name=f'{window}-Period Moving Average',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Update layout for better readability
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color='white'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=-45
            ),
            yaxis=dict(
                title="Sales Amount (S/.)",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickprefix="S/. "
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=100)
        )
        
        # Add hover template
        if trend_view == "Daily":
            fig.update_traces(
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                             "<b>Sales</b>: S/.%{y:,.2f}<extra></extra>"
            )
        elif trend_view == "Weekly":
            fig.update_traces(
                hovertemplate="<b>Week of</b>: %{x|%Y-%m-%d}<br>" +
                             "<b>Sales</b>: S/.%{y:,.2f}<extra></extra>"
            )
        else:
            fig.update_traces(
                hovertemplate="<b>Month</b>: %{x|%Y-%m}<br>" +
                             "<b>Sales</b>: S/.%{y:,.2f}<extra></extra>"
            )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Customer Analysis ---
    if 'Client ID' in df.columns and 'Date' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        st.markdown("## 👥 Customer Analysis")
        
        df_sorted = df.sort_values('Date')
        df_sorted['Is New Customer'] = ~df_sorted['Client ID'].duplicated()
        df_sorted['Customer Type'] = df_sorted['Is New Customer'].map({True: 'New Customers', False: 'Existing Customers'})
        
        # Group by month
        df_sorted['Month'] = df_sorted['Date'].dt.strftime('%Y-%m')
        monthly_sales = df_sorted.groupby(['Month', 'Customer Type'])['Payment Amount (Numeric)'].sum().unstack(fill_value=0)
        monthly_sales['Total'] = monthly_sales.sum(axis=1)
        monthly_sales['New %'] = (monthly_sales['New Customers'] / monthly_sales['Total'] * 100).round(2)
        monthly_sales['Existing %'] = (monthly_sales['Existing Customers'] / monthly_sales['Total'] * 100).round(2)

        # Prepare data for stacked bar chart
        chart_df = pd.DataFrame({
            'Month': monthly_sales.index,
            'New Customers': monthly_sales['New %'],
            'Existing Customers': monthly_sales['Existing %']
        })

        # Create the stacked bar chart with enhanced styling
        fig = go.Figure()
        
        # Add bars for each customer type
        fig.add_trace(go.Bar(
            x=chart_df['Month'],
            y=chart_df['New Customers'],
            name='New Customers',
            marker_color='#8ecae6'  # light blue
        ))
        
        fig.add_trace(go.Bar(
            x=chart_df['Month'],
            y=chart_df['Existing Customers'],
            name='Existing Customers',
            marker_color='#003366'  # darker blue
        ))

        # Update layout for better readability
        fig.update_layout(
            title=dict(
                text='Customer Distribution by Month',
                font=dict(size=24, color='white'),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Month",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=-45
            ),
            yaxis=dict(
                title="Percentage of Total Sales (%)",
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                range=[0, 100]
            ),
            barmode='stack',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=100)
        )

        # Add hover template
        fig.update_traces(
            hovertemplate="<b>Month</b>: %{x}<br>" +
                         "<b>Percentage</b>: %{y:.2f}%<br>" +
                         "<b>Customer Type</b>: %{fullData.name}<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add summary statistics
        col1, col2 = st.columns(2)
        with col1:
            avg_new_customers = chart_df['New Customers'].mean()
            st.metric("Average New Customer Percentage", f"{avg_new_customers:.1f}%")
        with col2:
            avg_existing_customers = chart_df['Existing Customers'].mean()
            st.metric("Average Existing Customer Percentage", f"{avg_existing_customers:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Weekly Sales Summary Table ---
    if 'Date' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        # Add view selection
        summary_view = st.radio(
            "Select Summary View",
            ("Weekly", "Monthly"),
            horizontal=True,
            key="summary_view"
        )

        summary_df = df.copy()
        # Ensure dates are datetime and normalized to midnight
        summary_df['Date'] = pd.to_datetime(summary_df['Date']).dt.normalize()

        if summary_view == "Weekly":
            # Weekly Summary Logic
            first_date = summary_df['Date'].min()
            # Ensure we start from Monday
            first_monday = first_date - pd.Timedelta(days=first_date.weekday())
            last_date = summary_df['Date'].max()
            
            # Create weekly bins from first Monday to last date
            bins = pd.date_range(start=first_monday, end=last_date + pd.Timedelta(days=6), freq='W-MON')
            
            # Create summary DataFrame with proper date handling
            summary_data = []
            week_num = 1
            
            for start_date in bins[:-1]:  # Exclude the last bin edge
                end_date = start_date + pd.Timedelta(days=6)  # End date is Sunday
                
                # Filter data for the week using inclusive range to match Excel's SUMIFS
                week_data = summary_df[
                    (summary_df['Date'] >= start_date) & 
                    (summary_df['Date'] <= end_date)
                ]
                
                # Calculate sales quantity by summing the Amount column
                sales_quantity = week_data['Amount'].sum()
                sales_amount = week_data['Payment Amount (Numeric)'].sum()
                
                # Only include weeks that have data or are within our date range
                if start_date <= last_date:
                    summary_data.append({
                        'Week': week_num,
                        'Initial_Date': start_date.strftime('%d/%m/%Y'),
                        'Ending_Date': end_date.strftime('%d/%m/%Y'),
                        'Sales_Quantity': sales_quantity,
                        'Sales_Amount': sales_amount,
                        'Days_in_Period': 7
                    })
                    week_num += 1
            
            summary = pd.DataFrame(summary_data)
            
            if not summary.empty:
                # Calculate daily averages
                summary['Sales_Quantity_per_Day'] = (summary['Sales_Quantity'] / summary['Days_in_Period']).round(2)
                summary['Sales_Amount_per_Day'] = (summary['Sales_Amount'] / summary['Days_in_Period']).round(2)
                
                # Format display columns
                summary['Sales Amount'] = summary['Sales_Amount'].apply(lambda x: f"S/.{x:,.0f}")
                summary['Sales Amount per Day'] = summary['Sales_Amount_per_Day'].apply(lambda x: f"S/.{x:.2f}")
                summary['Sales Quant per Day'] = summary['Sales_Quantity_per_Day'].apply(lambda x: f"{x:.2f}")
                
                # Select and order columns for display
                display_columns = [
                    'Week', 'Initial_Date', 'Ending_Date', 'Sales_Quantity',
                    'Sales Quant per Day', 'Sales Amount', 'Sales Amount per Day'
                ]
                
                display_summary = summary[display_columns].copy()
                display_summary.columns = [
                    'Week', 'Initial Date', 'Ending Date', 'Sales Quantity',
                    'Sales Quant per Day', 'Sales Amount', 'Sales Amount per Day'
                ]
                
                # Apply green gradient to Sales Quant per Day
                min_quant = summary['Sales_Quantity_per_Day'].min()
                max_quant = summary['Sales_Quantity_per_Day'].max()
                
                def style_weekly_quant(val):
                    try:
                        num_val = float(val)
                        return get_green_gradient(num_val, min_quant, max_quant)
                    except:
                        return ''
                
                styled_weekly = display_summary.style.applymap(
                    style_weekly_quant,
                    subset=['Sales Quant per Day']
                )
                
                # Display the table
                st.markdown(f'**{summary_view} Sales Summary**')
                st.dataframe(styled_weekly, use_container_width=True)
            else:
                st.warning("No data available for the selected time period.")
        elif summary_view == "Monthly":
            # Monthly Summary Logic
            monthly_data = []
            
            # Group by month
            monthly_groups = summary_df.groupby(summary_df['Date'].dt.to_period('M'))
            
            for month, month_data in monthly_groups:
                # Get first and last day of the calendar month
                start_date = month.to_timestamp()  # This gives first day of month
                end_date = (month.to_timestamp() + pd.offsets.MonthEnd(0))  # This gives last day of month
                days_in_period = (end_date - start_date).days + 1  # +1 because both start and end dates are inclusive
                
                sales_quantity = month_data['Amount'].sum()
                sales_amount = month_data['Payment Amount (Numeric)'].sum()
                
                monthly_data.append({
                    'Month': month.strftime('%B %Y'),
                    'Initial_Date': start_date.strftime('%d/%m/%Y'),
                    'Ending_Date': end_date.strftime('%d/%m/%Y'),
                    'Sales_Quantity': sales_quantity,
                    'Sales_Amount': sales_amount,
                    'Days_in_Period': days_in_period
                })
            
            monthly_summary = pd.DataFrame(monthly_data)
            
            if not monthly_summary.empty:
                # Calculate daily averages
                monthly_summary['Sales_Quantity_per_Day'] = (monthly_summary['Sales_Quantity'] / monthly_summary['Days_in_Period']).round(2)
                monthly_summary['Sales_Amount_per_Day'] = (monthly_summary['Sales_Amount'] / monthly_summary['Days_in_Period']).round(2)
                
                # Format display columns
                monthly_summary['Sales Amount'] = monthly_summary['Sales_Amount'].apply(lambda x: f"S/.{x:,.0f}")
                monthly_summary['Sales Amount per Day'] = monthly_summary['Sales_Amount_per_Day'].apply(lambda x: f"S/.{x:.2f}")
                monthly_summary['Sales Quant per Day'] = monthly_summary['Sales_Quantity_per_Day'].apply(lambda x: f"{x:.2f}")
                
                # Select and order columns for display
                display_columns = [
                    'Month', 'Initial_Date', 'Ending_Date', 'Sales_Quantity',
                    'Sales Quant per Day', 'Sales Amount', 'Sales Amount per Day'
                ]
                
                display_monthly = monthly_summary[display_columns].copy()
                display_monthly.columns = [
                    'Month', 'Initial Date', 'Ending Date', 'Sales Quantity',
                    'Sales Quant per Day', 'Sales Amount', 'Sales Amount per Day'
                ]
                
                # Apply green gradient to Sales Quant per Day
                min_quant = monthly_summary['Sales_Quantity_per_Day'].min()
                max_quant = monthly_summary['Sales_Quantity_per_Day'].max()
                
                def style_monthly_quant(val):
                    try:
                        num_val = float(val)
                        return get_green_gradient(num_val, min_quant, max_quant)
                    except:
                        return ''
                
                styled_monthly = display_monthly.style.applymap(
                    style_monthly_quant,
                    subset=['Sales Quant per Day']
                )
                
                # Display the table
                st.markdown(f'**{summary_view} Sales Summary**')
                st.dataframe(styled_monthly, use_container_width=True)
            else:
                st.warning("No data available for the selected time period.")

    # --- Inventory Analytics ---
    try:
        client = connect_to_sheets()
        if client:
            inventory_sheet = client.open_by_key('1WLn7DH3F1Sm5ZSEHgWVEILWvvjFRsrE0b9xKrYU43Hw').worksheet('Inventory')
            inventory_data = inventory_sheet.get_all_values()
            inventory_df = pd.DataFrame(inventory_data)

            st.markdown('## 📦 Inventory Analytics')

            # --- Inventory by Location Table ---
            loc_row = inventory_df.apply(lambda row: row.str.contains('Location', na=False)).any(axis=1)
            loc_start = loc_row[loc_row].index[0]
            loc_header = inventory_df.iloc[loc_start].tolist()
            try:
                first_col = loc_header.index('Location')
                last_col = loc_header.index('Total') + 1
            except ValueError:
                first_col = 0
                last_col = len(loc_header)
            loc_header = loc_header[first_col:last_col]
            # Dynamically find the end of the inventory table
            end_row = loc_start + 1
            while end_row < len(inventory_df):
                row_val = str(inventory_df.iloc[end_row, first_col]).strip()
                if row_val == '' or 'Total Inventory' in row_val:
                    break
                end_row += 1
            loc_table = inventory_df.iloc[loc_start+1:end_row, first_col:last_col]
            loc_table.columns = loc_header
            loc_table = loc_table.loc[:, ~loc_table.columns.duplicated()]
            loc_table = loc_table.reset_index(drop=True)

            # Convert numeric columns to float for calculations
            numeric_cols = loc_table.columns[1:]  # All columns except 'Location'
            for col in numeric_cols:
                loc_table[col] = pd.to_numeric(loc_table[col], errors='coerce').fillna(0)

            # Calculate column totals
            totals = loc_table[numeric_cols].sum()
            total_row = pd.DataFrame([['Total Location'] + list(totals)], columns=loc_table.columns)
            loc_table = pd.concat([loc_table, total_row], ignore_index=True)

            # Highlight low inventory (<10) in a more visible red and style totals
            def highlight_low_and_totals(val, is_total_row=False):
                try:
                    if is_total_row:
                        return 'font-weight: 900; font-size: 20px; background-color: #2C3E50; color: white; border: 2px solid #ECF0F1;'
                    num_val = float(val)
                    return 'background-color: #ff3333; color: white;' if num_val < 10 else ''
                except:
                    return ''

            # Set 'Location' as index to remove the default index column
            if 'Location' in loc_table.columns:
                loc_table = loc_table.set_index('Location')

            # Apply styling
            styled_loc = loc_table.style.apply(lambda x: [highlight_low_and_totals(v, i == len(x)-1) 
                                                        for i, v in enumerate(x)], axis=0)

            st.markdown('### Inventory by Location')
            st.dataframe(styled_loc, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Current Inventory table ---
            curr_inv_row = inventory_df.apply(lambda row: row.str.contains('Current Inventory', na=False)).any(axis=1)
            curr_inv_start = curr_inv_row[curr_inv_row].index[0]
            curr_inv_header = inventory_df.iloc[curr_inv_start].tolist()
            try:
                first_col = curr_inv_header.index('Spearmint')
                last_col = curr_inv_header.index('Total Mg') + 1
            except ValueError:
                first_col = 0
                last_col = len(curr_inv_header)
            curr_inv_header = curr_inv_header[first_col:last_col]
            curr_inv_table = inventory_df.iloc[curr_inv_start+1:curr_inv_start+6, first_col-1:last_col]
            curr_inv_table.columns = ['Index'] + curr_inv_header
            curr_inv_table = curr_inv_table.loc[:, ~curr_inv_table.columns.duplicated()]
            curr_inv_table = curr_inv_table.reset_index(drop=True)

            # Convert numeric columns to float
            numeric_columns = curr_inv_header
            for col in numeric_columns:
                curr_inv_table[col] = pd.to_numeric(curr_inv_table[col], errors='coerce').fillna(0)

            # Calculate column totals
            totals = curr_inv_table[numeric_columns].sum()
            total_row = pd.DataFrame([['Total Flavour'] + list(totals)], columns=['Index'] + numeric_columns)
            curr_inv_table = pd.concat([curr_inv_table, total_row])

            # Set first column as index (nicotine levels)
            curr_inv_table = curr_inv_table.set_index(curr_inv_table.columns[0])
            curr_inv_table.index.name = None

            # Style the table with color gradients and bold totals
            def style_curr_inv(df):
                # Create an empty DataFrame of strings with the same shape as our data
                styles = pd.DataFrame(index=df.index, columns=df.columns, data='')
                
                # Add color gradients based on thresholds (except for Total Mg column and Total Flavour row)
                for idx in df.index:
                    if idx != 'Total Flavour':
                        for col in df.columns:
                            if col != 'Total Mg':
                                try:
                                    val = float(df.loc[idx, col])
                                    if val >= 30:
                                        styles.loc[idx, col] = 'background-color: #90EE90; color: black'  # light green
                                    elif 15 <= val <= 29:
                                        styles.loc[idx, col] = 'background-color: #FFB347; color: black'  # orange
                                    elif val <= 14:
                                        styles.loc[idx, col] = 'background-color: #FFB6B6; color: black'  # light red
                                except:
                                    pass

                # Make Total Flavour row and Total Mg column bold with larger font and distinct styling
                total_style = 'font-weight: 900; font-size: 20px; background-color: #2C3E50; color: white; border: 2px solid #ECF0F1;'
                for col in df.columns:
                    styles.loc['Total Flavour', col] = total_style
                    if col == 'Total Mg':
                        for idx in df.index:
                            current_style = styles.loc[idx, col]
                            styles.loc[idx, col] = current_style + '; ' + total_style if current_style else total_style

                return styles

            styled_curr_inv = curr_inv_table.style.apply(style_curr_inv, axis=None)
            st.markdown('### Current Inventory')
            st.dataframe(styled_curr_inv, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Current + New Orders table ---
            plus_new_row = inventory_df.apply(lambda row: row.str.contains('Current \+ New Orders', na=False)).any(axis=1)
            plus_new_start = plus_new_row[plus_new_row].index[0]
            plus_new_header = inventory_df.iloc[plus_new_start].tolist()
            try:
                first_col = plus_new_header.index('Spearmint')
                last_col = plus_new_header.index('Total Mg') + 1
            except ValueError:
                first_col = 0
                last_col = len(plus_new_header)
            plus_new_header = plus_new_header[first_col:last_col]
            plus_new_table = inventory_df.iloc[plus_new_start+1:plus_new_start+6, first_col-1:last_col]
            plus_new_table.columns = ['Index'] + plus_new_header
            plus_new_table = plus_new_table.loc[:, ~plus_new_table.columns.duplicated()]
            plus_new_table = plus_new_table.reset_index(drop=True)

            # Convert numeric columns to float
            numeric_columns = plus_new_header
            for col in numeric_columns:
                plus_new_table[col] = pd.to_numeric(plus_new_table[col], errors='coerce').fillna(0)

            # Calculate column totals
            totals = plus_new_table[numeric_columns].sum()
            total_row = pd.DataFrame([['Total Flavour'] + list(totals)], columns=['Index'] + numeric_columns)
            plus_new_table = pd.concat([plus_new_table, total_row])

            # Set first column as index (nicotine levels)
            plus_new_table = plus_new_table.set_index(plus_new_table.columns[0])
            plus_new_table.index.name = None

            # Style the table with differences and bold totals
            def style_diff_and_total(df):
                # Create an empty DataFrame of strings with the same shape as our data
                styles = pd.DataFrame(index=df.index, columns=df.columns, data='')
                
                # Add yellow background to differences (except in total row)
                for idx in df.index:
                    if idx != 'Total Flavour':
                        for col in df.columns:
                            if col in curr_inv_table.columns and col != 'Total Mg':
                                try:
                                    if float(df.loc[idx, col]) != float(curr_inv_table.loc[idx, col]):
                                        styles.loc[idx, col] = 'background-color: yellow'
                                except (ValueError, KeyError, TypeError):
                                    pass
                
                # Make Total Flavour row and Total Mg column bold with larger font and distinct styling
                total_style = 'font-weight: 900; font-size: 20px; background-color: #2C3E50; color: white; border: 2px solid #ECF0F1;'
                for col in df.columns:
                    styles.loc['Total Flavour', col] = total_style
                    if col == 'Total Mg':
                        for idx in df.index:
                            current_style = styles.loc[idx, col]
                            styles.loc[idx, col] = current_style + '; ' + total_style if current_style else total_style
                
                return styles

            styled_plus_new = plus_new_table.style.apply(style_diff_and_total, axis=None)
            st.markdown('### Current + New Orders (differences highlighted)')
            st.dataframe(styled_plus_new, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load inventory analytics: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Enhanced Financial Results Statement from "Finances" Sheet ---
    try:
        client = connect_to_sheets()
        if client:
            spreadsheet = client.open_by_key('1WLn7DH3F1Sm5ZSEHgWVEILWvvjFRsrE0b9xKrYU43Hw')
            finances_ws = spreadsheet.worksheet('Finances')
            finances_data = finances_ws.get_all_records()
            finances_df = pd.DataFrame(finances_data)

            # Clean and prepare data
            finances_df = finances_df.rename(columns=lambda x: x.strip())
            finances_df = finances_df[finances_df['Amount (Local Currency)'].astype(str).str.strip() != '']
            finances_df['Amount (Local Currency)'] = (
                finances_df['Amount (Local Currency)']
                .astype(str)
                .str.replace('S/.', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.extract(r'([\d\.]+)')[0]
                .astype(float)
            )

            # Separate income and expenses
            income_df = finances_df[finances_df['+/-'] == 'Income']
            expense_df = finances_df[finances_df['+/-'] == 'Expense']

            # Revenue by product/concept
            revenue_by_product = (
                income_df.groupby('Concept')['Amount (Local Currency)']
                .sum()
                .sort_values(ascending=False)
            )
            revenue_total = revenue_by_product.sum()
            revenue_percent = (revenue_by_product / revenue_total * 100).round(2)

            # Expenses by concept
            expenses_by_concept = (
                expense_df.groupby('Concept')['Amount (Local Currency)']
                .sum()
                .sort_values(ascending=False)
            )
            expenses_total = expenses_by_concept.sum()
            expenses_percent = (expenses_by_concept / expenses_total * 100).round(2)

            # Totals and metrics
            net_result = revenue_total - expenses_total
            net_margin_pct = (net_result / revenue_total * 100) if revenue_total else np.nan

            # Header metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Revenue", f"S/. {revenue_total:,.2f}")
            with col2:
                st.metric("Total Expenses", f"S/. {expenses_total:,.2f}")
            with col3:
                st.metric("Net Profit Margin", f"{net_margin_pct:.2f}%")

            # Build enhanced DataFrame for display
            rows = []
            # Revenue section header
            rows.append(["REVENUE", "", "", "section_header"])
            for product, value in revenue_by_product.items():
                rows.append([
                    product,
                    f'S/. {value:,.2f}',
                    f'{revenue_percent[product]:.2f}%',
                    "revenue"
                ])
            rows.append(["Total Revenue", f'S/. {revenue_total:,.2f}', "100.00%", "revenue_total"])
            # Spacer row
            rows.append(["", "", "", "spacer"])
            # Expenses section header
            rows.append(["EXPENSES", "", "", "section_header"])
            for concept, value in expenses_by_concept.items():
                rows.append([
                    concept,
                    f'S/. {value:,.2f}',
                    f'{expenses_percent[concept]:.2f}%',
                    "expense"
                ])
            rows.append(["Total Expenses", f'S/. {expenses_total:,.2f}', "100.00%", "expense_total"])
            # Spacer row
            rows.append(["", "", "", "spacer"])
            # Net profit row
            rows.append(["Net Profit (Margin)", f'S/. {net_result:,.2f}', f'{net_margin_pct:.2f}%', "net_profit"])

            fin_df = pd.DataFrame(rows, columns=["Item", "Amount", "% of Total", "Type"])

            # Styling function
            def highlight_financials(row):
                style = [''] * (len(row)-1)  # Exclude 'Type' column
                if row['Type'] == 'section_header':
                    style = ['font-weight: bold; background-color: #222; color: #fff; font-size: 18px;'] * (len(row)-1)
                elif row['Type'] == 'revenue':
                    style = ['background-color: #e6ffe6; color: #222;'] * (len(row)-1)
                elif row['Type'] == 'revenue_total':
                    style = ['background-color: #00cc00; color: #fff; font-weight: bold;'] * (len(row)-1)
                elif row['Type'] == 'expense':
                    style = ['background-color: #ffe6e6; color: #222;'] * (len(row)-1)
                elif row['Type'] == 'expense_total':
                    style = ['background-color: #ff3333; color: #fff; font-weight: bold;'] * (len(row)-1)
                elif row['Type'] == 'net_profit':
                    style = ['background-color: #b3d1ff; color: #222; font-weight: bold;'] * (len(row)-1)
                elif row['Type'] == 'spacer':
                    style = ['background-color: #222; color: #222;'] * (len(row)-1)
                return style

            st.markdown("## 💰 Financial Results Statement")
            st.dataframe(
                fin_df.drop(columns=["Type"]).style.apply(highlight_financials, axis=1),
                use_container_width=True
            )

    except Exception as e:
        st.warning(f'Could not load financial analytics: {e}')

    # Raw data
    st.markdown("**Raw Data**")
    st.dataframe(df)
else:
    st.error("Please make sure you have set up the Google Sheets credentials correctly.") 
