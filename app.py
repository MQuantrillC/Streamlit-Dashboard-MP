import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Machu Pouches Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Machu Pouches Dashboard")

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
                start_date = st.date_input("Start Date", min_date, min_value=min_date)  # Removed max_value
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date)  # Removed max_value
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

    # Sales trend graph
        # Sales trend graph
    if 'Date' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add view selection
        trend_view = st.radio(
            "Select Sales Trend View",
            ("Daily", "Weekly", "Monthly"),
            horizontal=True
        )
        
        # Prepare data based on selected view
        if trend_view == "Daily":
            sales_trend = df.groupby('Date')['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Daily Sales Trend'
        elif trend_view == "Weekly":
            sales_trend = df.groupby(pd.Grouper(key='Date', freq='W-MON'))['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Weekly Sales Trend'
        else:  # Monthly
            sales_trend = df.groupby(pd.Grouper(key='Date', freq='M'))['Payment Amount (Numeric)'].sum().reset_index()
            title = 'Monthly Sales Trend'
        
        # Create the plot
        fig = px.line(sales_trend, x='Date', y='Payment Amount (Numeric)',
                     title=title)
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales Amount (S/.)",
            hovermode='x unified'
        )
        
        # Add hover template
        if trend_view == "Daily":
            fig.update_traces(
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Sales: S/.%{y:,.2f}<extra></extra>"
            )
        elif trend_view == "Weekly":
            fig.update_traces(
                hovertemplate="Week of: %{x|%Y-%m-%d}<br>Sales: S/.%{y:,.2f}<extra></extra>"
            )
        else:
            fig.update_traces(
                hovertemplate="Month: %{x|%Y-%m}<br>Sales: S/.%{y:,.2f}<extra></extra>"
            )
        
        st.plotly_chart(fig)

    st.markdown("<br>", unsafe_allow_html=True)
        
    # --- Repeat vs New Customers Analysis ---
    if 'Client ID' in df.columns and 'Date' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        # Add view selection
        customer_trend_view = st.radio(
            "Select Customer Analysis View",
            ("Daily", "Weekly", "Monthly"),
            horizontal=True,
            key="customer_analysis_view"
        )
        
        df_sorted = df.sort_values('Date')
        df_sorted['Is New Customer'] = ~df_sorted['Client ID'].duplicated()
        df_sorted['Customer Type'] = df_sorted['Is New Customer'].map({True: 'New Customers', False: 'Existing Customers'})
        
        # Prepare data based on selected view
        if customer_trend_view == "Daily":
            df_sorted['Time Period'] = df_sorted['Date'].dt.strftime('%Y-%m-%d')
            title = 'Daily New vs. Existing Customers'
        elif customer_trend_view == "Weekly":
            df_sorted['Time Period'] = df_sorted['Date'].dt.strftime('%Y-W%U')
            title = 'Weekly New vs. Existing Customers'
        else:  # Monthly
            df_sorted['Time Period'] = df_sorted['Date'].dt.strftime('%Y-%m')
            title = 'Monthly New vs. Existing Customers'

        # Group by selected time period
        period_sales = df_sorted.groupby(['Time Period', 'Customer Type'])['Payment Amount (Numeric)'].sum().unstack(fill_value=0)
        period_sales['Total'] = period_sales.sum(axis=1)
        period_sales['New %'] = (period_sales['New Customers'] / period_sales['Total'] * 100).round(2)
        period_sales['Existing %'] = (period_sales['Existing Customers'] / period_sales['Total'] * 100).round(2)

        # Prepare data for stacked bar chart
        chart_df = pd.DataFrame({
            'Time Period': period_sales.index,
            'New Customers': period_sales['New %'],
            'Existing Customers': period_sales['Existing %']
        })

        fig = px.bar(
            chart_df,
            x='Time Period',
            y=['New Customers', 'Existing Customers'],
            labels={'value': 'Percentage of Total Sales (%)', 'variable': 'Customer Type'},
            title=f'Percentage of Sales from New vs. Existing Customers ({customer_trend_view})',
            color_discrete_map={
                'New Customers': '#8ecae6',  # light blue
                'Existing Customers': '#003366'  # darker blue
            }
        )
        
        # Update layout for stacked bar chart
        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=-45,
            yaxis_range=[0, 100],
            showlegend=True,
            legend_title='Customer Type',
            yaxis_title='Percentage of Total Sales (%)',
            xaxis_title=f'{customer_trend_view} Period',
            hovermode='x unified'
        )
        
        # Update traces with dynamic text position based on percentage value
        for trace in fig.data:
            text_positions = ['outside' if val < 15 else 'inside' for val in trace.y]
            trace.update(
                texttemplate='%{y:.1f}%',
                textposition=text_positions,
                hovertemplate="%{y:.1f}%<br>"
            )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Weekly Sales Summary Table ---
    if 'Date' in df.columns and 'Amount' in df.columns and 'Payment Amount (Numeric)' in df.columns:
        # Add view selection
        summary_view = st.radio(
            "Select Summary View",
            ("Weekly", "Monthly"),
            horizontal=True,
            key="summary_view"
        )

        summary_df = df.copy()
        summary_df = summary_df.set_index('Date').sort_index()

        if summary_view == "Weekly":
            # Weekly Summary Logic
            first_date = summary_df.index.min().normalize()
            first_monday = first_date - pd.Timedelta(days=first_date.weekday())
            last_date = summary_df.index.max().normalize()
            # Create weekly bins from first Monday
            bins = pd.date_range(start=first_monday, end=last_date + pd.Timedelta(days=6), freq='7D')
            summary = summary_df.groupby(pd.cut(summary_df.index, bins, right=False)).agg(
                Initial_Date=('Amount', lambda x: x.index.min()),
                Ending_Date=('Amount', lambda x: x.index.max()),
                Sales_Quantity=('Amount', 'sum'),
                Sales_Amount_Numeric=('Payment Amount (Numeric)', 'sum')  # Changed this line
            ).reset_index(drop=True)
            
            # Fill missing dates for empty weeks
            summary['Initial Date'] = bins[:-1]
            summary['Ending Date'] = bins[1:] - pd.Timedelta(days=1)
            summary['Days_in_Period'] = 7
            
        else:  # Monthly Summary Logic
            # Group by month
            summary_df['Month'] = summary_df.index.to_period('M')
            summary = summary_df.groupby('Month').agg(
                Initial_Date=('Date', lambda x: x.min()),
                Ending_Date=('Date', lambda x: x.max()),
                Sales_Quantity=('Amount', 'sum'),
                Sales_Amount_Numeric=('Payment Amount (Numeric)', 'sum')  # Changed this line
            ).reset_index(drop=True)
            
            # Calculate days in each month for daily averages
            summary['Days_in_Period'] = (pd.to_datetime(summary['Ending_Date']) - 
                                       pd.to_datetime(summary['Initial_Date'])).dt.days + 1

        # Format dates
        summary['Initial Date'] = pd.to_datetime(summary['Initial_Date']).dt.strftime('%d/%m/%Y')
        summary['Ending Date'] = pd.to_datetime(summary['Ending_Date']).dt.strftime('%d/%m/%Y')
        
        # Calculate daily averages
        summary['Sales_Quantity_per_Day'] = round(summary['Sales_Quantity'] / summary['Days_in_Period'], 2)
        summary['Sales_Amount_per_Day'] = round(summary['Sales_Amount_Numeric'] / summary['Days_in_Period'], 2)

        # Format the display columns
        summary['Sales Amount'] = summary['Sales_Amount'].apply(lambda x: f"S/.{x:,.0f}")
        summary['Sales Amount per Day'] = summary['Sales_Amount_per_Day'].apply(lambda x: f"S/.{x:,.2f}")
        summary['Sales Quant per Day'] = summary['Sales_Quantity_per_Day'].apply(lambda x: f"{x:.2f}")
        
        # Select and rename columns for display
        display_columns = ['Initial Date', 'Ending Date', 'Sales_Quantity', 'Sales Quant per Day',
                         'Sales Amount', 'Sales Amount per Day']
        
        display_summary = summary[display_columns].copy()
        display_summary.columns = ['Initial Date', 'Ending Date', 'Sales Quantity', 'Sales Quant per Day',
                                 'Sales Amount', 'Sales Amount per Day']
        
        st.markdown(f'**{summary_view} Sales Summary**')
        
        # Convert Sales Quant per Day to numeric for comparison
        numeric_values = display_summary['Sales Quant per Day'].astype(str).str.replace(',', '').astype(float)
        max_val = numeric_values.max()
        min_val = numeric_values.min()
        
        # Custom styling function with darker green colors
        def color_scale(val):
            try:
                val = float(str(val).replace(',', ''))
                # Calculate the intensity (0 to 1)
                intensity = (val - min_val) / (max_val - min_val) if max_val != min_val else 0
                # Create a color scale from light to darker green
                r = int(200 - (intensity * 70))   # from 200 to 130
                g = int(220 - (intensity * 40))   # from 220 to 180
                b = int(200 - (intensity * 70))   # from 200 to 130
                return f'background-color: rgb({r},{g},{b})'
            except:
                return ''

        # Apply the custom styling
        styled_summary = display_summary.style.applymap(
            color_scale,
            subset=['Sales Quant per Day']
        )
        
        st.dataframe(
            styled_summary,
            use_container_width=True
        )

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
            loc_table = inventory_df.iloc[loc_start+1:loc_start+15, first_col:last_col]
            loc_table.columns = loc_header
            loc_table = loc_table.loc[:, ~loc_table.columns.duplicated()]
            loc_table = loc_table.reset_index(drop=True)
            # Highlight low inventory (<10) in a more visible red
            def highlight_low(val):
                try:
                    return 'background-color: #ff3333; color: white;' if float(val) < 10 else ''
                except:
                    return ''
            styled_loc = loc_table.style.applymap(highlight_low, subset=[col for col in loc_table.columns if col not in ['Location', 'Total']])
            st.markdown('### Inventory by Location')
            st.dataframe(styled_loc, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Find and extract Current Inventory table anywhere in the sheet ---
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
            curr_inv_table = inventory_df.iloc[curr_inv_start+1:curr_inv_start+6, first_col:last_col]
            curr_inv_table.columns = curr_inv_header
            curr_inv_table = curr_inv_table.loc[:, ~curr_inv_table.columns.duplicated()]
            curr_inv_table = curr_inv_table.reset_index(drop=True)
            st.markdown('### Current Inventory')
            st.dataframe(curr_inv_table, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Find and extract Current + New Orders table anywhere in the sheet ---
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
            plus_new_table = inventory_df.iloc[plus_new_start+1:plus_new_start+6, first_col:last_col]
            plus_new_table.columns = plus_new_header
            plus_new_table = plus_new_table.loc[:, ~plus_new_table.columns.duplicated()]
            plus_new_table = plus_new_table.reset_index(drop=True)

            # --- Highlight differences ---
            def highlight_diff(val, ref):
                try:
                    return 'background-color: yellow' if str(val) != str(ref) else ''
                except:
                    return ''
            styled_plus_new = plus_new_table.style.apply(
                lambda x: [highlight_diff(x[c], curr_inv_table.iloc[x.name, plus_new_table.columns.get_loc(c)]) if c in curr_inv_table.columns else '' for c in plus_new_table.columns], axis=1
            )
            st.markdown('### Current + New Orders (differences highlighted)')
            st.dataframe(styled_plus_new, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load inventory analytics: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Raw data
    st.markdown("**Raw Data**")
    st.dataframe(df)
else:
    st.error("Please make sure you have set up the Google Sheets credentials correctly.") 
