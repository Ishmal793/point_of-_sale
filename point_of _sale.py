import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
file_path = 'superstore.xlsx'
df = pd.read_excel(file_path, sheet_name='superstore_dataset')

# Function to load default data
@st.cache_data
def load_default_data():
    return pd.read_excel(
        'superstore.xlsx',
        sheet_name='superstore_dataset',
        engine='openpyxl'
    )

# Function to load uploaded files (supports Excel and CSV)
def load_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.error("Unsupported file type! Please upload an Excel or CSV file.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()

# Sidebar for file upload or default dataset
st.sidebar.title("Upload or Load Dataset")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ("Default Dataset", "Upload Your Own Dataset")
)

# Load dataset based on user input
if data_source == "Default Dataset":
    data = load_default_data()
    st.sidebar.success("Default dataset loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        data = load_uploaded_file(uploaded_file)
        st.sidebar.success("Dataset uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a dataset to proceed.")
        st.stop()

# Define color palettes
default_colors = px.colors.qualitative.Plotly
time_series_colors = px.colors.qualitative.Set2
# Refresh Button
if st.button("Refresh Dashboard"):
    st.experimental_set_query_params()

# Tooltip Message
tooltip_message = (
    "The dataset is a working process. You cannot open the Excel file directly, "
    "and no modifications can be made. You can only add data to existing columns, "
    "and you cannot change the column names."
)
st.markdown(
    f'<span style="color: grey; font-size: 12px; text-decoration: underline;">{tooltip_message}</span>',
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.title("Point of Sale Analysis")
options = st.sidebar.radio(
    "Select Analysis Type",
    ["Overall Overview", "Sales by Product Category", "Daily & Hourly Sales Trend","Customer Sales Analytics",
     "Inventory Turnover Rate", "Profit Margin by Product and Category", "Discount Effectiveness Analysis"]
)

# Sidebar filters
st.sidebar.header("Filters")

# Date filters positioned at the top
min_date, max_date = min(df['order_date']), max(df['order_date'])
start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Display an error if the start date is after the end date
if start_date > end_date:
    st.sidebar.error("Start Date cannot be after End Date")

# Additional filters
category_filter = st.sidebar.multiselect("Select Product Category", options=df['category'].unique())
region_filter = st.sidebar.multiselect("Select Region", options=df['region'].unique())
product_filter = st.sidebar.multiselect("Select Product", options=df['product_name'].unique())
segment_filter = st.sidebar.multiselect("Select Segment", options=df['segment'].unique())
subcategory_filter = st.sidebar.multiselect("Select Subcategory", options=df['subcategory'].unique())
state_filter = st.sidebar.multiselect("Select State", options=df['state'].unique())
city_filter = st.sidebar.multiselect("Select City", options=df['city'].unique())

# Filter the dataset based on sidebar selections with conditional checks
filtered_df = df[
    (df['order_date'] >= pd.to_datetime(start_date)) &
    (df['order_date'] <= pd.to_datetime(end_date)) &
    (df['category'].isin(category_filter) if category_filter else True) &
    (df['region'].isin(region_filter) if region_filter else True) &
    (df['product_name'].isin(product_filter) if product_filter else True) &
    (df['segment'].isin(segment_filter) if segment_filter else True) &
    (df['subcategory'].isin(subcategory_filter) if subcategory_filter else True) &
    (df['state'].isin(state_filter) if state_filter else True) &
    (df['city'].isin(city_filter) if city_filter else True)
]

# Overall Overview
if options == "Overall Overview":
    st.header("Overall Business Overview")

    # Overall metrics
    total_sales = filtered_df['sales'].sum()
    total_rows = len(filtered_df)
    total_profit = filtered_df['profit'].sum()
    total_discount = filtered_df['discount'].sum()
    total_quantity = filtered_df['quantity'].sum()
    avg_profit_margin = (filtered_df['profit'].sum() / filtered_df['sales'].sum()) * 100 if total_sales != 0 else 0

    # Creating a grid for the gauge charts (3 charts per row)
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_sales = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_sales,
            title={'text': "Total Sales"},
            gauge={'axis': {'range': [0, total_sales * 1.2]},
                   'bar': {'color': "darkblue"}}
        ))
        fig_sales.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_sales, use_container_width=True)

    with col2:
        fig_profit = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_profit,
            title={'text': "Total Profit"},
            gauge={'axis': {'range': [0, total_profit * 1.2]},
                   'bar': {'color': "green"}}
        ))
        fig_profit.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_profit, use_container_width=True)

    with col3:
        fig_discount = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_discount,
            title={'text': "Total Discount"},
            gauge={'axis': {'range': [0, total_discount * 1.2]},
                   'bar': {'color': "orange"}}
        ))
        fig_discount.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_discount, use_container_width=True)

    # Second row of metrics
    col4, col5, col6 = st.columns(3)
    with col4:
        fig_quantity = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_quantity,
            title={'text': "Total Quantity Sold"},
            gauge={'axis': {'range': [0, total_quantity * 1.2]},
                   'bar': {'color': "purple"}}
        ))
        fig_quantity.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_quantity, use_container_width=True)

    with col5:
        fig_rows = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_rows,
            title={'text': "Total Number of Rows"},
            gauge={'axis': {'range': [0, total_rows * 1.2]},
                   'bar': {'color': "teal"}}
        ))
        fig_rows.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_rows, use_container_width=True)

    with col6:
        fig_margin = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_profit_margin,
            title={'text': "Average Profit Margin (%)"},
            gauge={'axis': {'range': [0, avg_profit_margin * 1.2]},
                   'bar': {'color': "red"}}
        ))
        fig_margin.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_margin, use_container_width=True)
    # First Plot: Total Sales by Region
    st.subheader("Total Sales by Region")

    # Aggregate total sales by region
    total_sales_by_region = df.groupby('region')['sales'].sum().reset_index()

    # Create a Plotly bar chart for total sales by region
    fig1 = px.bar(total_sales_by_region,
                  x='region',
                  y='sales',
                  title='Total Sales by Region',
                  labels={'region': 'Region', 'sales': 'Total Sales'},
                  color='region',  # Color by region for better distinction
                  color_discrete_sequence=px.colors.qualitative.T10)

    # Customize layout with transparent background
    fig1.update_layout(
        xaxis_title='Region',
        yaxis_title='Total Sales',
        title_x=0.5,
        template='plotly_white',
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent overall background
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig1)

    # Second Plot: Average Profit Margin by Region
    st.subheader("Average Profit Margin by Region")

    # Calculate average profit margin by region
    avg_profit_margin_by_region = df.groupby('region')['profit_margin'].mean().reset_index()

    # Create a Plotly bar chart for average profit margin by region
    fig2 = px.bar(avg_profit_margin_by_region,
                  x='region',
                  y='profit_margin',
                  title='Average Profit Margin by Region',
                  labels={'region': 'Region', 'profit_margin': 'Average Profit Margin'},
                  color='region',  # Color by region for better distinction
                  color_discrete_sequence=px.colors.qualitative.T10)

    # Customize layout with transparent background
    fig2.update_layout(
        xaxis_title='Region',
        yaxis_title='Average Profit Margin',
        title_x=0.5,
        template='plotly_white',
        width=700,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent overall background
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig2)

    # Display first or last 5 rows of the data as a sample
    sample_data = st.radio("View Data Sample", ["First 5 rows", "Last 5 rows"])
    if sample_data == "First 5 rows":
        st.dataframe(filtered_df.head())
    else:
        st.dataframe(filtered_df.tail())

# Sales by Product Category
elif options == "Sales by Product Category":
    # Product Category Analysis Charts
    st.header("Sales and Profit Analysis by Product Category")

    # Aggregate data for sales and profit by product category
    category_sales_profit = filtered_df.groupby('category').agg({
        'sales': 'sum',
        'profit': 'sum'
    }).reset_index()

    # First Chart: Total Sales by Product Category (Bar Chart)
    fig_sales = px.bar(
        category_sales_profit,
        x='category',
        y='sales',
        title='Total Sales by Product Category',
        labels={'category': 'Product Category', 'sales': 'Total Sales'},
        color='category',
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_sales.update_layout(
        xaxis_title='Product Category',
        yaxis_title='Total Sales',
        title_x=0.5,
        template='plotly_dark',


    )
    st.plotly_chart(fig_sales)

    # Second Chart: Profit by Product Category (Bar Chart)
    fig_profit = px.bar(
        category_sales_profit,
        x='category',
        y='profit',
        title='Total Profit by Product Category',
        labels={'category': 'Product Category', 'profit': 'Total Profit'},
        color='category',
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_profit.update_layout(
        xaxis_title='Product Category',
        yaxis_title='Total Profit',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_profit)

    # Combined Chart: Scatter plot for comparing Sales and Profit
    fig_combined = px.scatter(
        category_sales_profit,
        x='sales',
        y='profit',
        text='category',
        title='Sales vs. Profit by Product Category',
        labels={'sales': 'Total Sales', 'profit': 'Total Profit'},
        color='category',
        size='sales',
        size_max=20,
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_combined.update_traces(textposition='top center')
    fig_combined.update_layout(
        xaxis_title='Total Sales',
        yaxis_title='Total Profit',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_combined)

    # Ensure 'order_date' is in datetime format before extracting year and month
    filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'], errors='coerce')

    # Now extract the year and month
    filtered_df['year'] = filtered_df['order_date'].dt.year
    filtered_df['month'] = filtered_df['order_date'].dt.month

    # Proceed with aggregating data for yearly sales and profit by product category
    yearly_category_sales_profit = filtered_df.groupby(['year', 'category']).agg({
        'sales': 'sum',
        'profit': 'sum'
    }).reset_index()

    # Generate the charts (the code for the charts remains the same as before)
    # Chart 1: Yearly Sales by Product Category
    fig_yearly_sales = px.line(
        yearly_category_sales_profit,
        x='year',
        y='sales',
        color='category',
        title='Yearly Sales by Product Category',
        labels={'year': 'Year', 'sales': 'Total Sales'},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_yearly_sales.update_layout(
        xaxis_title='Year',
        yaxis_title='Total Sales',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_yearly_sales)

    # Chart 2: Yearly Profit by Product Category
    fig_yearly_profit = px.line(
        yearly_category_sales_profit,
        x='year',
        y='profit',
        color='category',
        title='Yearly Profit by Product Category',
        labels={'year': 'Year', 'profit': 'Total Profit'},
        markers=True,
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_yearly_profit.update_layout(
        xaxis_title='Year',
        yaxis_title='Total Profit',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_yearly_profit)


# Daily & Hourly Sales Trend
elif options == "Daily & Hourly Sales Trend":
    st.header("Daily and Hourly Sales Trend")

    # Display top 5 customers by sales
    st.subheader("Top 5 Customers by Sales")
    top_customers = filtered_df.groupby('customer')['sales'].sum().nlargest(5).reset_index()
    st.dataframe(top_customers)

    # Select visualization level (day-wise or hour-wise)
    time_visualization = st.radio("Select Time-based Visualization", ("Day-wise", "Hour-wise"))

    # Total sales calculation
    total_sales = filtered_df['sales'].sum()
    st.subheader(f"Total Sales: ${total_sales:,.2f}")

    # If no data available
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        if time_visualization == "Day-wise":
            # Day-wise Sales
            filtered_df['day'] = filtered_df['order_date'].dt.date
            sales_over_time = filtered_df.groupby('day')['sales'].sum().reset_index()

            fig_time = px.line(
                sales_over_time,
                x='day',
                y='sales',
                title="Sales Over Time (Day-wise)",
                markers=True,
                color_discrete_sequence=["#FF5733"]
            )
            fig_time.update_traces(line=dict(width=2.5))
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Sales", template="plotly_dark")

        else:
            # Hour-wise Sales
            filtered_df['hour'] = filtered_df['order_date'].dt.hour
            selected_hours = st.sidebar.multiselect(
                "Select Hours", options=sorted(filtered_df['hour'].unique()),
                default=sorted(filtered_df['hour'].unique())
            )

            # Filter the dataset by selected hours
            if selected_hours:
                filtered_df = filtered_df[filtered_df['hour'].isin(selected_hours)]

            # Calculate total sales again after hour filter
            total_sales_hour = filtered_df['sales'].sum()
            st.subheader(f"Total Sales for Selected Hours: ${total_sales_hour:,.2f}")

            # Group data by hour for the line chart
            sales_over_time = filtered_df.groupby('hour')['sales'].sum().reset_index()

            fig_time = px.line(
                sales_over_time,
                x='hour',
                y='sales',
                title="Sales Over Time (Hour-wise)",
                markers=True,
                color_discrete_sequence=["#1E90FF"]
            )
            fig_time.update_traces(line=dict(width=2.5))
            fig_time.update_layout(xaxis_title="Hour", yaxis_title="Sales", template="plotly_dark")

        # Display the line chart
        st.plotly_chart(fig_time)

elif options=="Customer Sales Analytics":
    st.header("Customer Sales Analytics")

    # Show Total Number of Customers
    total_customers = df['customer'].nunique()
    st.subheader(f"Total Number of Customers: {total_customers}")

    # Display top 5 customers by profit
    st.subheader("Top 5 Customers by Profit")
    top_customers = df.groupby('customer')['profit'].sum().nlargest(5).reset_index()
    st.dataframe(top_customers)
    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
    else:
        # Select a customer to filter data
        selected_customer = st.selectbox("Select Customer", options=filtered_df['customer'].unique())
        customer_data = filtered_df.loc[filtered_df['customer'] == selected_customer].copy()

        st.subheader(f"Sales for Customer: {selected_customer}")

        # Display table for customer purchase details
        st.write("Purchase Details")
        st.dataframe(customer_data[['order_date', 'product_name', 'sales', 'quantity']])

        # Visualize sales by product for this customer
        product_sales = customer_data.groupby('product_name')['sales'].sum().reset_index()
        fig = px.bar(product_sales, y='product_name', x='sales', title=f'Sales by Product for {selected_customer}')
        st.plotly_chart(fig)

        # Visualize purchase history over time for this customer
        sales_over_time = customer_data.groupby('order_date')['sales'].sum().reset_index()
        fig = px.line(sales_over_time, x='order_date', y='sales', title=f'Sales Over Time for {selected_customer}',
                      markers=True)
        st.plotly_chart(fig)

# Inventory Turnover Rate (ITR)
elif options == "Inventory Turnover Rate":
    # Header
    st.header("Inventory Turnover Rate (ITR) Analysis")

    # Calculate Inventory Turnover Rate (ITR)
    average_inventory = 100  # Placeholder; adjust based on actual data
    itr_data = df.groupby('product_name')['sales'].sum().reset_index()
    itr_data['ITR'] = itr_data['sales'] / average_inventory

    # Split ITR data into fast-moving and slow-moving products
    threshold = itr_data['ITR'].mean()  # Define a threshold based on average ITR
    fast_moving = itr_data[itr_data['ITR'] > threshold].sort_values(by='ITR', ascending=False)
    slow_moving = itr_data[itr_data['ITR'] <= threshold].sort_values(by='ITR', ascending=True)

    # Display fast-moving products
    st.subheader("Fast-Moving Products")
    st.write("These products have high inventory turnover rates, indicating frequent sales.")
    st.dataframe(fast_moving[['product_name', 'sales', 'ITR']])

    # Bar chart for fast-moving products with product names on the y-axis
    fig_fast_moving = px.bar(
        fast_moving,
        y='product_name',
        x='ITR',
        title="Inventory Turnover Rate for Fast-Moving Products",
        labels={'ITR': 'Inventory Turnover Rate (ITR)', 'product_name': 'Product'},
        color='ITR',
        color_continuous_scale='Teal'
    )
    fig_fast_moving.update_layout(
        yaxis_title='Product',
        xaxis_title='ITR',
        title_x=0.5,
        template='plotly_white'
    )
    st.plotly_chart(fig_fast_moving)

    # Display slow-moving products
    st.subheader("Slow-Moving Products")
    st.write("These products have low inventory turnover rates, indicating slow sales.")
    st.dataframe(slow_moving[['product_name', 'sales', 'ITR']])

    # Bar chart for slow-moving products with product names on the y-axis
    fig_slow_moving = px.bar(
        slow_moving,
        y='product_name',
        x='ITR',
        title="Inventory Turnover Rate for Slow-Moving Products",
        labels={'ITR': 'Inventory Turnover Rate (ITR)', 'product_name': 'Product'},
        color='ITR',
        color_continuous_scale='OrRd'
    )
    fig_slow_moving.update_layout(
        yaxis_title='Product',
        xaxis_title='ITR',
        title_x=0.5,
        template='plotly_white'
    )
    st.plotly_chart(fig_slow_moving)

    # Summary table
    st.subheader("Inventory Turnover Summary")
    summary_table = itr_data.copy()
    summary_table['Status'] = summary_table['ITR'].apply(lambda x: 'Fast-Moving' if x > threshold else 'Slow-Moving')
    st.dataframe(summary_table[['product_name', 'sales', 'ITR', 'Status']])

    # Pie chart for summary visualization
    status_counts = summary_table['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_status_pie = px.pie(
        status_counts,
        names='Status',
        values='Count',
        title="Proportion of Fast- and Slow-Moving Products",
        color='Status',
        color_discrete_map={'Fast-Moving': 'green', 'Slow-Moving': 'red'}
    )
    fig_status_pie.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    st.plotly_chart(fig_status_pie)
# Profit Margin by Product and Category
elif options == "Profit Margin by Product and Category":
    # Section: Profit Margin Analysis by Product and Category
    st.header("Profit Margin Analysis by Product and Category")

    # Calculate profit margin as a percentage
    filtered_df['profit_margin'] = (filtered_df['profit'] / filtered_df['sales']) * 100

    # Group by category and product to calculate average profit margin
    category_product_margin = filtered_df.groupby(['category', 'product_name'])['profit_margin'].mean().reset_index()

    # Toggle to view top 5 and bottom 5 products by profit margin
    margin_toggle = st.radio("Select Margin View", ("High Margin Products", "Low Margin Products"))

    # Define colors for categories (you can customize these colors as needed)
    category_colors = {
        "Category 1": "rgb(66, 135, 245)",  # Example color for Category 1
        "Category 2": "rgb(245, 66, 66)",  # Example color for Category 2
        "Category 3": "rgb(66, 245, 173)",  # Example color for Category 3
        "Category 4": "rgb(245, 221, 66)",  # Example color for Category 4
        # Add more colors for each category as needed
    }

    if margin_toggle == "High Margin Products":
        # Top 5 High Margin Products per Category
        st.subheader("High Margin Products per Category")
        top_margin_products = category_product_margin.groupby('category').apply(
            lambda x: x.nlargest(5, 'profit_margin')).reset_index(drop=True)

        # Display in table format
        st.dataframe(top_margin_products)

        # Bar chart for top 5 high-margin products per category
        fig_top_margin = px.bar(
            top_margin_products,
            x='profit_margin',
            y='product_name',
            color='category',
            title="High Margin Products by Category",
            labels={'profit_margin': 'Profit Margin (%)', 'product_name': 'Product'},
            orientation='h',
            color_discrete_map=category_colors  # Assign colors based on category
        )
        fig_top_margin.update_layout(xaxis_title='Profit Margin (%)', yaxis_title='Product', title_x=0.5,
                                     template='plotly_white')
        st.plotly_chart(fig_top_margin)

    elif margin_toggle == "Low Margin Products":
        # Bottom 5 Low Margin Products per Category
        st.subheader("Low Margin Products per Category")
        bottom_margin_products = category_product_margin.groupby('category').apply(
            lambda x: x.nsmallest(5, 'profit_margin')).reset_index(drop=True)

        # Display in table format
        st.dataframe(bottom_margin_products)

        # Bar chart for top 5 low-margin products per category
        fig_bottom_margin = px.bar(
            bottom_margin_products,
            x='profit_margin',
            y='product_name',
            color='category',
            title="Low Margin Products by Category",
            labels={'profit_margin': 'Profit Margin (%)', 'product_name': 'Product'},
            orientation='h',
            color_discrete_map=category_colors  # Assign colors based on category
        )
        fig_bottom_margin.update_layout(xaxis_title='Profit Margin (%)', yaxis_title='Product', title_x=0.5,
                                        template='plotly_white')
        st.plotly_chart(fig_bottom_margin)


# Discount Effectiveness Analysis
elif options == "Discount Effectiveness Analysis":
    st.header("Discount Effectiveness Analysis")

    # Show overall discount impact (if no filter is applied)
    st.write("### Overall Discount Strategy Impact on Sales and Profit")
    overall_discount_impact = df.groupby('discount')[['sales', 'profit']].sum().reset_index()

    # Show overall discount impact using a line chart
    fig_overall = px.line(overall_discount_impact, x='discount', y=['sales', 'profit'],
                          title="Overall Sales and Profit by Discount",
                          labels={'sales': 'Total Sales', 'profit': 'Total Profit'},
                          markers=True)
    fig_overall.update_traces(mode='lines+markers')
    fig_overall.update_layout(
        xaxis_title='Discount',
        yaxis_title='Amount',
        legend_title='Metrics'
    )
    # Customize colors for the lines
    fig_overall.update_traces(line=dict(color='blue'), selector=dict(name='sales'))
    fig_overall.update_traces(line=dict(color='red'), selector=dict(name='profit'))

    # Add hover data to display detailed information
    fig_overall.update_traces(
        hovertemplate='Discount: %{x}<br>Sales: %{y}<br>Profit: %{customdata[1]}<extra></extra>',
        customdata=overall_discount_impact[['discount', 'profit']].values
    )
    st.plotly_chart(fig_overall)

    filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'], errors='coerce')

    # 2. Discount vs. Profit Margin (Scatter Plot)
    fig_discount_profit_margin = px.scatter(
        filtered_df,
        x='discount',
        y='profit_margin',
        size='sales',
        color='profit_margin',
        title='Discount vs. Profit Margin',
        labels={'discount': 'Discount (%)', 'profit_margin': 'Profit Margin'},
        color_continuous_scale=px.colors.diverging.RdYlGn,
        size_max=20
    )
    fig_discount_profit_margin.update_layout(
        xaxis_title='Discount (%)',
        yaxis_title='Profit Margin',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_discount_profit_margin)

    # 3. Sales and Profit Trends by Discount Range (Box Plot)
    # Define discount ranges (bins) for grouping
    discount_bins = pd.cut(filtered_df['discount'], bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
                           labels=['0-10%', '10-20%', '20-30%', '30-50%', '50-100%'])
    filtered_df['discount_range'] = discount_bins

    discount_range_sales_profit = filtered_df.groupby('discount_range').agg({
        'sales': 'sum',
        'profit': 'sum'
    }).reset_index()

    fig_discount_range_sales_profit = px.bar(
        discount_range_sales_profit,
        x='discount_range',
        y=['sales', 'profit'],
        title='Sales and Profit by Discount Range',
        labels={'discount_range': 'Discount Range', 'value': 'Amount', 'variable': 'Metrics'},
        color_discrete_sequence=px.colors.qualitative.T10,
        barmode='group'
    )
    fig_discount_range_sales_profit.update_layout(
        xaxis_title='Discount Range',
        yaxis_title='Amount',
        title_x=0.5,
        template='plotly_dark'
    )
    st.plotly_chart(fig_discount_range_sales_profit)

