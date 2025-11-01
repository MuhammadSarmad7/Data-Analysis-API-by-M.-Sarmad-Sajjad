import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# --- Configuration and Setup (Minimalist Aesthetic) ---
st.set_page_config(
    page_title="Data Analysis API",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a unique, minimalist look with a fresh color palette
st.markdown("""
<style>
/* Light, clean background */
.stApp {{
    background-color: #F8F8F8; 
}}
/* Dark, professional text */
body, .st-emotion-cache-1jm926g, h1, h2, h3, h4 {{ 
    color: #333333; 
}}
/* Accent Color: A professional teal for impact */
.st-emotion-cache-1jm926g, h1 {{ 
    color: #007ACC; 
    font-weight: 700;
}}
/* Subtle divider line */
h1 {{
    border-bottom: 1px solid #DDDDDD;
    padding-bottom: 5px;
}}
/* Sidebar Header */
.st-emotion-cache-16dsld0 {{ 
    color: #007ACC;
    font-size: 1.2em;
    font-weight: 600;
}}
/* Customizing component containers for clean separation */
.st-emotion-cache-13v0vgi {{
    border: 1px solid #DDDDDD;
    border-radius: 5px;
    padding: 10px;
}}
</style>
""", unsafe_allow_html=True)

st.title("💡 Interactive Data Analysis and Visualization Dashboard")
st.markdown("An interactive dashboard for exploring, cleaning, and visualizing datasets, built using Streamlit and core Python APIs.")
st.markdown("---")

# Initialize session state for the DataFrame
if 'df' not in st.session_state or 'df_original' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['df_original'] = None

# --- Custom Functions ---

def load_data(uploaded_file):
    """Reads the uploaded CSV file into a pandas DataFrame with validation."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded file is empty. Please upload a file with data.")
            return
            
        st.session_state['df_original'] = df.copy()
        st.session_state['df'] = df.copy()
        st.success("Dataset loaded successfully! Navigate to '2. Data Summary' to begin exploration.")
    except Exception as e:
        st.error(f"Error loading file. Please ensure it is a valid, uncorrupted CSV file: {e}")

def get_column_types(df):
    """Categorizes columns into numeric and categorical."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    all_cols = df.columns.tolist()
    return numeric_cols, categorical_cols, all_cols

def create_download_report(df):
    """Generates descriptive statistics and returns a downloadable CSV."""
    description = df.describe(include='all').T.reset_index()
    description.rename(columns={'index': 'Statistic'}, inplace=True)
    
    csv_buffer = BytesIO()
    description.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer.getvalue(), "descriptive_report.csv"

# --- Sidebar Navigation (6. Interactive Navigation) ---
st.sidebar.header("Navigation Menu")
menu_selection = st.sidebar.radio(
    "Select Analysis Step:",
    (
        "1. Dataset Upload", 
        "2. Data Summary", 
        "3. Data Visualization", 
        "4. Missing Data Handling", 
        "5. Download Report",
        "6. About Dashboard"
    )
)

# --- Main Content Area ---

df = st.session_state['df']

# 1. Dataset Upload (Core Feature 1)
if menu_selection == "1. Dataset Upload":
    st.header("1. Dataset Upload")
    uploaded_file = st.file_uploader(
        "📂 Upload your CSV file (.csv only)",
        type=['csv'],
        help="Only CSV files are supported. Input validation is applied to check file type and integrity."
    )
    
    if uploaded_file is not None:
        load_data(uploaded_file)
    
    if df is not None:
        st.subheader("Data Preview and Structure")
        st.dataframe(df.head())
        
        col_r, col_c = st.columns(2)
        with col_r:
            st.metric("Total Rows", df.shape[0])
        with col_c:
            st.metric("Total Columns", df.shape[1])
        
        st.subheader("Data Types and Nulls")
        numeric_cols, categorical_cols, _ = get_column_types(df)
        
        if not numeric_cols:
            st.warning("Warning: No numerical columns detected in the dataset.")
        if not categorical_cols:
            st.warning("Warning: No categorical columns detected in the dataset.")

        # Display structure including missing values
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%'
        }).reset_index(drop=True)
        st.dataframe(dtype_df, use_container_width=True)

# 2. Data Summary (Core Feature 2)
elif menu_selection == "2. Data Summary":
    st.header("2. Data Summary")
    if df is None:
        st.warning("Please upload a dataset first in the '1. Dataset Upload' section.")
    else:
        numeric_cols, categorical_cols, _ = get_column_types(df)
        
        st.subheader("Descriptive Statistics")
        
        if numeric_cols:
            st.markdown("#### Numerical Columns")
            # Calculate mean, median, std dev, min, max (NumPy operations via Pandas)
            st.dataframe(df[numeric_cols].describe().T)
        
        if categorical_cols:
            st.markdown("#### Categorical Columns")
            st.dataframe(df[categorical_cols].describe(include='object').T)

# 3. Data Visualization (Core Feature 3 - Complete Plotting Suite)
elif menu_selection == "3. Data Visualization":
    st.header("3. Data Visualization")
    if df is None:
        st.warning("Please upload a dataset first in the '1. Dataset Upload' section.")
    else:
        numeric_cols, categorical_cols, all_cols = get_column_types(df)
        
        st.sidebar.subheader("Plot Controls")
        
        visualization_type = st.sidebar.selectbox(
            "Select Plot Type",
            [
                "Histogram (1 Numeric)", 
                "Count Plot (1 Categorical)",
                "Bar Chart (1 Cat vs 1 Num)", 
                "Box Plot (1 Numeric)", 
                "Violin Plot (1 Num vs Opt Cat)", 
                "Scatter Plot (2 Numeric)",
                "Heatmap (Correlation)", 
                "Pairplot (Multiple Numeric)"
            ]
        )
        
        st.subheader(f"Plot Type: {visualization_type}")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Set seaborn style for clean look
        sns.set_style("whitegrid")

        try:
            if visualization_type == "Histogram (1 Numeric)":
                if numeric_cols:
                    col = st.sidebar.selectbox("Select Column (Numeric)", numeric_cols)
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#007ACC')
                    ax.set_title(f'Distribution of {col}', fontsize=14)
                    st.pyplot(fig)
                else:
                    st.warning("A Histogram requires at least one numerical column.")
                    
            elif visualization_type == "Count Plot (1 Categorical)":
                if categorical_cols:
                    col = st.sidebar.selectbox("Select Column (Categorical)", categorical_cols)
                    sns.countplot(x=col, data=df, ax=ax, palette='viridis')
                    ax.set_title(f'Frequency Count of {col}', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.warning("A Count Plot requires at least one categorical column.")

            elif visualization_type == "Bar Chart (1 Cat vs 1 Num)":
                if categorical_cols and numeric_cols:
                    col_x = st.sidebar.selectbox("Select X-axis (Categorical)", categorical_cols)
                    col_y = st.sidebar.selectbox("Select Y-axis (Numerical - Mean)", numeric_cols)
                
                    sns.barplot(x=col_x, y=col_y, data=df, ax=ax, errorbar='sd', palette='crest')
                    ax.set_title(f'Mean of {col_y} per {col_x}', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.warning("A Bar Chart (for mean comparison) requires one Categorical and one Numerical column.")

            elif visualization_type == "Box Plot (1 Numeric)":
                if numeric_cols:
                    col = st.sidebar.selectbox("Select Column (Numeric)", numeric_cols)
                    sns.boxplot(y=df[col].dropna(), ax=ax, color='#007ACC')
                    ax.set_title(f'Box Plot of {col} (Outlier Detection)', fontsize=14)
                    st.pyplot(fig)
                else:
                    st.warning("A Box Plot requires at least one numerical column.")

            elif visualization_type == "Violin Plot (1 Num vs Opt Cat)":
                if numeric_cols:
                    col_y = st.sidebar.selectbox("Select Numerical Column (Y-axis)", numeric_cols)
                    cat_options = ["None"] + categorical_cols
                    col_x = st.sidebar.selectbox("Select Optional Categorical Grouping (X-axis)", cat_options)
                    
                    if col_x != "None":
                        sns.violinplot(x=col_x, y=col_y, data=df, ax=ax, palette='husl')
                        ax.set_title(f'Distribution of {col_y} Grouped by {col_x}', fontsize=14)
                        plt.xticks(rotation=45, ha='right')
                    else:
                        sns.violinplot(y=col_y, data=df, ax=ax, color='#007ACC')
                        ax.set_title(f'Overall Distribution of {col_y}', fontsize=14)
                    st.pyplot(fig)
                else:
                    st.warning("A Violin Plot requires at least one Numerical column.")
            
            elif visualization_type == "Scatter Plot (2 Numeric)":
                if len(numeric_cols) >= 2:
                    col_x = st.sidebar.selectbox("Select X-axis (Numeric)", numeric_cols, index=0)
                    col_y = st.sidebar.selectbox("Select Y-axis (Numeric)", numeric_cols, index=min(1, len(numeric_cols) - 1))
                    
                    if col_x == col_y:
                        st.error("X and Y axes must be different columns.")
                    else:
                        sns.scatterplot(x=df[col_x], y=df[col_y], data=df, ax=ax, color='#007ACC')
                        ax.set_title(f'Relationship: {col_x} vs {col_y}', fontsize=14)
                        st.pyplot(fig)
                else:
                    st.warning("A Scatter Plot requires at least two numerical columns to explore correlation.")

            elif visualization_type == "Heatmap (Correlation)":
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr()
                    plt.figure(figsize=(10, 8)) 
                    # Use a subtle cmap for minimalist aesthetic
                    sns.heatmap(corr, annot=True, cmap='bone_r', fmt=".2f", linewidths=.5)
                    plt.title('Correlation Heatmap (Numerical Only)', fontsize=14)
                    st.pyplot()
                else:
                    st.warning("A Correlation Heatmap requires at least two numerical columns.")
                    
            elif visualization_type == "Pairplot (Multiple Numeric)":
                if len(numeric_cols) >= 2:
                    default_cols = numeric_cols[:min(5, len(numeric_cols))]
                    selected_cols = st.sidebar.multiselect("Select Numerical Columns (max 5 recommended)", numeric_cols, default=default_cols)
                    
                    if len(selected_cols) > 1:
                        st.info("Generating Pairplot (This may take a moment for large datasets)...")
                        pair_fig = sns.pairplot(df[selected_cols].dropna())
                        st.pyplot(pair_fig)
                    elif len(selected_cols) <= 1:
                        st.warning("Please select at least two numerical columns for the Pairplot.")
                else:
                    st.warning("A Pairplot requires at least two numerical columns.")

        except Exception as e:
            st.error(f"Error rendering plot. Please verify your column types and selections: {e}")
            
# 4. Missing Data Handling (Core Feature 4 - Robust Imputation)
elif menu_selection == "4. Missing Data Handling":
    st.header("4. Missing Data Handling")
    if df is None:
        st.warning("Please upload a dataset first in the '1. Dataset Upload' section.")
    else:
        numeric_cols, categorical_cols, _ = get_column_types(df)
        num_cols_with_na = [col for col in numeric_cols if df[col].isnull().any()]
        cat_cols_with_na = [col for col in categorical_cols if df[col].isnull().any()]
        total_missing = df.isnull().sum().sum()
        
        st.subheader("Missing Value Overview")
        if total_missing == 0:
            st.success("🎉 No missing values found in the current dataset! Data is clean.")
        else:
            # --- Visualizing Missing Data (Heatmap) ---
            st.subheader("Missing Data Map")
            st.info(f"Total missing values found: **{total_missing}**")
            fig, ax = plt.subplots(figsize=(12, 6))
            # Use 'binary' cmap: white for missing (True), black for present (False)
            sns.heatmap(df.isnull(), cbar=False, cmap='binary', ax=ax)
            ax.set_title("Missing Data Heatmap (Black Lines Indicate Missing Values)", fontsize=14)
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Data Cleaning Options")

            # Option 1: Drop Missing Rows
            with st.expander("🗑️ Drop All Rows with Missing Values"):
                rows_to_drop = df.isnull().any(axis=1).sum()
                st.info(f"This will permanently remove **{rows_to_drop}** rows that contain at least one missing value.")
                if st.button("Apply Drop Rows"):
                    rows_before = len(df)
                    df.dropna(how='any', inplace=True)
                    rows_after = len(df)
                    st.session_state['df'] = df
                    st.success(f"Successfully dropped {rows_before - rows_after} rows. Dataframe updated.")
                    st.info("Navigate to '2. Data Summary' to confirm the new row count and re-check missing values.")

            # --- Impute Values ---
            st.markdown("#### Imputation Strategies")

            # Numerical Imputation (Mean/Median)
            if num_cols_with_na:
                st.markdown("##### 1. Numerical Imputation")
                imputation_method_num = st.radio(
                    "Select method to fill missing numerical values:",
                    ["Fill with Mean", "Fill with Median"],
                    key="num_impute_method"
                )
                st.write("Columns to be imputed:", ', '.join(num_cols_with_na))

                if st.button("Apply Numerical Imputation"):
                    method_name = imputation_method_num.split()[-1]
                    for col in num_cols_with_na:
                        if method_name == "Mean":
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method_name == "Median":
                            df[col].fillna(df[col].median(), inplace=True)
                    st.session_state['df'] = df
                    st.success(f"Missing numerical values imputed using **{method_name}**.")
                    st.info("Navigate to '2. Data Summary' to confirm changes.")
            
            # Categorical Imputation (Mode)
            if cat_cols_with_na:
                st.markdown("##### 2. Categorical Imputation")
                st.info("Missing categorical data is best imputed with the **Mode (Most Frequent)**.")
                st.write("Columns to be imputed:", ', '.join(cat_cols_with_na))
                
                if st.button("Apply Categorical Imputation"):
                    for col in cat_cols_with_na:
                        # Use mode()[0] to handle potential multiple modes
                        df[col].fillna(df[col].mode()[0], inplace=True) 
                    st.session_state['df'] = df
                    st.success(f"Missing categorical values imputed using **Mode**.")
                    st.info("Navigate to '2. Data Summary' to confirm changes.")

# 5. Download Report (Core Feature 5)
elif menu_selection == "5. Download Report":
    st.header("5. Download Descriptive Report")
    if df is None:
        st.warning("Please upload a dataset first in the '1. Dataset Upload' section.")
    else:
        st.subheader("Report Preview")
        
        report_data_preview = df.describe(include='all').T
        st.dataframe(report_data_preview)
        
        # Generate the downloadable data
        report_data, filename = create_download_report(df)
        
        st.download_button(
            label="⬇️ Download Full Descriptive Report (CSV)",
            data=report_data,
            file_name=filename,
            mime='text/csv',
            help="Exports descriptive statistics for all columns in CSV format."
        )
        st.success("Report is ready for download!")

# 6. About Dashboard (Core Feature 6)
elif menu_selection == "6. About Dashboard":
    st.header("6. About Dashboard & Toolset")
    st.markdown("This dashboard demonstrates a core data analysis pipeline using powerful Python libraries. Each step utilizes a specific tool for maximum efficiency.")
    st.markdown("---")

    st.subheader("Key Technologies (APIs)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### NumPy")
        st.info("The foundational library for numerical computing in Python, enabling high-performance calculations for statistics like mean, median, and standard deviation.")
        
        st.markdown("#### Pandas")
        st.info("The primary tool for data handling, analysis, and cleaning, offering the DataFrame structure to manage, filter, and summarize complex datasets.")

        st.markdown("#### Matplotlib")
        st.info("The base plotting library that provides full control over chart creation, serving as the foundation for custom and advanced visualizations.")

    with col2:
        st.markdown("#### Seaborn")
        st.info("A high-level interface built on Matplotlib, used for drawing attractive and informative statistical graphics with beautiful, aesthetic default settings.")
        
        st.markdown("#### Streamlit")
        st.info("The framework used to instantly turn Python scripts into interactive, shareable web applications and dashboards with minimal engineering overhead.")
