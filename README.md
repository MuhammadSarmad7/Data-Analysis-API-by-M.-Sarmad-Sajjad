
📋 Data Analysis and Visualization Dashboard : Core Feature List

The dashboard provides a complete 6-step analytical pipeline accessible via an interactive sidebar:

1. Dataset Upload
Functionality: Allows the user to upload any dataset in .csv format.

2. Data Summary
Numerical Analysis: Provides descriptive statistics for all numerical columns, including the mean, median, standard deviation (std), count, minimum (min), maximum (max), and quartile values (25%, 50%, 75%).
Categorical Analysis: Provides statistics for all categorical columns, including the total count, number of unique values, the top (most frequent) value, and its frequency.

3. Data Visualization
Plot Selection: Offers 8 essential plots suitable for different analytical tasks.
Column Criteria: Each plot type enforces the selection of statistically suitable columns for the X and Y axes (e.g., Scatter Plot requires two numerical columns, Bar Chart requires one numerical and one categorical).


4. Missing Data Handling
Visual Analysis: Displays the total missing count and percentage per column. A Missing Data Heatmap is provided, where black lines indicate the location of missing values across the dataset.
Data Cleaning: Provides interactive options for applying changes to the working dataset:
Drop Rows: Removes entire rows that contain any missing data.
Numerical Imputation: Offers choice to fill missing values in numerical columns with either the Mean or Median.
Categorical Imputation: Fills missing values in categorical columns with the Mode (Most Frequent Value).

5. Download Summary
Report Generation: Creates a final report containing the full descriptive statistics (numerical and categorical) of the current working dataset (after any cleaning/imputation steps).
Export: Allows the user to download this complete report as a .csv file.

6. About Dashboard
Toolset Overview: Provides a summary of the core Python APIs used in the dashboard, highlighting the function of Streamlit, Pandas, NumPy, Matplotlib, and Seaborn.
