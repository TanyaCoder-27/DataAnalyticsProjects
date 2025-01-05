import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your actual path)
url = 'Top 2000 Companies Financial Data 2024.csv'

try:
    data = pd.read_csv(url)
except FileNotFoundError:
    print(f"Error: The file at {url} was not found.")
    data = pd.DataFrame()  # Create an empty DataFrame to avoid further errors
except pd.errors.EmptyDataError:
    print(f"Error: The file at {url} is empty.")
    data = pd.DataFrame()
except pd.errors.ParserError:
    print(f"Error: The file at {url} could not be parsed.")
    data = pd.DataFrame()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    data = pd.DataFrame()

if not data.empty:
    # Display the first few rows of the dataset
    print(data.head())

    # Data Summary
    print(data.info())
    print(data.describe())

    # Data Cleaning (handling missing values, converting data types)
    def convert_to_billions(value):
        value = value.replace(',', '')  # Remove commas
        if 'B' in value:
            return float(value.replace('$', '').replace(' B', ''))
        elif 'M' in value:
            return float(value.replace('$', '').replace(' M', '')) / 1000
        else:
            return float(value.replace('$', ''))

    data['Sales'] = data['Sales'].apply(convert_to_billions)
    data['Profit'] = data['Profit'].apply(convert_to_billions)
    data['Assets'] = data['Assets'].apply(convert_to_billions)
    data['Market Value'] = data['Market Value'].apply(convert_to_billions)

    # Handling any missing values
    data = data.dropna()

    # Ensure numeric columns are of correct type
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
    data['Profit'] = pd.to_numeric(data['Profit'], errors='coerce')
    data['Assets'] = pd.to_numeric(data['Assets'], errors='coerce')
    data['Market Value'] = pd.to_numeric(data['Market Value'], errors='coerce')

    # Drop any rows with NaN values after conversion
    data = data.dropna()

    # Statistical Analysis
    # Correlation Matrix
    corr_matrix = data[['Sales', 'Profit', 'Assets', 'Market Value']].corr()
    print(corr_matrix)

    # Visualize the Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Distribution of Key Metrics (Sales, Profit, Assets, Market Value)
    metrics = ['Sales', 'Profit', 'Assets', 'Market Value']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[metric], kde=True, bins=30)
        plt.title(f'Distribution of {metric}')
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        plt.show()

    # Top 10 Companies by Market Value
    top_10_companies = data.nlargest(10, 'Market Value')
    print(top_10_companies[['Name', 'Market Value']])

    # Visualization of Top 10 Companies by Market Value
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Market Value', y='Name', data=top_10_companies)
    plt.title('Top 10 Companies by Market Value')
    plt.xlabel('Market Value')
    plt.ylabel('Company Name')
    plt.show()

    # Distribution by Country
    country_distribution = data['Country'].value_counts()
    print(country_distribution)

    # Visualization of Company Distribution by Country
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Country', data=data, order=country_distribution.index)
    plt.title('Company Distribution by Country')
    plt.xlabel('Number of Companies')
    plt.ylabel('Country')
    plt.show()

    # Mean Sales by Country
    country_sales = data.groupby('Country')['Sales'].mean().sort_values(ascending=False)
    print(country_sales)

    # Visualization of Mean Sales by Country
    plt.figure(figsize=(14, 10))
    country_sales.plot(kind='bar')
    plt.title('Mean Sales by Country')
    plt.xlabel('Country')
    plt.ylabel('Mean Sales')
    plt.show()

    # Total Assets vs. Market Value
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Assets', y='Market Value', data=data)
    plt.title('Total Assets vs. Market Value')
    plt.xlabel('Total Assets')
    plt.ylabel('Market Value')
    plt.show()
else:
    print("No data to display or summarize.")