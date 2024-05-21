import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
df = pd.read_csv('salaries.csv')

# Streamlit app
st.set_page_config(page_title="Data Science Salary Analysis and Prediction", layout="wide")
st.title('Data Science Salary Analysis and Prediction Dashboard')

# Sidebar for navigation
st.sidebar.header('Navigation')
section = st.sidebar.radio('Go to:', ['Data Overview', 'Visualizations', 'Model Training', 'Model Evaluation'])

# Data Overview section
if section == 'Data Overview':
    st.header('Dataset Overview')
    st.write(df.head())

    st.subheader('Data Statistics')
    st.write(df.describe())

    st.subheader('Unique Values')
    st.write('Experience Levels:', df['experience_level'].unique())
    st.write('Employment Types:', df['employment_type'].unique())
    st.write('Job Titles:', df['job_title'].unique())

# Visualization section
elif section == 'Visualizations':
    st.header('Data Visualizations')
    
    plot_type = st.selectbox('Select Plot Type:', ['Seaborn Histogram', 'Seaborn Barplot', 'Plotly Histogram', 'Plotly Boxplot', 'Plotly Scatter'])
    
    if plot_type == 'Seaborn Histogram':
        st.subheader('Salary Distribution')
        fig, ax = plt.subplots()
        sns.histplot(df['salary_in_usd'], bins=20, kde=True, ax=ax)
        ax.set_xlabel('Salary (USD)')
        ax.set_ylabel('Frequency')
        ax.set_title('Salary Distribution')
        ax.grid()
        st.pyplot(fig)

    elif plot_type == 'Seaborn Barplot':
        st.subheader('Average Salary by Experience Level')
        fig, ax = plt.subplots()
        sns.barplot(x='experience_level', y='salary_in_usd', data=df, ax=ax)
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Salary (USD)')
        ax.set_title('Average Salary by Experience Level')
        st.pyplot(fig)

    elif plot_type == 'Plotly Histogram':
        st.subheader('Salary Distribution')
        fig = px.histogram(df, x="salary_in_usd", nbins=20, title='Salary Distribution')
        fig.update_layout(xaxis_title='Salary (USD)', yaxis_title='Frequency')
        st.plotly_chart(fig)

    elif plot_type == 'Plotly Boxplot':
        st.subheader('Salary Distribution Across Experience Levels')
        fig = px.box(df, x="experience_level", y="salary_in_usd", title='Salary Distribution Across Experience Levels')
        fig.update_layout(xaxis_title='Experience Level', yaxis_title='Salary (USD)')
        st.plotly_chart(fig)

    elif plot_type == 'Plotly Scatter':
        st.subheader('Salary vs. Years of Experience')
        fig = px.scatter(df, x="work_year", y="salary_in_usd", color="experience_level", title='Salary vs. Years of Experience')
        fig.update_layout(xaxis_title='Years of Experience', yaxis_title='Salary (USD)')
        st.plotly_chart(fig)

# Model Training section
elif section == 'Model Training':
    st.header('Model Training')
    
    df = pd.get_dummies(df, columns=['experience_level', 'employment_type'])
    X = df.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)  # Features
    y = df['salary_in_usd']  # Target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_encoded = pd.get_dummies(X_train)
    X_test_encoded = pd.get_dummies(X_test)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    st.subheader('Select Model for Training')
    model_choice = st.selectbox('Choose a model:', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Decision Tree', 'Random Forest'])

    if st.button('Train Model'):
        if model_choice == 'Linear Regression':
            model = LinearRegression()
        elif model_choice == 'Ridge Regression':
            model = Ridge()
        elif model_choice == 'Lasso Regression':
            model = Lasso()
        elif model_choice == 'Decision Tree':
            model = DecisionTreeRegressor()
        elif model_choice == 'Random Forest':
            model = RandomForestRegressor()
        
        model.fit(X_train_scaled, y_train)
        st.success(f'{model_choice} model trained successfully!')

        # Save the model
        filename = 'salary_prediction_model.sav'
        pickle.dump(model, open(filename, 'wb'))
        st.success(f'Model saved as {filename}')

        # Display model metrics
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        st.write(f"Train Mean Squared Error: {mean_squared_error(y_train, y_pred_train)}")
        st.write(f"Train R-squared: {r2_score(y_train, y_pred_train)}")
        st.write(f"Test Mean Squared Error: {mean_squared_error(y_test, y_pred_test)}")
        st.write(f"Test R-squared: {r2_score(y_test, y_pred_test)}")

# Model Evaluation section
elif section == 'Model Evaluation':
    st.header('Model Evaluation')

    # Load the model
    try:
        loaded_model = pickle.load(open("salary_prediction_model.sav", 'rb'))

        # Ensure the preprocessing steps are applied again
        df = pd.get_dummies(df, columns=['experience_level', 'employment_type'])
        X = df.drop(['salary', 'salary_currency', 'salary_in_usd'], axis=1)
        y = df['salary_in_usd']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_encoded = pd.get_dummies(X_train)
        X_test_encoded = pd.get_dummies(X_test)
        X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)

        y_pred = loaded_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

        st.subheader('Actual vs. Predicted Salary')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel('Actual Salary (USD)', color='red')
        ax.set_ylabel('Predicted Salary (USD)', color='green')
        ax.set_title('Actual vs. Predicted Salary')
        st.pyplot(fig)
    except FileNotFoundError:
        st.error('Model file not found. Please train the model first.')
