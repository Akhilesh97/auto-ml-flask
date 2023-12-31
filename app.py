# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:53:45 2023

@author: Akhilesh
"""

from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import os
import plotly.express as px
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from numbers import Number
import datetime
import ml_pipeline

app = Flask(__name__)

# Define a global variable to store the uploaded data
uploaded_data = None


def get_col_counts(x):
    numeric_counts = 0
    object_counts = 0
    date_types = 0

    for val in x:
        if isinstance(val, Number):
            numeric_counts+=1
        elif isinstance(val, str):
            object_counts+=1
        elif isinstance(val, datetime.date):
            date_types+=1

    return numeric_counts, object_counts, date_types

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'csv_file' not in request.files:
        return redirect(request.url)

    file = request.files['csv_file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)
        columns = list(data.columns)

        uploaded_data = data
        print(uploaded_data)
        data_ = data.iloc[:,1:]
        # Separate columns by data type
        numeric_columns = data_.select_dtypes(include=['number']).columns
        categorical_columns = data_.select_dtypes(exclude=['number']).columns

        # Task 2: Count null values for each variable
        null_counts = data.isnull().sum()
        
        # Task 3: Get duplicate values df
        # Find and display the number of duplicates based on all columns
        duplicate_counts = data.duplicated().sum()
        print(f'Number of duplicate rows: {duplicate_counts}')
        
        # Display the duplicate rows in a new DataFrame
        duplicates_df = data[data.duplicated(keep=False)]
        print('\nDuplicate Rows:')
        print(duplicates_df)
        
        # Task 4: Get inconsistent values df
        numeric_counts_ = []
        object_counts_ = []
        date_types_ = []
        inconsistencies_ = []
        for col in columns:
            numeric_counts, object_counts, date_types = get_col_counts(data[col])    
            inconsistency = 100-(max(numeric_counts, object_counts, date_types)/len(data))*100    
            numeric_counts_.append(numeric_counts)
            object_counts_.append(object_counts)
            date_types_.append(date_types)
            inconsistencies_.append(inconsistency)
        
        inconsistency_df = pd.DataFrame(columns=['column', 'numeric_vals', 'object_vals', 'date_vals', 'inconsistency_percentage'])
        inconsistency_df['column'] = columns
        inconsistency_df['numeric_vals'] = numeric_counts_
        inconsistency_df['object_vals'] = object_counts_
        inconsistency_df['date_vals'] = date_types_
        inconsistency_df['inconsistency_percentage'] = inconsistencies_
        
        # Task 4: Categorical distribution
        categorical_distributions = []
        for col in categorical_columns:
            plot = px.bar(data, x=col, title=f'Distribution of {col}')
            plot.update_layout(
                    autosize=False,
                    width=650,
                    height=500,
                )
            categorical_distributions.append({'plot': plot.to_json(), 'name': col})

        # Task 5: Histogram distribution for numeric columns
        histogram_plots = []
        for col in numeric_columns:
            plot = px.histogram(data, x=col, title=f'Histogram of {col}')
            plot.update_layout(
                    autosize=False,
                    width=650,
                    height=500,
                )
            histogram_plots.append({'plot': plot.to_json(), 'name': col})

        # Task 6: Correlation heatmap for numeric columns
        correlation_heatmap = go.Figure(data=go.Heatmap(
            z=data[numeric_columns].corr(),
            x=numeric_columns,
            y=numeric_columns,
            colorscale='Viridis'
        ))
        correlation_heatmap.update_layout(
                autosize=False,
                width=800,
                height=500,
            )

        # Task 7: Contingency plot for the last column as the target variable
        # Create a contingency table
        contingency_plots = []
        df = pd.DataFrame(data)
        
        # Create separate contingency plots for each variable
        variables_to_plot = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        
        for variable in variables_to_plot:
            contingency_table = pd.crosstab(df['Loan_Status'], df[variable])
        
            fig = px.imshow(contingency_table, labels=dict(x=variable, y='Loan Status', color='Count'),
                            x=contingency_table.columns,
                            y=contingency_table.index,
                            color_continuous_scale='YlGnBu',
                            title=f"Contingency Plot of Loan Status with {variable}")
            fig.update_layout(
                    autosize=False,
                    width=650,
                    height=500,
                )
            
            contingency_plots.append({'plot': fig.to_json(), 'name': variable})

        return render_template('results.html',
                               numeric_columns=numeric_columns,
                               categorical_columns=categorical_columns,
                               null_counts=null_counts,
                               duplicates_df = duplicates_df,
                               inconsistency_df = inconsistency_df,
                               categorical_distributions=categorical_distributions,
                               histogram_plots=histogram_plots,
                               correlation_heatmap=correlation_heatmap.to_json(),
                               contingency_plots=contingency_plots)
    
@app.route('/clean_data', methods=['POST'])
def clean_data():
    # Perform your cleaning operations on the uploaded data
    # For example, remove null values, check for data inconsistencies, etc.
    data = pd.read_csv("data/LoanApprovalPrediction.csv")
    cleaned_data = data.dropna()  # Replace with your cleaning logic    
    # Save the cleaned data to a CSV file in memory
    cleaned_data_csv = io.StringIO()
    cleaned_data.to_csv(cleaned_data_csv, index=False)
    cleaned_data_csv.seek(0)

    # Provide a download link for the user
    download_link = url_for('download_cleaned_data')

    response = {
        'success': True,
        'message': 'Cleaning operations completed successfully.',
        'download_link': download_link,
    }

    return json.dumps(response)

@app.route('/transform')
def transform():
    # Retrieve selected values from query parameters
    data = pd.read_csv("data/LoanApprovalPrediction.csv")
    columns = list(data.columns)
    duplicate_action = request.args.get('duplicate_action', '')
    null_action_radios = request.args.get('null_action_radios', '').split(',')
    incon_action_radios = request.args.get('incon_action_radios', '').split(',')
    msg = "<h3>Duplicate handling operations</h3>"
    if duplicate_action == "keep":
        msg+="Keeping duplicate columns<br>"
    else:
        msg+="Dropping duplicate columns<br>"
    
    msg+="<hr>"
    msg += "<h3>Null value handling operations</h3>"    
    for index, action in enumerate(null_action_radios):
        msg += "%d. %sing column - %s<br> "%(index+1, action, columns[index])
    
    msg+="<hr>"
    msg += "<h3>Inconsistency value handling operations performed</h3>"    
    for index, action in enumerate(incon_action_radios):
        msg += "%d. %sing column - %s<br> "%(index+1, action, columns[index])

    print(duplicate_action)
    print(null_action_radios)
    print(incon_action_radios)
    return render_template('transform.html', transformation_message=msg)


@app.route('/download_cleaned_data')
def download_cleaned_data():
    # Retrieve the cleaned data and provide it as a downloadable file
    # You should have the cleaned data available in your cleaning logic
    data = pd.read_csv("data/LoanApprovalPrediction.csv")
    cleaned_data = ml_pipeline.get_clean_data(data)  # Replace with your cleaning logic
    cleaned_data_csv = io.BytesIO()
    cleaned_data.to_csv(cleaned_data_csv, index=False)
    cleaned_data_csv.seek(0)
    print(cleaned_data.shape)
    # Retrieve the selected radio button values from the request
    # The names "duplicate_action" and "action_radio" should match the names you used in the HTML    

     # Save the cleaned data CSV file in the current directory
    cleaned_data_filename = 'data/cleaned_data.csv'
    cleaned_data_csv_path = os.path.join(os.getcwd(), cleaned_data_filename)
    with open(cleaned_data_csv_path, 'wb') as f:
        f.write(cleaned_data_csv.getvalue())

    return send_file(cleaned_data_csv_path,
                     as_attachment=True,
                     download_name=cleaned_data_filename,
                     mimetype='text/csv')
def query_table_from_database(table_name):
    # Implement your logic to query the specified table from the database
    # and return the data as a DataFrame.
    # Replace this with your actual database query logic.
    return pd.DataFrame()

@app.route('/data_modeling',methods=['GET', 'POST'])
def data_modeling():
    import re
    import etl_pipeline
    def extract_table_names(sql_content):
        table_names = set()
    
        # Use regular expression to find CREATE TABLE statements
        create_table_regex = re.compile(r'CREATE TABLE\s+"?(\w+)"?\s*\(', re.IGNORECASE)
    
        matches = create_table_regex.findall(sql_content)
        table_names.update(matches)
    
        return table_names
    
    # Replace this variable with the path to your SQL file
    sql_file_path = 'db/init.sql'
    
    # Read the SQL file
    with open(sql_file_path, 'r') as sql_file:
        sql_content = sql_file.read()
    
    # Extract table names from the SQL file
    table_names = extract_table_names(sql_content)
    if etl_pipeline.table_check(table_names):
        msg1 = "Table already loaded"
        msg2 = "Init tables already created"
        msg3 = "Diemension and Fact tables already loaded"
    else:
        msg1 = etl_pipeline.load_main_table()        
        msg2 = etl_pipeline.create_init_tables()
        msg3 = etl_pipeline.load_data()
    
    print(table_names, msg1, msg2, msg3)
    # Handle the selected table from the dropdown
    selected_table = request.form.get('selected_table')
    if selected_table == None:
        selected_table = "dim_gender"
    queried_table = etl_pipeline.query_table(selected_table)
    print(queried_table)
    return render_template('data_modeling.html', msg1=msg1, msg2=msg2, msg3=msg3, table_names = table_names,selected_table=selected_table, queried_table=queried_table)


@app.route('/download_table/<table_name>')
def download_table(table_name):
    # Assume queried_table is a DataFrame containing the queried data
    queried_table = query_table_from_database(table_name)

    if queried_table is not None:
        # Save the queried table CSV file in the current directory
        queried_table_csv_filename = f'data/{table_name}_queried_table.csv'
        queried_table_csv_path = os.path.join(os.getcwd(), queried_table_csv_filename)
        queried_table.to_csv(queried_table_csv_path, index=False)

        return send_file(queried_table_csv_path,
                         as_attachment=True,
                         download_name=queried_table_csv_filename,
                         mimetype='text/csv')

    return "Error: Table not found or unable to query."

@app.route('/modelling', methods=['GET', 'POST'])
def modelling():
    
    data = pd.read_csv("data/cleaned_data.csv")
    
    if data is None:
        # Handle the case where data is not available
        return "Data not available. Please upload data first."
    if request.method == 'POST':
        # Get selected features and model choice from the form
        selected_features = request.form.getlist('features[]')
        print(selected_features)
        selected_model = request.form['model']
        
        df_clean = data[selected_features]
        df_clean["Loan_Status"] = data["Loan_Status"]
                
        # Initialize the selected model
        if selected_model == 'LR':
            report, cm, accuracy = ml_pipeline.build_logistic_regression(df_clean)
        elif selected_model == 'DT':
            report, cm, accuracy = ml_pipeline.build_decision_tree(df_clean)
        elif selected_model == "KNN":
            report, cm, accuracy = ml_pipeline.build_knn(df_clean)
        elif selected_model == "NB":
            report, cm, accuracy = ml_pipeline.build_nb(df_clean)
        elif selected_model == "SVM":
            report, cm, accuracy = ml_pipeline.build_svm(df_clean)
        cm_json = json.dumps(cm.tolist())

        return render_template('metrics.html', accuracy=accuracy, classification_report=report, cm=cm_json)

    # Get the list of available features
    available_features = data.columns.tolist()

    return render_template('modelling.html', features=available_features)    


if __name__ == '__main__':
    app.run(debug=True)
