# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:45:51 2023

@author: Akhilesh
"""

import pandas as pd
from google.cloud.sql.connector import Connector
import sqlalchemy
import os

df = pd.read_csv("data/LoanApprovalPrediction.csv")
df['Dependents'].fillna('-',inplace=True)
df['Loan_Amount_Term'].fillna('-',inplace=True)
df['Credit_History'].fillna('-',inplace=True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "dazzling-tensor-405719-b0b850808aff.json"
INSTANCE_CONNECTION_NAME = "dazzling-tensor-405719:us-central1:auto-ml" # i.e demo-project:us-central1:demo-instance
print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
DB_USER = "postgres"
DB_PASS = "}gL<t,[bmnSzF-s:"
DB_NAME = "loan-app"
# function to return the database connection object

connector = Connector()

def getconn():
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn

# create connection pool with 'creator' argument to our connection object function
pool = sqlalchemy.create_engine(
    "postgresql+pg8000://", 
    creator=getconn,
)
def query_table(table_name):
    with pool.connect() as db_conn:
        # query and fetch ratings table
        results = db_conn.execute(sqlalchemy.text("SELECT * FROM %s;"%table_name)).fetchall()
        print(len(results))
            
    return pd.DataFrame(results)

def table_check(table_names):
    for table in table_names:
        df = query_table(table)
        if len(df)>0:
            return True
        else:
            return False
def load_main_table():
    df.to_sql("loan_approvals", pool, if_exists="replace", index=False)
    return "Initial Cleaned CSV Loaded Successfully"

def create_init_tables():
    with pool.connect() as db_conn:
      # create ratings table in our sandwiches database
        with open("db/init.sql") as file:
            query = sqlalchemy.sql.text(file.read())        
            db_conn.execute(query)
            db_conn.commit()
    return "Init tables created successfully"

def load_data():
    with pool.connect() as db_conn:
        with open("db/load_data.sql") as file:
            query = sqlalchemy.sql.text(file.read())        
            db_conn.execute(query)
            db_conn.commit()
            
    return "Data Loaded Successfully"