import argparse
import pandas as pd
import numpy as np
import sqlalchemy

def read_args():
    print('In function read_args()\nParsing arguments')
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input", help="<input filename>")
    return argParser.parse_args()

def extract_data(file):
    print('In function extract_data()\nReading input file')
    return pd.read_csv(file)
    
def transform_data(df):   
    print('In function transform_data()')
    df['Dependents'].fillna('-',inplace=True)
    df['Loan_Amount_Term'].fillna('-',inplace=True)
    df['Credit_History'].fillna('-',inplace=True)
    return df

def load_data(df):
    db_url = "postgresql+psycopg2://shrestha:hunter2@db:5432/loanapp"
    engine = sqlalchemy.create_engine(db_url)
    print('In function load_data()\nWriting output file')
    #df.to_sql("outcomes", conn, if_exists="append", index=False)
    #df.to_sql("outcomes", conn, if_exists="replace", index=False)
    df.to_sql("loan_approvals", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        with open("load_data.sql") as file:
            query = sqlalchemy.sql.text(file.read())
            print(query)
            conn.execute(query)
    with engine.connect() as conn:
        with open("sql_queries.sql") as file:
            query = sqlalchemy.sql.text(file.read())
            print(query)
            conn.execute(query)


def print_data(data):
    print(data)


if __name__ == "__main__":
    print('In function main()')
    args = read_args()
    print("args=%s" %args)
    print("args.input=%s" %args.input)
    
    df = extract_data(args.input)
    print(len(df))
    print('Before processing data:')
    print_data(df)
    df = transform_data(df)
    print('After processing data:')
    print_data(df)
    load_data(df)
