# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:50:42 2023

@author: Akhilesh
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def get_clean_data(df):
    #df = pd.read_csv('LoanApprovalPrediction.csv')
    df = df.drop_duplicates()
    # copy
    df_clean = df.copy()
    
    # drop Loan_ID column
    df_clean = df_clean.drop('Loan_ID', axis=1)
    
    # convert NaN to 0 in 'Dependents' column
    df_clean['Dependents'] = df_clean['Dependents'].fillna(0)
    
    # convert NaN to 0 in 'LoanAmount' column
    df_clean['LoanAmount'] = df_clean['LoanAmount'].fillna(0)
    
    # convert NaN to 0 in 'Loan_Amount_Term' column
    df_clean['Loan_Amount_Term'] = df_clean['Loan_Amount_Term'].fillna(0)
    
    # convert NaN to 0 in 'Credit_History' column
    df_clean['Credit_History'] = df_clean['Credit_History'].fillna(0)
    return df_clean

def get_test_train_split(df_clean):
    ## OHE

    # initialize OneHotEncoder
    ohe = OneHotEncoder()

    # get list of categorical columns (not target column)
    ohe_columns = [col for col in df_clean.columns if df_clean[col].dtype == 'object' and col != 'Loan_Status']

    # run get_dummies on categorical columns
    df_clean = pd.get_dummies(df_clean, columns=ohe_columns)
    # split into input features, output label
    X = df_clean.drop('Loan_Status', axis=1)
    y = df_clean['Loan_Status']
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def build_logistic_regression(df_clean):
    
    X_train, X_test, y_train, y_test = get_test_train_split(df_clean)
    # LogisticRegression
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)
    
    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred_LR)
    cm_LR = confusion_matrix(y_test, y_pred_LR)
    return report, cm_LR, accuracy_score(y_test, y_pred_LR)


def build_decision_tree(df_clean):
    # DecisionTreeClassifier
    X_train, X_test, y_train, y_test = get_test_train_split(df_clean)
    model_DT = DecisionTreeClassifier()
    model_DT.fit(X_train, y_train)
    y_pred_DT = model_DT.predict(X_test)
    
    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred_DT)
    cm_DT = confusion_matrix(y_test, y_pred_DT)
    return report, cm_DT, accuracy_score(y_test, y_pred_DT)

def build_knn(df_clean):
    # DecisionTreeClassifier
    X_train, X_test, y_train, y_test = get_test_train_split(df_clean)
    model_KNN = KNeighborsClassifier(n_neighbors=5)
    model_KNN.fit(X_train, y_train)
    y_pred_KNN = model_KNN.predict(X_test)
    
    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred_KNN)
    cm_KNN = confusion_matrix(y_test, y_pred_KNN)
    return report, cm_KNN, accuracy_score(y_test, y_pred_KNN)

def build_svm(df_clean):
    X_train, X_test, y_train, y_test = get_test_train_split(df_clean)
    model_SVM = SVC()
    model_SVM.fit(X_train, y_train)
    y_pred_SVM = model_SVM.predict(X_test)
    
    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred_SVM)
    cm_SVM = confusion_matrix(y_test, y_pred_SVM)
    return report, cm_SVM, accuracy_score(y_test, y_pred_SVM)

def build_nb(df_clean):
    X_train, X_test, y_train, y_test = get_test_train_split(df_clean)
    model_NB = GaussianNB()
    model_NB.fit(X_train, y_train)
    y_pred_NB = model_NB.predict(X_test)
    # Classification Report and Confusion Matrix
    report = classification_report(y_test, y_pred_NB)
    cm_NB = confusion_matrix(y_test, y_pred_NB)
    return report, cm_NB, accuracy_score(y_test, y_pred_NB)
