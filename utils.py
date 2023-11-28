import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pptx
import streamlit as st
from plotly.offline import plot
import time
from PyPDF2 import PdfReader
from datetime import datetime
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
import whisper
from pytube import YouTube
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from fpdf import FPDF
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
import tempfile
import base64
import config
import warnings
from sklearn.preprocessing import OrdinalEncoder
import plotly.graph_objs as go
from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import plotly.io as pio
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.ensemble import IsolationForest
import scipy.stats as stats
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from scipy.stats import norm
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import plotly.express as px
import webbrowser
from matplotlib import style
from src.InfraredProduct.exception import CustomException
import sys
from sklearn.covariance import EllipticEnvelope
from statsmodels.tsa.arima_model import ARMA, ARMAResults, ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from tensorflow import keras
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from tensorflow.keras.layers import Dropout, RepeatVector, TimeDistributed
import plotly.offline
import graphviz
import plost

# Function for KNN-based anomaly detection
def apply_anomaly_detection_KNN(data, n_neighbors=5, contamination=0.05):
    data_with_anomalies = data.copy()

    # Fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(data)

    # Calculate distances to the k-nearest neighbors for each data point
    distances, _ = knn_model.kneighbors(data)

    # Calculate an anomaly score based on distances
    anomaly_scores = distances[:, -1]

    # Determine anomalies based on a contamination threshold
    anomaly_threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    data_with_anomalies['Anomaly_KNN'] = np.where(anomaly_scores > anomaly_threshold, 1, 0)

    return data_with_anomalies
def apply_anomaly_detection_PCA(data, n_components=None, threshold=0.099):
    """
    Apply PCA for anomaly detection.

    Parameters:
    - data: DataFrame
        The input data for anomaly detection.
    - n_components: int or None, optional (default=None)
        The number of principal components to retain. If None, it uses all components.
    - threshold: float, optional (default=0.95)
        The threshold for cumulative explained variance to retain. Ignored if n_components is specified.

    Returns:
    - data_with_anomalies: DataFrame
        The input data with an additional column indicating whether each data point is an anomaly (1) or not (0).
    """

    # Copy the original data to avoid modifying the original dataframe
    data_with_anomalies = data.copy()

    # Perform PCA
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    # Reconstruct the data from reduced components
    data_reconstructed = pca.inverse_transform(data_pca)

    # Calculate the reconstruction error for each data point
    reconstruction_errors = np.sum(np.square(data - data_reconstructed), axis=1)

    # Determine anomalies based on the reconstruction error
    anomaly_threshold = np.percentile(reconstruction_errors, 100 * (1 - threshold))
    data_with_anomalies['Anomaly_PCA'] = np.where(reconstruction_errors > anomaly_threshold, 1, 0)

    return data_with_anomalies
def apply_anomaly_detection_dbscan(data, eps, min_samples):
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Anomaly_DBSCAN'] = dbscan.fit_predict(standardized_data)

    # Anomalies are labeled as -1, convert to 1 for visualization
    data['Anomaly_DBSCAN'] = (data['Anomaly_DBSCAN'] == -1).astype(int)

    return data

def apply_anomaly_detection_kmeans(data, num_clusters):
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(standardized_data)

    # Calculate the distance of each point to its cluster center
    distances = np.min(kmeans.transform(standardized_data), axis=1)
    threshold = np.percentile(distances, 95)  # Set a threshold for anomaly detection

    # Identify anomalies and assign 1 for anomalies, 0 for non-anomalies
    data['Anomaly_KMeans'] = (distances > threshold).astype(int)

    return data

def save_plot_as_html(fig, filename):
    fig.write_html(filename)

def apply_anomaly_detection_EllipticEnvelope(data):
    # Standardize the data
    # scaler = StandardScaler()
    # standardized_data = scaler.fit_transform(data)

    # Apply EllipticEnvelope
    ee = EllipticEnvelope(contamination=0.1)  # Set the contamination parameter as needed
    ee.fit(data)

    y_pred = ee.predict(data)
    data['Anomaly_EllipticEnvelope'] = np.where(y_pred == -1, 1, 0)

    # anomalies = ee.fit_predict(standardized_data)

    


    # Add anomaly column to the original DataFrame
    # data['Anomaly_EllipticEnvelope'] = anomalies

    return data
# starting-functions for EO-PO workflows
def categorize_material(Material):
    if pd.isnull(Material):
        return "Non Coded Material"
    else:
        return "Coded"
    

def get_dataframes_EO_PO(df1, df2, df3):
    tab1=df1[['Purchasing Document', 'Item', 'Deletion Indicator','Short Text', 'Material','Plant','Storage Location',
                    'Material Group','Purchasing Info Rec.','Vendor Material Number','Order Quantity','Order Unit','Order Price Unit',
                    'Net Order Price','Price Unit','Net Order Value','GR Processing Time','Tax code','Item Category','Distribut. indicator',
                    'Partial Invoice','Goods Receipt','Invoice Receipt','GR-Based Inv. Verif.','Outline Agreement','Reconciliation Date',
                    'Base Unit of Measure','Shipping Instr.','Planned Deliv. Time','Incoterms','Incoterms (Part 2)',
                    'Net order value','Shipping type','Purchase Requisition','Item of Requisition','Requisitioner','Order priority',
                    'Delivery Priority','Document Date']]
    tab2=df2[['Company Code','Purchasing Document','Purch. Doc. Category','Purchasing Doc. Type','Deletion Indicator','Created on',
                'Created by','Vendor','Terms of Payment','Payment in','Discount Percent 1',
                'Discount Percent 2','Purch. Organization','Purchasing Group','Exchange Rate','Exchange Rate Fixed',
                'Document Date','Supplying Vendor','Outline Agreement','Release group','Release Strategy',
                'Release indicator','Release status','Shipping type',]]
    tab3=df3[['Vendor','City','Name 1','Name 1 (#1)','Personnel No.','DME Indicator','ISR Number','Corporate Group',
                    'Alternative payee','Payee in document','Trading Partner','Block function','Date of birth',
                    'Credit information number','Last ext.review','Actual QM system','Reference acct group','Factory calendar',
                    'Transportation zone','Accts for alt. payee','Tax base','QM system valid to','Tax office responsible',
                    'DUNS+4 (the last 4 digit)','Name 1 (#2)','Trade License No.','Trade License Valid From Date',
                    'Trade License Valid To Date','Place of Issue','Portal User','CDC User','VAT Exempted','Date','Date (#1)',
                  'Date (#2)','Jaggaer ID']]
    return tab1, tab2, tab3

def join_dataframes_EO_PO(tab1, tab2, tab3):
    join1 = pd.merge(tab1, tab2, on='Purchasing Document', how='left')
    # ... rest of the joins and filters ...
    filter1= join1[join1['Deletion Indicator_x'].isnull()]
    filter2=filter1[(filter1['Purch. Doc. Category'] == 'F') & (~filter1['Vendor'].isnull())]
    join2 = pd.merge(filter2, tab3, on='Vendor', how='left')
    join2['Coded or NonCoded'] = join2['Material'].apply(categorize_material)
    join2['Order Quantity'] = join2['Order Quantity'].astype(float)
    join2['Net Order Price'] = join2['Net Order Price'].astype(float)
    join2['Net Order Value'] = join2['Net Order Value'].astype(float)

    filter3= join2[join2['Net Order Price'] > 0.0]
    join2['Net Order Price INR'] = join2['Net Order Price'] * join2['Exchange Rate']
    join2['Net Order Value INR'] = join2['Net Order Price'] * join2['Order Quantity']
    
    return join2

def calculate_price_variance_EO_PO(join2):
    # calculations for final1, final2, final3, final4, final5, final6
    tab4=join2[['Short Text','Material','Order Unit','Net Order Price INR','Net Order Value INR','Order Quantity','Purchasing Document']]

    tab4['Material'].fillna('Unknown', inplace=True)
    group1 = tab4.groupby(['Short Text', 'Material', 'Order Unit']).agg({
                    'Net Order Price INR': ['min', 'max', 'nunique', 'mean'],
                    'Net Order Value INR': 'sum',
                    'Order Quantity': 'sum',
                    'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())]  # Calculate unique count and concatenate
                }).reset_index()
    unique_values = group1[['Short Text', 'Material', 'Order Unit']].drop_duplicates()
    group1 = unique_values.merge(group1, on=['Short Text', 'Material', 'Order Unit'], how='left')
    # Rename columns for clarity
    group1.columns = ['Short Text', 'Material', 'Order Unit',
                                'Min Net Order Price INR', 'Max Net Order Price INR',
                                'Unique Count Net Order Price INR', 'Mean Net Order Price INR',
                                'Sum Net Order Value INR', 'Sum Order Quantity','Unique Count Purchasing Document',
                                'Unique Concatenate Purchasing Document']
    group1['Material'].replace('Unknown', None, inplace=True)

    final1=group1[group1['Unique Count Net Order Price INR'] > 1.0]
    final1['Percentage of MaxPrice wiht Diff'] = round((final1['Max Net Order Price INR'] - final1['Min Net Order Price INR']) / final1['Max Net Order Price INR'] * 100, 2)
    final1['Percent to Avg Price With Diff'] = round((final1['Mean Net Order Price INR'] - final1['Min Net Order Price INR']) / final1['Mean Net Order Price INR'] * 100, 2)
    tab5=join2[['Short Text','Material','Order Unit','Created on','Vendor','Name 1',
                'Net Order Price INR','Net Order Value INR','Order Quantity','Purchasing Document']]

    # Group by columns 'Short Text', 'Material', and 'Order Unit'
    group2 = tab5.groupby(['Short Text', 'Material', 'Order Unit','Created on','Vendor','Name 1']).agg({
        'Net Order Price INR': ['min', 'max', 'nunique', 'mean'],
        'Net Order Value INR':'sum',
        'Order Quantity': 'sum',
     'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())]  # Calculate unique count and concatenate
     }).reset_index()

     # Rename columns for clarity
    group2.columns = ['Short Text', 'Material', 'Order Unit','Created on','Vendor','Name 1',
                                'Min Net Order Price INR', 'Max Net Order Price INR',
                                'Unique Count Net Order Price INR', 'Mean Net Order Price INR',
                                'Sum Net Order Value INR', 'Sum Order Quantity','Unique Count Purchasing Document',
                                'Unique Concatenate Purchasing Document']

     # Replace 'Unknown' with None (optional, if you prefer None for null values)
     # Replace 'Unknown' with None
    group2['Material'].replace('Unknown', None, inplace=True)
    group2['Name 1'].replace('Unknown2', None, inplace=True)
    group2['Vendor'].replace('Unknown3', None, inplace=True)         



    final2=group2[group2['Unique Count Net Order Price INR'] > 1.0]
    final2['Percentage to MaxPrice wiht Diff'] = round((final2['Max Net Order Price INR'] - final2['Min Net Order Price INR']) / final2['Max Net Order Price INR'] * 100, 2)
    final2['Percent to Avg Price With Diff'] = round(((final2['Mean Net Order Price INR'] - final2['Min Net Order Price INR']) / final2['Mean Net Order Price INR']) * 100, 2)


    tab6=join2[['Short Text','Material','Plant','Order Unit','Created on','Vendor','Name 1',
                'Net Order Price INR','Net Order Value INR','Order Quantity','Purchasing Document']]
    tab6['Material'].fillna('Unknown', inplace=True)
    tab6['Name 1'].fillna('Unknown2', inplace=True)
    tab6['Vendor'].fillna('Unknown3', inplace=True)




    group3 = tab6.groupby(['Short Text', 'Material','Plant', 'Order Unit','Created on','Vendor','Name 1']).agg({
       'Net Order Price INR': ['min', 'max', 'nunique', 'mean'],
      'Net Order Value INR':'sum',
      'Order Quantity': 'sum',
      'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())]  # Calculate unique count and concatenate
    }).reset_index()

      # Rename columns for clarity
    group3.columns = ['Short Text', 'Material','Plant','Order Unit','Created on','Vendor','Name 1',
                                'Min Net Order Price INR', 'Max Net Order Price INR',
                                'Unique Count Net Order Price INR', 'Mean Net Order Price INR',
                                'Sum Net Order Value INR', 'Sum Order Quantity','Unique Count Purchasing Document',
                                'Unique Concatenate Purchasing Document']

     # Replace 'Unknown' with None
    group3['Material'].replace('Unknown', None, inplace=True)
    group3['Name 1'].replace('Unknown2', None, inplace=True)
    group3['Vendor'].replace('Unknown3', None, inplace=True)

    final3=group3[group3['Unique Count Net Order Price INR'] > 1.0]
    final3['Percent to max price with diff'] = round(((final3['Max Net Order Price INR'] - final3['Min Net Order Price INR']) / final3['Max Net Order Price INR']) * 100, 2)
    final3['Perecnt to Avergae Price with diff'] = round(((final3['Mean Net Order Price INR'] - final3['Min Net Order Price INR']) / final3['Mean Net Order Price INR']) * 100, 2)

    tab7=join2[['Short Text','Material','Plant','Order Unit','Created on','Vendor','Name 1',
                'Net Order Price INR','Net Order Value INR','Order Quantity','Purchasing Document']]
    tab7['Material'].fillna('Unknown', inplace=True)

    group4 = tab7.groupby(['Short Text', 'Material','Plant', 'Order Unit','Created on']).agg({
                    'Net Order Price INR': ['min', 'max', 'nunique', 'mean'],
                    'Net Order Value INR':'sum',
                    'Order Quantity': 'sum',
                'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())], 
                    'Vendor': ['nunique', lambda x: ', '.join(x.astype(str).unique())],
                    'Name 1': [lambda x: ', '.join(x.astype(str).unique())],
    }).reset_index()

    # Rename columns for clarity
    group4.columns = ['Short Text', 'Material','Plant','Order Unit','Created on',
                                'Min Net Order Price INR', 'Max Net Order Price INR',
                                'Unique Count Net Order Price INR', 'Mean Net Order Price INR',
                                'Sum Net Order Value INR', 'Sum Order Quantity','Unique Count Purchasing Document',
                                'Unique Concatenate Purchasing Document','Unique Count Vendor','Unique Concatenate Vendor',
                                'Unique Concatenate Name1',]
    group4['Material'].replace('Unknown', None, inplace=True)
    final4 = group4[(group4['Unique Count Net Order Price INR'] > 1) & (group4['Unique Count Vendor'] > 1)]
    final4['Percentage to MaxPrice with Diff'] = round((final4['Max Net Order Price INR'] - final4['Min Net Order Price INR']) / final4['Max Net Order Price INR'] * 100, 2)
    final4['Percent to Avg Price With Diff'] = round((final4['Mean Net Order Price INR'] - final4['Min Net Order Price INR']) / final4['Mean Net Order Price INR'] * 100, 2)


    tab8=join2[['Short Text','Material','Order Unit','Created on','Vendor','Name 1',
                'Net Order Price INR','Net Order Value INR','Order Quantity','Purchasing Document']]
    # Fill null values 
    tab8['Material'].fillna('Unknown', inplace=True)
    tab8['Name 1'].fillna('Unknown2', inplace=True)
    tab8['Vendor'].fillna('Unknown3', inplace=True)

    group5 = tab8.groupby(['Short Text', 'Material', 'Order Unit','Created on','Vendor','Name 1']).agg({
        'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())], 
    }).reset_index()

    # Rename columns for clarity
    group5.columns = ['Short Text', 'Material','Order Unit','Created on','Vendor','Name 1',
                                'Unique Count Purchasing Document','Unique Concatenate Purchasing Document']
    
    group5['Material'].replace('Unknown', None, inplace=True)
    group5['Name 1'].replace('Unknown2', None, inplace=True)
    group5['Vendor'].replace('Unknown3', None, inplace=True)
    final5 = group5[(group5['Unique Count Purchasing Document'] > 1)]
    tab9=join2[['Short Text','Material','Order Unit','Created on','Vendor','Name 1','Release Strategy','Purchasing Document']]

    tab9['Material'].fillna('Unknown', inplace=True)
    tab9['Name 1'].fillna('Unknown2', inplace=True)
    tab9['Vendor'].fillna('Unknown3', inplace=True)

    group6 = tab9.groupby(['Short Text', 'Material', 'Order Unit','Created on','Vendor','Name 1']).agg({
                'Purchasing Document': ['nunique', lambda x: ', '.join(x.astype(str).unique())], 
                    'Release Strategy': ['nunique', lambda x: ', '.join(x.astype(str).unique())], 
                    
                }).reset_index()
    # Rename columns for clarity
    group6.columns = ['Short Text', 'Material','Order Unit','Created on','Vendor','Name 1',
                                'Unique Count Purchasing Document','Unique Concatenate Purchasing Document',
                                'Unique Count Release Strategy','Unique Concatenate Release Strategy']
                # Replace 'Unknown' with None
    group6['Material'].replace('Unknown', None, inplace=True)
    group6['Name 1'].replace('Unknown2', None, inplace=True)
    group6['Vendor'].replace('Unknown3', None, inplace=True)

    final6= group6[(group6['Unique Count Purchasing Document'] > 1)]


    return final1, final2, final3, final4, final5, final6

def write_to_excel_EO_PO(final1, final2, final3, final4, final5, final6):
    # Create Excel files
    excel_writer1 = pd.ExcelWriter('FINAL1.xlsx', engine='xlsxwriter')
    final1.to_excel(excel_writer1, sheet_name='Material-wise Price Variance', index=False)
    final2.to_excel(excel_writer1, sheet_name='Date-wise Price Variance', index=False)
    final3.to_excel(excel_writer1, sheet_name='-Date-Plant Price Variance', index=False)
    final4.to_excel(excel_writer1, sheet_name='Plant-Vendor Price Variance', index=False)
    excel_writer1.book.close()

    excel_writer2 = pd.ExcelWriter('FINAL2.xlsx', engine='xlsxwriter')
    final5.to_excel(excel_writer2, sheet_name='Matl+Ven+Day Multi POs', index=False)
    final6.to_excel(excel_writer2, sheet_name='Matl+Ven+Day+Rel Sta. POs', index=False)
    excel_writer2.book.close()

    # Display final1 and final2 on Streamlit UI
    st.header("Final1 Data")
    st.write(final1)

    st.header("Final2 Data")
    st.write(final2)

    # Add download buttons for CSV and Excel files
    st.subheader("Download Data")
    st.write("Download FINAL1:")
    csv_data1 = final1.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data1, "final1.csv", "text/csv")

    excel_data1 = open('FINAL1.xlsx', 'rb').read()
    st.download_button("Download Excel", excel_data1, "FINAL1.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.write("Download FINAL2:")
    csv_data2 = final2.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv_data2, "final2.csv", "text/csv")

    excel_data2 = open('FINAL2.xlsx', 'rb').read()
    st.download_button("Download Excel", excel_data2, "FINAL2.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ending-functions for EO-PO workflows



# starting function for Node 679


def process_data_node_679(df, category_filter, vendor_filter):
    """Process data based on category and vendor filters."""
    # Filter data based on category and non-null Material
    filtered_df = df[(df['Purch. Doc. Category'] == category_filter) & (df['Material'].notna())]

    # Convert data types
    filtered_df['Order Quantity'] = filtered_df['Order Quantity'].astype(float)
    filtered_df['Net Order Price'] = filtered_df['Net Order Price'].astype(float)
    filtered_df['Net Order Value'] = filtered_df['Net Order Value'].astype(float)

    # Select relevant columns
    tab_df = filtered_df[['Short Text', 'Material', 'Vendor', 'Name 1', 'Purchasing Document', 'Net Order Value']]

    # Convert columns to appropriate data types
    tab_df['Vendor'] = tab_df['Vendor'].astype(str)
    tab_df['Purchasing Document'] = tab_df['Purchasing Document'].astype(str)
    tab_df['Material'] = tab_df['Material'].astype(int)

    # Group by 'Short Text' and 'Material' columns and apply aggregation functions
    gb_df = tab_df.groupby(['Short Text', 'Material']).agg({
        'Vendor': ['nunique', lambda x: ', '.join(map(str, x.unique())) if isinstance(x.unique(), (list, pd.Series)) else str(x.unique())],
        'Name 1': [lambda x: ', '.join(map(str, x.unique())) if isinstance(x.unique(), (list, pd.Series)) else str(x.unique())],
        'Purchasing Document': ['nunique', lambda x: ', '.join(map(str, x.unique())) if isinstance(x.unique(), (list, pd.Series)) else str(x.unique())],
        'Net Order Value': 'sum'
    }).reset_index()

    gb_df.columns = [
        'Short Text',
        'Material',
        'Unique Count(Vendor)',
        'Unique Concatenate(Vendor)',
        'Unique Concatenate(Name 1)',
        'Unique Count(Purchasing Document)',
        'Unique Concatenate(Purchasing Documents)',
        'Sum(Net Order Value)'
    ]

    # Filter rows
    final_df = gb_df[(gb_df['Unique Count(Purchasing Document)'] == 1) & (gb_df['Unique Count(Vendor)'] == 1)]

    return final_df

def process_data_final2_node_679(df, category_filter, vendor_filter):
    """Process data for final2 based on category and vendor filters."""
    # Similar processing as in process_data function with category filter 'F'
    filtered_df = df[(df['Purch. Doc. Category'] == category_filter) & (df['Material'].notna())]
    
    # Rest of the code for processing final2
    
    # Return the processed DataFrame
    return filtered_df

def process_data_final3_node_679(df, category_filter, vendor_filter):
    """Process data for final3 based on category and vendor filters."""
    # Similar processing as in process_data function with category filter 'F' and vendor filter > 1
    filtered_df = df[(df['Purch. Doc. Category'] == category_filter) & (df['Material'].notna())]
    
    # Rest of the code for processing final3
    
    # Return the processed DataFrame
    return filtered_df
# ending function for Node 679


# starting function of inv prior df function
def process_port1_InvDt_To_PODt(df1):
    """
    Process data from port1_csv.csv.

    Args:
        df1 (pd.DataFrame): The DataFrame containing data from port1_csv.csv.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    filter1 = df1[df1['PO History Category'] == 'Q']
    final_df1 = filter1[filter1['Document Date'] != '30.06.0222']

    final_df1['Posting Date'] = pd.to_datetime(final_df1['Posting Date'], format='%Y-%m-%d', errors='coerce')
    final_df1['Entry Date'] = pd.to_datetime(final_df1['Entry Date'], format='%Y-%m-%d', errors='coerce')
    final_df1['Document Date'] = pd.to_datetime(final_df1['Document Date'], format='%Y-%m-%d', errors='coerce')

    final_df1['Quantity'] = final_df1['Quantity'].astype(float)
    final_df1['Amount in LC'] = final_df1['Amount in LC'].astype(float)
    final_df1['Amount'] = final_df1['Amount'].astype(float)
    final_df1['Created by'] = final_df1['Created by'].astype(float)

    final_df1['Quantity'] = final_df1.apply(lambda row: row['Quantity'] * -1 if row['Debit/Credit Ind.'] == 'H' else row['Quantity'], axis=1)
    final_df1['Amount in LC'] = final_df1.apply(lambda row: row['Amount in LC'] * -1 if row['Debit/Credit Ind.'] == 'H' else row['Amount in LC'], axis=1)

    return final_df1

def process_port2_InvDt_To_PODt(df2):
    """
    Process data from port 2_csv.csv.

    Args:
        df2 (pd.DataFrame): The DataFrame containing data from port 2_csv.csv.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    final_df2 = df2[['Purchasing Document', 'Item', 'Storage Location', 'Purchasing Info Rec.', 'Order Price Unit',
                     'Quantity Conversion', 'Quantity Conversion (#1)', 'Equal To', 'Denominator',
                     'Delivery Completed', 'Goods Receipt', 'Shipping type', 'Purchase Requisition',
                     'Item of Requisition', 'Shipping block', 'Requisitioner', 'Created on', 'Created by',
                     'Vendor', 'Release indicator', 'User Name', 'First name', 'Last name', 'Complete name',
                     'Department', 'Change doc. object', 'Object value', 'Document number', 'Table Name', 'Field Name',
                     'Change Indicator', 'Transaction Code', 'Time-N']]

    return final_df2

def process_data_InvDt_To_PODt(df1, df2):
    """
    Process and merge data from port1 and port2.

    Args:
        df1 (pd.DataFrame): The DataFrame containing data from port1_csv.csv.
        df2 (pd.DataFrame): The DataFrame containing data from port 2_csv.csv.

    Returns:
        pd.DataFrame: The merged and processed DataFrame.
    """
    final_df1 = process_port1_InvDt_To_PODt(df1)
    final_df2 = process_port2_InvDt_To_PODt(df2)

    join1 = pd.merge(final_df1, final_df2, on=['Purchasing Document', 'Item'], how='left')
    join1['Document Date'] = pd.to_datetime(join1['Document Date'], format='%Y %m %d')
    join1['Created on'] = pd.to_datetime(join1['Created on'], format='%Y %m %d')
    join1['Diff (Inv. Doc Dt - PO Dt)'] = (join1['Created on'] - join1['Document Date']).dt.days

    return join1

def read_tab3_InvDt_To_PODt(file_path, encoding):
    """
    Read data from tab3_csv.csv.

    Args:
        file_path (str): The path to the CSV file.
        encoding (str): The encoding of the CSV file.

    Returns:
        pd.DataFrame: The DataFrame containing the data.
    """
    return pd.read_csv(file_path, encoding=encoding)[['Vendor', 'Name 1']]

def merge_data_with_tab3_InvDt_To_PODt(join1, df3):
    """
    Merge data from join1 with data from tab3_csv.csv.

    Args:
        join1 (pd.DataFrame): The merged and processed DataFrame from process_data.
        df3 (pd.DataFrame): The DataFrame containing data from tab3_csv.csv.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    join1['Vendor'] = join1['Vendor'].astype(str)
    join1['Vendor'] = join1['Vendor'].astype(str)
    df3['Vendor'] = df3['Vendor'].astype(str)

    join2 = pd.merge(join1, df3, left_on='Vendor', right_on='Vendor', how='left')
    return join2

def filter_final_data_InvDt_To_PODt(join2):
    """
    Filter rows where 'Diff (Inv. Doc Dt - PO Dt)' is greater than 0.

    Args:
        join2 (pd.DataFrame): The merged DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    final_451 = join2[join2['Diff (Inv. Doc Dt - PO Dt)'] > 0.0]
    final = final_451.drop_duplicates(keep='first')
    return final

def save_to_excel_InvDt_To_PODt(data_df, file_path, sheet_name, index=False):
    """
    Save a DataFrame to an Excel file.

    Args:
        data_df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet in the Excel file.
        index (bool): Whether to include the index in the Excel file.

    Returns:
        None
    """
    excel_writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    data_df.to_excel(excel_writer, sheet_name=sheet_name, index=index)
    excel_writer.save()

# ending function of inv prior df function


# starting of the function for the node 837 function over here
def group_data_node_837(df):
    """
    Group data by 'Purchasing Document', 'Short Text', 'Material', and 'Plant' columns and perform aggregations.
    
    Args:
        df (pd.DataFrame): The DataFrame to group and aggregate.
    
    Returns:
        pd.DataFrame: The grouped and aggregated DataFrame.
    """
    gb = df.groupby(['Purchasing Document', 'Short Text', 'Material', 'Plant']).agg({
        'Net Order Price': [('Unique Count', 'nunique'), ('Unique Concatenate', lambda x: ', '.join(map(str, x.unique())))]
    }).reset_index()

    gb.columns = ['Purchasing Document', 'Short Text', 'Material', 'Plant',
                  'Unique_count(Net Order Price)', 'Unique_concat(Net Order Price)']
    
    return gb

def filter_grouped_data_node_837(gb_df, threshold=1):
    """
    Filter grouped data based on a threshold for 'Unique_count(Net Order Price)'.
    
    Args:
        gb_df (pd.DataFrame): The grouped and aggregated DataFrame.
        threshold (int): The threshold for 'Unique_count(Net Order Price)'.
    
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    return gb_df[gb_df['Unique_count(Net Order Price)'] > threshold]

def merge_and_filter_data_node_837(main_df, filter_df):
    """
    Merge the main DataFrame with the filtered DataFrame and filter rows.
    
    Args:
        main_df (pd.DataFrame): The main DataFrame containing data.
        filter_df (pd.DataFrame): The filtered DataFrame.
    
    Returns:
        pd.DataFrame: The merged and filtered DataFrame.
    """
    merged_df = pd.merge(main_df, filter_df, on=['Purchasing Document', 'Short Text', 'Material', 'Plant'], how='left')
    filtered_df = merged_df[merged_df['Unique_count(Net Order Price)'].notna()]
    
    return filtered_df

def save_to_excel_node_837(data_df, file_path):
    """
    Save a DataFrame to an Excel file.
    
    Args:
        data_df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the Excel file.
    """
    excel_writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    data_df.to_excel(excel_writer, sheet_name='Summary', index=False)
    excel_writer.save()
# ending of the function for the node 837 function over here


# starting of the function for the node 753 function over here
def filter_dataframe_Node_753(df):
    filter1 = df[df['Material'].notna()]
    filter1['Order Quantity'] = filter1['Order Quantity'].astype(float)
    filter1['Net Order Price'] = filter1['Net Order Price'].astype(float)
    filter1['Net Order Value'] = filter1['Net Order Value'].astype(float)
    filter1['Principal Agmt Item'] = filter1['Net Order Value'].astype(float)
    filter1['Material'] = filter1['Material'].astype(str)
    filter1['Short Text'] = filter1['Short Text'].astype(str)
    filter1['Unique Key'] = filter1['Material'] + "_" + filter1['Short Text']
    return filter1

def group_by_net_order_price_Node_753(df):
    gb1 = df.groupby(['Unique Key']).agg({
        'Net Order Price': ['nunique']
    }).reset_index()
    gb1.columns = ['Unique Key', 'Unique Key*(Net Order Price)']
    return gb1

def filter_by_unique_key_Node_753(gb_df):
    filter2 = gb_df[gb_df['Unique Key*(Net Order Price)'] > 1]
    return filter2

def merge_dataframes_Node_753(df1, df2, include=True):
    main_column = df1['Unique Key']
    reference_column = df2['Unique Key']
    if include:
        result_table = pd.merge(df1, df2, left_on=main_column, right_on=reference_column, how='inner')
    else:
        result_table = df1[~df1[main_column].isin(df2[reference_column])]
    result_table.drop(['key_0', 'Unique Key_y', 'Unique Key*(Net Order Price)'], axis=1, inplace=True)
    result_table.rename(columns={'Unique Key_x': 'Unique Key'}, inplace=True)
    return result_table
# ending of the function for the node 753 function over here

# starting function for pan node over here


# ending function for pan node over here





# starting of the node 686 over here

def process_data_NODE_686(input_file1, input_file2, input_file3, output_file):
    # Read the input CSV files
    df1 = pd.read_csv(input_file1, encoding='Windows-1252')
    df2 = pd.read_csv(input_file2, encoding='Windows-1252')
    df3 = pd.read_csv(input_file3, encoding='Windows-1252')

def process_df1_NODE_686(df):
    df['Requisition Date'] = pd.to_datetime(df['Requisition Date'])
    df['Created on'] = pd.to_datetime(df['Created on'])
    df['Date&Time diff'] = (df['Created on'] - df['Requisition Date']).dt.days
    return df[(df['Date&Time diff'] >= -1) & (df['Date&Time diff'] <= 1)]


def process_df2_NODE_686(df2, df1):
    filter1 = df2[df2['PO History Category'] == 'Q']
    tab1 = filter1[['Purchasing Document', 'Item', 'Material Doc. Year', 'Material Document', 'Material Doc.Item',
                        'PO History Category', 'Movement Type', 'Posting Date', 'Quantity', 'Amount in LC', 'Currency',
                        'Debit/Credit Ind.', 'Delivery Completed', 'Entry Date', 'Time of Entry', 'Material', 'Plant',
                        'Local currency', 'Batch', 'Document Date']]

    tab2 = df1[['Purchasing Document', 'Item', 'Deletion Indicator', 'RFQ status', 'Short Text', 'Material',
                    'Company Code', 'Plant', 'Storage Location', 'Material Group', 'Order Quantity', 'Order Unit',
                    'Order Price Unit', 'Quantity Conversion', 'Quantity Conversion (#1)', 'Equal To', 'Denominator',
                    'Net Order Price', 'Price Unit', 'Net Order Value', 'Overdeliv. Tolerance', 'Unltd Overdelivery',
                    'Underdel. Tolerance', 'Valuation Type', 'Valuation Category', 'Delivery Completed', 'Invoice Receipt',
                    'GR-Based Inv. Verif.', 'Outline Agreement', 'Principal Agmt Item', 'Shipping Instr.', 'Incoterms',
                    'Incoterms (Part 2)', 'Purchase Requisition', 'Item of Requisition', 'Requisitioner',
                    'Company Code (right)', 'Purch. Doc. Category', 'Purchasing Doc. Type', 'Status', 'Created on',
                    'Vendor', 'Terms of Payment', 'Payment in', 'Purch. Organization', 'Purchasing Group', 'Currency',
                    'Exchange Rate', 'Validity Per. Start', 'Validity Period End', 'Incoterms (right)',
                    'Incoterms (Part 2) (right)', 'Release group', 'Release Strategy', 'Release indicator', 'Release status',
                    'VAT Registration No.', 'Retention', 'Retention in Percent', 'Down Payment', 'Down Payt Percentage',
                    'Down Payment Amount', 'Due Date for Down Payment', 'Contract Name', 'Release Date of Contract',
                    'Shipping type', 'Shipping Conditions', 'RFQ Selected', 'RFQ Comments', 'Type of Contract', 'Contract Value',
                    'Purch. Order Type Desc.', 'Purch Doc. Cate Desc.', 'Name 1', 'Document Type', 'Deletion Indicator (right)',
                    'Release indicator (right)', 'Created by (right)', 'Quantity Requested', 'Unit of Measure', 'Requisition Date',
                    'Release Date']]

    join1 = pd.merge(tab1, tab2, on=['Purchasing Document', 'Item'], how='left')
    join1['Document Date'] = pd.to_datetime(join1['Document Date'], format='%Y-%m-%d', errors='coerce')
    join1['Created on'] = pd.to_datetime(join1['Created on'], format='%Y-%m-%d', errors='coerce')
    join1['Date&Time diff'] = (join1['Created on'] - join1['Document Date']).dt.days
    return join1[join1['Date&Time diff'] >= 0]

def process_df3_NODE_686(df3, df2):
        filter2 = df3[df3['Field Name'] == 'FRGKE']
        filter2['Date'] = pd.to_datetime(filter2['Date'], format='%Y-%m-%d', errors='coerce')
        tab3 = filter2[['Object value', 'Date & Time Stamp', 'Date']]
        tab3['Object value'].fillna('Unknown', inplace=True)
        gb1 = tab3.groupby('Object value').agg({'Date & Time Stamp': 'max', 'Date': 'max'}).reset_index()
        gb1.columns = ['Object value', 'Max(Date & Time Stamp)', 'Max(Date)']
        gb1['Object value'].replace('Unknown', None, inplace=True)
        tab4 = df2[['Purchasing Document', 'Item', 'Material Doc. Year', 'Material Document', 'Material Doc.Item',
                    'PO History Category', 'Movement Type', 'Posting Date', 'Quantity', 'Amount in LC', 'Currency_x',
                    'Debit/Credit Ind.', 'Delivery Completed_x', 'Entry Date', 'Time of Entry', 'Material_x', 'Plant_x',
                    'Local currency', 'Batch', 'Document Date', 'Deletion Indicator', 'RFQ status', 'Short Text',
                    'Material_y', 'Plant_y', 'Storage Location', 'Material Group', 'Order Quantity', 'Order Unit',
                    'Order Price Unit', 'Quantity Conversion', 'Quantity Conversion (#1)', 'Equal To', 'Denominator',
                    'Net Order Price', 'Price Unit', 'Net Order Value', 'Overdeliv. Tolerance', 'Unltd Overdelivery',
                    'Underdel. Tolerance', 'Valuation Type', 'Valuation Category', 'Delivery Completed_y', 'Invoice Receipt',
                    'GR-Based Inv. Verif.', 'Outline Agreement', 'Principal Agmt Item', 'Shipping Instr.', 'Incoterms',
                    'Incoterms (Part 2)', 'Purchase Requisition', 'Item of Requisition', 'Requisitioner',
                    'Company Code (right)', 'Purch. Doc. Category', 'Purchasing Doc. Type', 'Status', 'Created on',
                    'Vendor', 'Terms of Payment', 'Payment in', 'Purch. Organization', 'Purchasing Group', 'Currency_y',
                    'Exchange Rate', 'Validity Per. Start', 'Validity Period End', 'Incoterms (right)', 'Incoterms (Part 2) (right)',
                    'Release group', 'Release Strategy', 'Release indicator', 'Release status', 'VAT Registration No.', 'Retention',
                    'Retention in Percent', 'Down Payment', 'Down Payt Percentage', 'Down Payment Amount', 'Due Date for Down Payment',
                    'Contract Name', 'Release Date of Contract', 'Shipping type', 'Shipping Conditions', 'RFQ Selected', 'RFQ Comments',
                    'Type of Contract', 'Contract Value', 'Purch. Order Type Desc.', 'Purch Doc. Cate Desc.', 'Name 1', 'Document Type',
                    'Deletion Indicator (right)', 'Release indicator (right)', 'Created by (right)', 'Quantity Requested', 'Unit of Measure',
                    'Requisition Date', 'Release Date']]

        join2 = pd.merge(tab4, gb1, left_on=['Purchasing Document'], right_on=['Object value'], how='left')
        join2.drop('Object value', axis=1, inplace=True)
        join2['Document Date'] = pd.to_datetime(join2['Document Date'], format='%Y-%m-%d', errors='coerce')
        join2['Max(Date)'] = pd.to_datetime(join2['Max(Date)'], format='%Y-%m-%d', errors='coerce')
        join2['Diff Inv. Dt vs Max. Dt'] = (join2['Max(Date)'] - join2['Document Date']).dt.days
        return join2[join2['Diff Inv. Dt vs Max. Dt'] > 0]

# ending of the node 686 over here



def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")




def get_download_link(file_path):
    try:
        with open(file_path, "rb") as file:
            contents = file.read()
        encoded_file = base64.b64encode(contents).decode("utf-8")
        href = f'<a href="data:file/csv;base64,{encoded_file}" download="{file_path}">Click here to download</a>'
        return href
    except Exception as e:
        raise CustomException(e, sys)
    



    
def drop_features_with_missing_values(data):
    try:
        # Calculate the number of missing values in each column
        missing_counts = data.isnull().sum()

        # Get the names of columns with missing values
        columns_with_missing_values = missing_counts[missing_counts > 0].index

        # Drop the columns with missing values
        data_dropped = data.drop(columns=columns_with_missing_values)
        return data_dropped
    except Exception as e:
        raise CustomException(e, sys)




def apply_anomaly_detection_Mahalanobis(data):
        # Assuming 'data' is a pandas DataFrame with numerical columns
        # You may need to preprocess and select appropriate features for Mahalanobis distances

        # Calculate the mean and covariance matrix of the data
        data_mean = data.mean()
        data_cov = data.cov()

        # Calculate the inverse of the covariance matrix
        data_cov_inv = np.linalg.inv(data_cov)

        # Calculate Mahalanobis distances for each data point
        mahalanobis_distances = data.apply(lambda row: mahalanobis(row, data_mean, data_cov_inv), axis=1)

        # Set a threshold to identify anomalies (you can adjust this threshold based on your dataset)
        threshold = mahalanobis_distances.mean() + 2 * mahalanobis_distances.std()

        # Create a new column 'Anomaly' to indicate anomalies (1 for anomalies, 0 for inliers)
        data['Anomaly_RC'] = (mahalanobis_distances > threshold).astype(int)

        return data
    

# Function to define and train the autoencoder model
def train_autoencoder(data):
    try:
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Define the autoencoder model architecture
        input_dim = data.shape[1]
        encoding_dim = int(input_dim / 2)  # You can adjust this value as needed
        autoencoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])

        # Compile and train the autoencoder with verbose=1
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(scaled_data, scaled_data, epochs=100, batch_size=64, shuffle=True, verbose=1)  # Set verbose to 1

        # Get the encoded data
        encoded_data = autoencoder.predict(scaled_data)

        # Calculate the reconstruction error
        reconstruction_error = np.mean(np.square(scaled_data - encoded_data), axis=1)

        # Add the reconstruction error as a new column 'ReconstructionError' to the data
        data['ReconstructionError'] = reconstruction_error

        return data
    except Exception as e:
        raise CustomException(e, sys)
    


# Function to apply autoencoder for anomaly detection
def apply_anomaly_detection_autoencoder(data):
    try:
        # Train the autoencoder and get the reconstruction error
        data_with_reconstruction_error = train_autoencoder(data)

        # Set a threshold for anomaly detection (you can adjust this threshold)
        threshold = data_with_reconstruction_error['ReconstructionError'].mean() + 3 * data_with_reconstruction_error['ReconstructionError'].std()

        # Classify anomalies based on the threshold
        data_with_reconstruction_error['Anomaly'] = np.where(data_with_reconstruction_error['ReconstructionError'] > threshold, 1, 0)

        return data_with_reconstruction_error
    except Exception as e:
        raise CustomException(e, sys)
    
# over here we have the fuction for isolation forest
def apply_anomaly_detection_IsolationForest(data):
    # Make a copy of the data
    data_copy = data.copy()

    # Fit the Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.03, random_state=42)
    isolation_forest.fit(data_copy)

    # Predict the anomaly labels
    anomaly_labels = isolation_forest.predict(data_copy)

    # Create a new column in the original DataFrame for the anomaly indicator
    data['Anomaly_IF'] = np.where(anomaly_labels == -1, 1, 0)
    return data

def apply_anomaly_detection_LocalOutlierFactor(data, neighbors=200):
    try:
        lof = LocalOutlierFactor(n_neighbors=neighbors, contamination='auto')
        data['Anomaly_LOF'] = lof.fit_predict(data)
        data['Anomaly_LOF'] = np.where(data['Anomaly_LOF'] == -1, 1, 0)
        return data
    except Exception as e:
        raise CustomException(e, sys)


# def apply_anomaly_detection_LocalOutlierFactor(data):
#     try:


#         # Make a copy of the data
#         data_copy = data.copy()

#         from sklearn.neighbors import LocalOutlierFactor

#         # Step 3: Apply Local Outlier Factor
#         lof = LocalOutlierFactor(n_neighbors=200, metric='euclidean', contamination=0.04)

#         outlier_labels = lof.fit_predict(data_copy)

#         # Display the outlier labels for each data point
#         data['Outlier_Label'] = outlier_labels
#         return data
#     except Exception as e:
#         raise CustomException(e, sys)



def find_duplicate_vendors(vendors_df, threshold):
    try:
        duplicates = []
        lf = vendors_df.copy()
        lf['NAME1'] = lf['NAME1'].astype(str)
        vendor_names = lf['NAME1'].unique().tolist()
        columns = ['Vendor 1', 'Vendor 2', 'Score']
        df_duplicates = pd.DataFrame(data=[], columns=columns)

        for i, name in enumerate(vendor_names):
            # Compare the current name with the remaining names
            matches = process.extract(name, vendor_names[i+1:], scorer=fuzz.ratio)

            # Check if any match exceeds the threshold
            for match, score in matches:
                if score >= threshold:
                    duplicates.append((name, match))
                    df_duplicates.loc[len(df_duplicates)] = [name, match, score]

        return duplicates, df_duplicates
    except Exception as e:
        raise CustomException(e, sys)



def apply_anomaly_detection_OneClassSVM(data):
        # Copy the original data to avoid modifying the original dataframe
        data_with_anomalies = data.copy()

        # Perform One-Class SVM anomaly detection
        clf = OneClassSVM(nu=0.05)
        y_pred = clf.fit_predict(data)
        data_with_anomalies['Anomaly_OneClassSVM'] = np.where(y_pred == -1, 1, 0)

        return data_with_anomalies


def apply_anomaly_detection_SGDOCSVM(data):
        # Copy the original data to avoid modifying the original dataframe
        data_with_anomalies = data.copy()

        
        # Perform One-Class SVM anomaly detection using SGD solver
        clf = SGDOneClassSVM(nu=0.05)
        clf.fit(data)
        y_pred = clf.predict(data)
        data_with_anomalies['Anomaly_OneClassSVM(SGD)'] = np.where(y_pred == -1, 1, 0)

        return data_with_anomalies



def calculate_first_digit(data):
    try:
        idx = np.arange(0, 10)
        first_digits = data.astype(str).str.strip().str[0].astype(int)
        counts = first_digits.value_counts(normalize=True, sort=False)
        benford = np.log10(1 + 1 / np.arange(0, 10))

        df = pd.DataFrame(data.astype(str).str.strip().str[0].astype(int).value_counts(normalize=True, sort=False)).reset_index()
        df1 = pd.DataFrame({'index': idx, 'benford': benford})
        return df, df1, counts, benford
    except Exception as e:
        raise CustomException(e, sys)

def calculate_2th_digit(data):
    try:
        idx = np.arange(0, 100)
        nth_digits = data.astype(int).astype(str).str.strip().str[:2]
        numeric_mask = nth_digits.str.isnumeric()
        counts = nth_digits[numeric_mask].astype(int).value_counts(normalize=True, sort=False)
        benford = np.log10(1 + 1 / np.arange(0, 100))

        df = pd.DataFrame(data.astype(int).astype(str).str.strip().str[:2].astype(int).value_counts(normalize=True, sort=False)).reset_index()
        df1 = pd.DataFrame({'index': idx, 'benford': benford})

        return df, df1, counts, benford
    except Exception as e:
        raise CustomException(e, sys)
    
# working of benford's third law over here
def extract_first_three_digits(number):
    if isinstance(number, (int, float)):
        first_three = int(str(int(number))[:3])
        if 100 <= first_three <= 999:
            return first_three
    return None


# working of benford's second law over here
def extract_first_two_digits(number):
    if isinstance(number, (int, float)):
        first_two = int(str(int(number))[:2])
        if 10 <= first_two <= 99:
            return first_two
    return None

def extract_first_digit(number):
    if isinstance(number, (int, float)):
        number_str = str(number)
        if '.' in number_str:  # Check for decimal point
            number_str = number_str.split('.')[0]  # Remove decimal part
        if number_str[0] == '-' and len(number_str) > 1:  # Check for negative sign
            first_digit = int(number_str[1])
        else:
            first_digit = int(number_str[0])
        if 1 <= first_digit <= 9:
            return first_digit
    return None


def apply_anomaly_detection_GMM(data):
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture()
        data['Anomaly'] = gmm.fit_predict(data)
        data['Anomaly'] = np.where(data['Anomaly'] == 1, 0, 1)
        return data
    except Exception as e:
        raise CustomException(e, sys)


def z_score_anomaly_detection(data, column, threshold):

        # Calculate the z-score for the specified column
        z_scores = stats.zscore(data[column])

        # Identify outliers based on the z-score exceeding the threshold
        outlier_indices = np.where(np.abs(z_scores) > threshold)[0]

        # Create a copy of the data with an "Anomaly" column indicating outliers
        data_with_anomalies_zscore = data.copy()
        data_with_anomalies_zscore['Anomaly_ZScore'] = 0
        data_with_anomalies_zscore.iloc[outlier_indices, -1] = 1

        return data_with_anomalies_zscore

def navbar():
    try:
        
        tabs = ["HOME", "ABOUT","EXCEL TO CSV", "PROCESS MINING","EXPLORATORY DATA ANALYSIS", "STATISTICAL METHODS", "MACHINE LEARNING METHODS", "DEEP LEARNING METHODS", "TIME SERIES METHODS"]
        tab0, tab1,tab2, tab3,tab8,tab4, tab5, tab6, tab7 = st.tabs(tabs)

        with tab0:
            video_file = open('D:\InfraredProduct\src\InfraredProduct\infrared.mp4', 'rb')
            video_bytes = video_file.read()


            st.video(video_bytes)
            pass
        with tab8:
            st.header("Performing EDA over here ")
            st.write("""
            As we all know, Exploratory Data Analysis (EDA) is all about gathering brief information regarding the dataset. 
            In this particular section, we have added multiple useful features, such as:
            - Number of one-time vendor accounts
            - Duplicate vendor accounts based on name and fuzzy logic
            - Changes in the vendor bank account
            - Vendor details matching with employee
            - Dormant vendors
            - Vendors with bank accounts in different countries or cities
            - Vendors belonging to countries in Africa, Turkey, Bangkok, etc.
            - Vendors having email addresses with domains like Gmail, Yahoo Mail, Hotmail
            - Vendors that are not private limited, public limited, or LLP
            - Vendors with the maximum quality rejections
            - And many more
            """)
            st.image("https://editor.analyticsvidhya.com/uploads/24537Zoom-EDA.png", use_column_width=True)

        with tab1:
            st.header("Infrared")
            st.write("""
            Introducing a groundbreaking, first-of-its-kind concept that revolutionizes the way you uncover counterintuitive patterns and gain insights often concealed by the inherent limitations of the human mind, prevalent biases, and the sheer volume of data.

Imagine unleashing the formidable power of cutting-edge machine learning and advanced statistical techniques to meticulously identify outliers and exceptions within your datasets. This innovative application goes beyond traditional data analysis, providing an instantaneous output that can be promptly reviewed and acted upon with unparalleled agility.

The implications are profound—this tool empowers you to proactively address potential revenue leakages, enhance operational efficiency, and fortify your defenses against fraudulent activities. By transcending the boundaries of conventional analytics, it opens up new vistas of understanding, allowing you to stay ahead of the curve and make data-driven decisions that drive success.

In a landscape where data complexity often hinders true comprehension, this solution stands as a beacon, guiding you through the intricacies of your information landscape. Break free from the shackles of limited perception and explore the untapped potential within your data ecosystem, redefining the way you navigate, interpret, and leverage your information for strategic advantage.
""")
            # st.write("A first of its kind concept that lets you discover counterintuitive patterns and insights often invisible due to limitations of the human mind, biases, and voluminous data.")
            # st.write("Unleash the power of machine learning and advanced statistics to find outliers and exceptions in your data. This application provides an instant output that can be reviewed and acted upon with agility to stop revenue leakages, improve efficiency, and detect/prevent fraud.")

            st.image("https://img.freepik.com/free-vector/gradient-network-connection-background_23-2148881321.jpg?w=1380&t=st=1700479268~exp=1700479868~hmac=db8e177b8b6e32cf5061f64fc6579b1f5d42051153318d5aa4e0bd7db07b8bdb", use_column_width=True)


        with tab2:
                
                def convert_excel_to_csv(uploaded_file, page_number):
                    if page_number == 1:
                        excel_data = pd.read_excel(uploaded_file)
                    else:
                        excel_data = pd.read_excel(uploaded_file, sheet_name=page_number - 1)
                    csv_file = BytesIO()
                    excel_data.to_csv(csv_file, index=False)
                    csv_file.seek(0)
                    return csv_file.getvalue()


                st.header("Excel to CSV Converter")
                uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
                selected_page = st.number_input("Enter the page number", min_value=1, value=1)

                if uploaded_file is not None:
                    csv_data = convert_excel_to_csv(uploaded_file, selected_page)
                    st.download_button(
                        "Download CSV file",
                        csv_data,
                        file_name="output.csv",
                        mime="text/csv"
                    )

                    with st.expander("Excel Data"):
                        excel_data = pd.read_excel(uploaded_file, sheet_name=selected_page - 1)
                        st.dataframe(excel_data)

                    with st.expander("Converted CSV Data"):
                        csv_data = pd.read_csv(BytesIO(csv_data))
                        st.dataframe(csv_data)


                

        # Move this code block below the page

        with tab4:

            st.subheader("Z Score - Anomaly Detection using Standard Deviation")

            st.write(
                "Z Score is a statistical method commonly used for anomaly detection, particularly when dealing with univariate data. It measures how many standard deviations a data point is from the mean of the dataset. Here's a brief overview of how Z Score works and its application in anomaly detection:"
            )

            st.subheader("How Z Score Works and Its Application in Anomaly Detection")

            st.write(
                "1. Calculation of Z Score:"
            )
            st.write("   • Z Score is calculated using the formula: Z = (X - μ) / σ, where X is the data point, μ is the mean, and σ is the standard deviation of the dataset.")
            st.write(
                "2. Threshold Setting:"
            )
            st.write("   • Anomalies are identified based on a predefined threshold. Data points with Z Scores beyond this threshold are considered anomalies.")
            st.write(
                "3. Positive and Negative Z Scores:"
            )
            st.write("   • Positive Z Scores indicate that a data point is above the mean, while negative Z Scores indicate that a data point is below the mean.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Simplicity: Z Score is a simple and widely used method for identifying anomalies.")
            st.write("• Interpretability: The interpretation of Z Scores is intuitive, as they represent the number of standard deviations a data point is from the mean.")
            st.write("• Sensitivity to Deviations: Z Score is sensitive to deviations from the mean and can effectively identify data points that significantly differ from the norm.")

            st.subheader("Considerations")

            st.write("• Assumption of Normality: Z Score assumes that the underlying distribution of the data is approximately normal.")
            st.write("• Single Variable Focus: Z Score is primarily designed for univariate data and may not capture anomalies in multivariate scenarios.")
            st.write("• Manual Thresholding: Setting an appropriate threshold requires manual tuning and understanding of the data distribution.")


            st.subheader("Boxplot - Anomaly Detection using Quartiles and Outliers")

            st.write(
                "Boxplot, also known as a box-and-whisker plot, is a graphical method for depicting groups of numerical data through their quartiles. It's commonly used for detecting outliers and anomalies in a univariate dataset. Here's a brief overview of how Boxplot works and its application in anomaly detection:"
            )

            st.subheader("How Boxplot Works and Its Application in Anomaly Detection")

            st.write(
                "1. Visualization of Quartiles:"
            )
            st.write("   • A Boxplot visually represents the distribution of data using quartiles. The box represents the interquartile range (IQR) between the first (Q1) and third (Q3) quartiles.")
            st.write(
                "2. Identification of Outliers:"
            )
            st.write("   • Outliers are identified as individual points beyond a certain distance from the box, typically defined as 1.5 times the IQR.")
            st.write(
                "3. Whiskers and Fences:"
            )
            st.write("   • Whiskers extend from the box to the minimum and maximum values within a defined range, known as the 'fences.' Data points beyond the fences are considered potential outliers.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Robustness: Boxplots are robust to skewed distributions and provide a clear visual representation of the central tendency and spread of the data.")
            st.write("• Outlier Detection: Boxplots effectively highlight potential outliers and anomalies in a dataset.")
            st.write("• Easy Interpretation: The interpretation of Boxplots is intuitive, making it accessible for a wide range of users.")

            st.subheader("Considerations")

            st.write("• Univariate Focus: Boxplots are primarily designed for univariate data and may not capture anomalies in multivariate scenarios.")
            st.write("• Sensitivity to Outliers: While Boxplots are good at identifying outliers, they may not be as sensitive to anomalies within the interquartile range.")
            st.write("• Manual Inspection: Interpretation of potential outliers often requires manual inspection, and the choice of outlier detection threshold may vary.")


            st.subheader("Probability Density Function (PDF) - Anomaly Detection using Data Distribution")

            st.write(
                "Probability Density Function (PDF) is a statistical concept used to describe the likelihood of a continuous random variable falling within a particular range. It is commonly employed in anomaly detection to model the distribution of normal data and identify deviations. Here's a brief overview of how PDF works and its application in anomaly detection:"
            )

            st.subheader("How Probability Density Function Works and Its Application in Anomaly Detection")

            st.write(
                "1. Estimation of Data Distribution:"
            )
            st.write("   • PDF involves estimating the underlying data distribution, assuming normality or another appropriate distribution.")
            st.write(
                "2. Calculation of Likelihoods:"
            )
            st.write("   • For each data point, the PDF is used to calculate the likelihood of that point occurring under the assumed distribution.")
            st.write(
                "3. Anomaly Detection Threshold:"
            )
            st.write("   • Anomalies are identified based on a predefined threshold. Data points with low likelihoods under the PDF are considered potential anomalies.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Flexibility: PDF can adapt to various data distributions, making it suitable for capturing complex patterns.")
            st.write("• No Assumption of Specific Distribution: PDF does not assume a specific parametric form for the underlying distribution, allowing it to handle diverse datasets.")
            st.write("• Local Anomaly Detection: PDF can capture local anomalies, identifying regions of the data space where the density is significantly different from the surrounding areas.")

            st.subheader("Considerations")

            st.write("• Distribution Assumption: The effectiveness of PDF depends on the accuracy of the assumed data distribution.")
            st.write("• Parameter Estimation: Proper estimation of distribution parameters is crucial for accurate PDF calculations.")
            st.write("• Sensitivity to Distribution Shape: PDF may not perform well if the shape of the distribution significantly deviates from the assumed form.")

            st.subheader("Benford's Law - Anomaly Detection using First Digit Distribution")

            st.write(
                "Benford's Law, also known as the first-digit law, is a statistical phenomenon that describes the distribution of leading digits in many real-life datasets. It is commonly utilized for anomaly detection, especially in financial and numerical datasets. Here's a brief overview of how Benford's Law works and its application in anomaly detection:"
            )

            st.subheader("How Benford's Law Works and Its Application in Anomaly Detection")

            st.write(
                "1. Leading Digit Distribution:"
            )
            st.write("   • Benford's Law predicts that in many naturally occurring datasets, the distribution of first digits (1 through 9) is not uniform, but follows a logarithmic pattern.")
            st.write(
                "2. Expected Frequency:"
            )
            st.write("   • The law provides expected frequencies for each leading digit based on the logarithmic distribution. For example, the digit '1' is expected to occur more frequently than '9'.")
            st.write(
                "3. Anomaly Detection:"
            )
            st.write("   • Anomalies are identified by comparing the observed distribution of first digits in a dataset with the expected distribution according to Benford's Law. Significant deviations may indicate irregularities.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Applicability Across Datasets: Benford's Law is applicable to a wide range of datasets, including financial statements, population numbers, and scientific data.")
            st.write("• Sensitivity to Deviations: It is sensitive to unexpected deviations in the distribution of first digits, making it useful for detecting anomalies in numerical datasets.")
            st.write("• Non-Intrusive: Benford's Law is non-intrusive and does not require extensive knowledge of the dataset content.")

            st.subheader("Considerations")

            st.write("• Contextual Understanding: Interpretation of results requires understanding the context of the dataset, as certain datasets may naturally deviate from Benford's Law.")
            st.write("• Size of Dataset: Benford's Law is more effective with larger datasets, and results may be less reliable with small sample sizes.")
            st.write("• Multiple Digit Analysis: While Benford's Law primarily focuses on the first digit, extensions exist for analyzing subsequent digits for more detailed analysis.")


            st.subheader("Benford's Second Digit Law - Anomaly Detection using Second Digit Distribution")

            st.write(
                "Benford's Second Digit Law is an extension of Benford's Law, focusing on the distribution of the second digits in numerical datasets. Similar to its predecessor, it is utilized for anomaly detection, particularly in datasets where the first digits follow Benford's Law. Here's a brief overview of how Benford's Second Digit Law works and its application in anomaly detection:"
            )

            st.subheader("How Benford's Second Digit Law Works and Its Application in Anomaly Detection")

            st.write(
                "1. Second Digit Distribution:"
            )
            st.write("   • Benford's Second Digit Law predicts that, in datasets conforming to Benford's Law for the first digits, the distribution of second digits will also exhibit non-uniform patterns.")
            st.write(
                "2. Expected Frequencies:"
            )
            st.write("   • Similar to Benford's Law, the law provides expected frequencies for each second digit based on the logarithmic distribution.")
            st.write(
                "3. Anomaly Detection:"
            )
            st.write("   • Anomalies are identified by comparing the observed distribution of second digits in a dataset with the expected distribution according to Benford's Second Digit Law. Significant deviations may indicate irregularities.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Sequential Pattern Analysis: Benford's Second Digit Law adds a layer of analysis by examining the distribution of second digits, capturing more nuanced deviations in datasets.")
            st.write("• Complementary to First Digit Analysis: It can be used in conjunction with Benford's First Digit Law for a more comprehensive analysis of numerical datasets.")
            st.write("• Sensitivity to Deviations: Benford's Second Digit Law is sensitive to unexpected deviations in the distribution of second digits, enhancing anomaly detection capabilities.")

            st.subheader("Considerations")

            st.write("• Contextual Understanding: Interpretation of results requires understanding the context of the dataset, as certain datasets may naturally deviate from Benford's Second Digit Law.")
            st.write("• Size of Dataset: The effectiveness of Benford's Second Digit Law increases with larger datasets, and results may be less reliable with small sample sizes.")
            st.write("• Sequential Digit Analysis: While the second digit is the focus, extensions exist for analyzing subsequent digits for even more detailed analysis.")

            st.subheader("Benford's Third Digit Law - Anomaly Detection using Third Digit Distribution")

            st.write(
                "Benford's Third Digit Law is an extension of Benford's Law, focusing on the distribution of the third digits in numerical datasets. While less common, it can be utilized for anomaly detection in datasets where the first and second digits follow Benford's Laws. Here's a brief overview of how Benford's Third Digit Law works and its application in anomaly detection:"
            )

            st.subheader("How Benford's Third Digit Law Works and Its Application in Anomaly Detection")

            st.write(
                "1. Third Digit Distribution:"
            )
            st.write("   • Benford's Third Digit Law predicts that, in datasets conforming to Benford's Laws for the first and second digits, the distribution of third digits will also exhibit non-uniform patterns.")
            st.write(
                "2. Expected Frequencies:"
            )
            st.write("   • Similar to Benford's First and Second Digit Laws, the law provides expected frequencies for each third digit based on the logarithmic distribution.")
            st.write(
                "3. Anomaly Detection:"
            )
            st.write("   • Anomalies are identified by comparing the observed distribution of third digits in a dataset with the expected distribution according to Benford's Third Digit Law. Significant deviations may indicate irregularities.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Comprehensive Digit Analysis: Benford's Third Digit Law extends the analysis to a deeper level, capturing potential anomalies in the distribution of third digits.")
            st.write("• Sequential Pattern Examination: It complements the analysis of first and second digits, providing a more comprehensive view of the entire numerical dataset.")
            st.write("• Sensitivity to Deviations: Benford's Third Digit Law is sensitive to unexpected deviations in the distribution of third digits, enhancing anomaly detection capabilities.")

            st.subheader("Considerations")

            st.write("• Limited Applicability: Benford's Third Digit Law is less commonly applied than its predecessors, and its effectiveness may vary depending on the dataset.")
            st.write("• Contextual Understanding: Interpretation of results requires understanding the context of the dataset, as certain datasets may naturally deviate from Benford's Third Digit Law.")
            st.write("• Size of Dataset: The effectiveness of Benford's Third Digit Law increases with larger datasets, and results may be less reliable with small sample sizes.")



            
        with tab5:
            st.subheader("Isolation Forest - Anomaly Detection Algorithm")

            st.write(
                "Isolation Forest is an anomaly detection algorithm that works by isolating instances that are rare and different from the majority of the data. It is particularly useful for identifying outliers or anomalies within a dataset."
            )

            st.subheader("How Isolation Forest Works")

            st.write(
                "1. Random Partitioning: The algorithm randomly selects a feature and then randomly selects a split value within the range of that feature."
            )
            st.write(
                "2. Recursive Partitioning: It recursively applies this random partitioning to create a binary tree. This process continues until each data point is isolated or a predetermined depth is reached."
            )
            st.write(
                "3. Isolation Score: The isolation score is calculated for each data point based on the average path length in the tree. Anomalies typically have shorter average path lengths because they require fewer splits to be isolated."
            )
            st.write(
                "4. Anomaly Detection: The lower the isolation score, the more likely a data point is considered an anomaly. By setting a threshold on the isolation score, you can identify instances that deviate significantly from the majority."
            )

            st.subheader("Usefulness in Anomaly Detection")

            st.write("1. Efficiency: Isolation Forest is computationally efficient, especially for high-dimensional data, making it suitable for large datasets.")
            st.write("2. Unsupervised Learning: It doesn't require a labelled dataset for training, making it an unsupervised learning method.")
            st.write("3. Scalability: It performs well on datasets with a mix of numerical and categorical features and can handle outliers in high-dimensional spaces.")
            st.write("4. Robustness: It is less sensitive to the size and shape of the normal data distribution, making it robust in various scenarios.")




            st.subheader("Kernel Density Estimation (KDE) - Anomaly Detection Method")

            st.write(
                "Kernel Density Estimation (KDE) is a non-parametric method for estimating the probability density function of a random variable. In the context of anomaly detection, KDE can be used to model the distribution of normal data points and identify instances that deviate significantly from this estimated distribution."
            )

            st.subheader("How KDE Works")

            st.write(
                "1. Estimation of Probability Density Function (PDF): KDE starts by placing a kernel (a smooth, symmetric function, e.g., Gaussian) on each data point in the dataset. These kernels represent the probability density at each data point."
            )
            st.write(
                "2. Summation of Kernels: The individual kernel functions are summed up to create a smooth estimate of the probability density function for the entire dataset."
            )
            st.write(
                "3. Normalization: The resulting density function is normalized so that the integral (area under the curve) is equal to 1, ensuring it represents a valid probability distribution."
            )
            st.write(
                "4. Anomaly Detection: New data points are evaluated based on their likelihood under the estimated distribution. Points with low probability density are considered potential anomalies."
            )

            st.subheader("Usefulness in Anomaly Detection")

            st.write("1. Flexibility: KDE is flexible and can adapt to various data distributions, making it suitable for capturing complex patterns.")
            st.write("2. No Assumptions about Data Distribution: It doesn't assume a specific parametric form for the underlying distribution, which can be beneficial when dealing with diverse datasets.")
            st.write("3. Local Anomaly Detection: KDE can capture local anomalies, identifying regions of the data space where the density is significantly lower than the surrounding areas.")
            st.write(
                "4. Bandwidth Parameter: The choice of the bandwidth parameter in KDE is crucial. A smaller bandwidth can lead to overfitting, capturing noise as anomalies, while a larger bandwidth may over smooth the distribution and miss anomalies."
            )

            st.subheader("K-Means - Clustering Algorithm with Limited Use in Anomaly Detection")

            st.write(
                "K-means is a clustering algorithm that partitions a dataset into K clusters, where each data point belongs to the cluster with the nearest mean. While K-means itself is not designed for anomaly detection, it can be used in a simple way for this purpose."
            )

            st.subheader("How K-Means Works and Limited Use in Anomaly Detection")

            st.write(
                "1. Initialization: Randomly select K initial cluster centres."
            )
            st.write(
                "2. Assignment: Assign each data point to the cluster whose mean is closest to it (usually using Euclidean distance)."
            )
            st.write(
                "3. Update: Recalculate the mean of each cluster based on the assigned data points."
            )
            st.write(
                "4. Repeat: Repeat steps 2 and 3 until convergence (when the cluster assignments stabilize or a maximum number of iterations is reached)."
            )

            st.subheader("Usefulness in Anomaly Detection")

            st.write(
                "While K-means itself is not designed for anomaly detection, anomalies might be identified based on how dissimilar they are to the cluster centres. Here's a simple approach:"
            )
            st.write("1. Define Cluster Centres: After running K-means and obtaining cluster centres, consider data points that are significantly far away from their assigned cluster centre.")
            st.write("2. Distance Threshold: Set a distance threshold beyond which a data point is considered an anomaly. Points with distances exceeding this threshold are flagged as potential anomalies.")

            st.subheader("Limitations")

            st.write("1. Spherical Clusters: K-means assumes that clusters are spherical and equally sized, which may not be suitable for datasets with irregularly shaped or differently sized clusters.")
            st.write("2. Sensitive to Initialization: The choice of initial cluster centres can affect the final results, and the algorithm may converge to different solutions.")
            st.write("3. Not Robust to Outliers: K-means is sensitive to outliers, and the presence of anomalies can skew the cluster centres.")





            st.subheader("Density-Based Spatial Clustering of Applications with Noise (DBSCAN) - Clustering Algorithm and Anomaly Detection")

            st.write(
                "Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a clustering algorithm that can also be used for anomaly detection. It works by identifying dense regions of data points, separated by sparser areas."
            )

            st.subheader("How DBSCAN Works and Its Application in Anomaly Detection")

            st.write(
                "1. Density-Based Clustering: DBSCAN defines clusters as dense regions of data points separated by areas that are less dense. It doesn't assume spherical clusters and can find clusters of arbitrary shapes."
            )
            st.write(
                "2. Parameters:"
            )
            st.write("   - Epsilon (ε): A distance threshold that defines the maximum distance between two data points for one to be considered in the neighborhood of the other.")
            st.write("   - MinPts: The minimum number of data points required to form a dense region (cluster).")
            st.write(
                "3. Core Points, Border Points, and Noise:"
            )
            st.write("   - Core Point: A data point is a core point if there are at least MinPts data points (including itself) within distance ε.")
            st.write("   - Border Point: A data point that is within distance ε of a core point but doesn't have enough neighbors to be a core point itself.")
            st.write("   - Noise Point: A data point that is neither a core point nor a border point.")
            st.write(
                "4. Cluster Formation:"
            )
            st.write("   - A core point starts a new cluster, and all data points within distance ε from this core point are added to the cluster.")
            st.write("   - If a border point is within distance ε of a core point, it is added to the same cluster.")
            st.write("   - The process continues until no more points can be added to the cluster.")
            st.write(
                "5. Outliers as Noise:"
            )
            st.write("   - Data points that are not part of any cluster are considered outliers or noise.")
            st.write(
                "Usefulness in Anomaly Detection:"
            )
            st.write("   - Anomalies as Noise: DBSCAN naturally identifies outliers as noise points that don't belong to any cluster. These noise points can be considered anomalies.")
            st.write("   - Variable Cluster Shapes: DBSCAN is robust to clusters of arbitrary shapes, making it suitable for datasets where clusters may not be well-defined geometrically.")
            st.write("   - Not Sensitive to Cluster Size: Unlike K-means, DBSCAN does not assume clusters are of equal size, making it more flexible.")
            st.write(
                "Considerations:"
            )
            st.write("   - Parameter Tuning: Proper choice of ε and MinPts is crucial. Tuning these parameters can impact the algorithm's sensitivity to outliers.")
            st.write("   - Density Variation: DBSCAN might struggle with datasets where the density of normal points varies significantly.")



            st.subheader("Principal Component Analysis (PCA) - Dimensionality Reduction and Anomaly Detection")

            st.write(
                "Principal Component Analysis (PCA) is a dimensionality reduction technique that is commonly used for feature extraction and data compression. While PCA itself is not designed for anomaly detection, it can indirectly be used for this purpose by identifying components that capture most of the variance in the data."
            )

            st.subheader("How PCA Works and Its Application in Anomaly Detection")

            st.write(
                "1. Covariance Matrix: PCA starts by computing the covariance matrix of the original data. The covariance matrix describes the relationships between different features in the dataset."
            )
            st.write(
                "2. Eigendecomposition: PCA then performs eigendecomposition on the covariance matrix to find its eigenvectors and eigenvalues. The eigenvectors represent the principal components, and the eigenvalues indicate the amount of variance captured by each principal component."
            )
            st.write(
                "3. Selection of Principal Components: Principal components are sorted based on their corresponding eigenvalues, and the top components are selected to retain most of the variance in the data. This reduces the dimensionality of the dataset."
            )
            st.write(
                "4. Projection: The original data is then projected onto the subspace spanned by the selected principal components, effectively reducing the dimensionality of the data."
            )

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Anomalies in Reduced Dimensionality: Anomalies may manifest as data points that do not align well with the patterns captured by the dominant principal components. These anomalies might be identified by examining the residuals (the difference between the original data and its projection onto the reduced-dimensional space).")
            st.write("• Visualization: By reducing the data to a lower-dimensional space, it becomes easier to visualize and identify outliers or anomalies.")
            st.write("• Focus on High-Variance Directions: The top principal components capture the directions of maximum variance in the data. Anomalies often exhibit variations that might not be well-represented by these dominant directions.")

            st.subheader("Considerations")

            st.write("• Threshold Setting: Anomalies can be identified by setting a threshold on the reconstruction error or residuals. Instances with high reconstruction errors are potential anomalies.")
            st.write("• Assumptions: PCA assumes that the normal data lies in a subspace of lower dimensionality. If anomalies do not conform to this assumption, they might not be effectively captured.")



            st.subheader("K Nearest Neighbors (KNN) - Supervised Learning Algorithm with Anomaly Detection Adaptation")

            st.write(
                "K Nearest Neighbors (KNN) is a supervised machine learning algorithm commonly used for classification and regression tasks. However, it can also be adapted for anomaly detection. Here's a brief overview of how KNN works and its application in anomaly detection:"
            )

            st.subheader("How KNN Works and Its Application in Anomaly Detection")

            st.write(
                "1. Training Phase:"
            )
            st.write("   • In the training phase of KNN, the algorithm memorizes the entire dataset.")
            st.write(
                "2. Distance Calculation:"
            )
            st.write("   • To classify or detect anomalies for a new data point, KNN calculates the distance between that point and every other point in the training dataset.")
            st.write(
                "3. Nearest Neighbors:"
            )
            st.write("   • It identifies the k-nearest neighbors of the new data point based on the calculated distances. These neighbors are the data points with the smallest distances to the new point.")
            st.write(
                "4. Classification or Anomaly Detection:"
            )
            st.write("   • For classification, the new data point is assigned the majority class label among its k-nearest neighbors.")
            st.write("   • For anomaly detection, the distance to the k-nearest neighbors can be used to determine whether the new point is significantly different from its neighbors.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Outlier Detection: Anomalies are identified as data points that have few nearby neighbors or are significantly distant from their k-nearest neighbors.")
            st.write("• Distance Metric Choice: The choice of distance metric (e.g., Euclidean, Manhattan) is crucial and depends on the characteristics of the data.")
            st.write("• Adjusting Thresholds: Anomalies can be detected by setting a threshold on the distances. Points with distances beyond the threshold are considered anomalies.")
            st.write("• Adaptability: KNN is adaptable to different types of data and can be effective in scenarios where anomalies have distinct patterns.")

            st.subheader("Considerations")

            st.write("• Parameter Tuning: The choice of the value for 'k' (number of neighbors) and the distance metric can impact the algorithm's performance. Cross-validation or other tuning methods may be necessary.")
            st.write("• Computational Cost: Calculating distances to all data points in the training set can be computationally expensive for large datasets.")
            st.write("• Curse of Dimensionality: KNN can suffer from the curse of dimensionality, where the effectiveness decreases as the number of dimensions increases.")




            st.subheader("Elliptic Envelope - Robust Covariance Estimation for Anomaly Detection")

            st.write(
                "Elliptic Envelope is an algorithm used for robustly estimating the covariance of a dataset and identifying anomalies based on the assumption that the normal data points follow a multivariate Gaussian distribution. Here's a brief overview of how Elliptic Envelope works and its application in anomaly detection:"
            )

            st.subheader("How Elliptic Envelope Works and Its Application in Anomaly Detection")

            st.write(
                "1. Covariance Estimation:"
            )
            st.write("   • Elliptic Envelope estimates the mean and covariance matrix of the input data. It assumes that the majority of the data points are normally distributed.")
            st.write(
                "2. Mahalanobis Distance:"
            )
            st.write("   • It calculates the Mahalanobis distance for each data point, which measures the distance of a point from the center of the distribution in terms of the spread of the distribution (covariance).")
            st.write(
                "3. Outlier Detection:"
            )
            st.write("   • Points with Mahalanobis distances exceeding a certain threshold are considered anomalies. The assumption is that normal data points should fall within a certain range defined by the estimated covariance.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Robust to Outliers: Elliptic Envelope is robust to outliers and can provide a measure of the 'normality' of data points based on their Mahalanobis distances.")
            st.write("• Adaptable Covariance Estimation: It can adaptively estimate the shape and orientation of the distribution, making it suitable for datasets with varying covariance structures.")
            st.write("• Multivariate Gaussian Assumption: Well-suited for datasets where anomalies are expected to deviate from the multivariate Gaussian distribution assumed for normal data.")

            st.subheader("Considerations")

            st.write("• Assumption of Gaussian Distribution: The effectiveness of Elliptic Envelope depends on how well the assumption of a multivariate Gaussian distribution holds for the normal data. It may not perform well if this assumption is violated.")
            st.write("• Threshold Setting: The choice of the Mahalanobis distance threshold is crucial, and it may need to be adjusted based on the characteristics of the data.")
            st.write("• Sensitive to Outliers in Training: Outliers in the training set used for covariance estimation can influence the algorithm's performance.")

            st.subheader("Local Outlier Factor (LOF) - Unsupervised Anomaly Detection Algorithm")

            st.write(
                "Local Outlier Factor (LOF) is an unsupervised anomaly detection algorithm that measures the local density deviation of a data point with respect to its neighbors. It is based on the idea that anomalies will have lower local density compared to their neighbors. Here's a brief overview of how LOF works and its application in anomaly detection:"
            )

            st.subheader("How LOF Works and Its Application in Anomaly Detection")

            st.write(
                "1. Local Density Estimation:"
            )
            st.write("   • LOF calculates the local density for each data point by comparing its distance to its k-nearest neighbors. The local density is a measure of how crowded or sparse the neighborhood of a point is.")
            st.write(
                "2. Reachability Distance:"
            )
            st.write("   • The reachability distance of a point is the distance to its k-nearest neighbor with the highest local density. It represents how far a point can 'reach' within its local density neighborhood.")
            st.write(
                "3. Local Outlier Factor Calculation:"
            )
            st.write("   • The Local Outlier Factor of a point is the ratio of its local density to the average local density of its k-nearest neighbors. A lower LOF indicates that the point is less dense than its neighbors, suggesting it might be an anomaly.")
            st.write(
                "4. Anomaly Detection:"
            )
            st.write("   • Data points with higher LOF values are considered anomalies. These are points with lower local density compared to their neighbors.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Adaptability to Local Density: LOF is effective at identifying anomalies in regions where the local density varies, making it suitable for datasets with clusters of different densities.")
            st.write("• No Assumption of Global Distribution: LOF does not assume a specific global distribution for the data, making it more robust to complex and irregularly shaped clusters.")
            st.write("• Sensitivity to Local Context: LOF captures anomalies based on their local context, allowing it to identify anomalies that may not be apparent when considering the entire dataset.")

            st.subheader("Considerations")

            st.write("• Parameter Tuning: The choice of the number of neighbors (k) influences the sensitivity of LOF. It may require some parameter tuning based on the characteristics of the data.")
            st.write("• Scalability: Calculating LOF for large datasets can be computationally expensive.")
            st.write("• Normalization: Features with different scales can impact LOF. Normalization or scaling may be necessary.")


            st.subheader("One-Class Support Vector Machine (SVM) with Stochastic Gradient Descent (SGD) - Anomaly Detection Algorithm")

            st.write(
                "One-Class Support Vector Machine (SVM), often implemented using Stochastic Gradient Descent (SGD), is an anomaly detection algorithm that learns a representation of normal data and identifies anomalies based on deviations from this representation. Here's a brief overview of how One-Class SVM with SGD works and its application in anomaly detection:"
            )

            st.subheader("How One-Class SVM with SGD Works and Its Application in Anomaly Detection")

            st.write(
                "1. Training on Normal Data:"
            )
            st.write("   • One-Class SVM is trained on a dataset containing only normal data, assuming that anomalies are rare and might not be well-represented.")
            st.write(
                "2. Hyperplane Construction:"
            )
            st.write("   • The algorithm seeks to find a hyperplane that separates the normal data from the origin in feature space. This hyperplane is positioned to maximize the margin between the normal data and the origin.")
            st.write(
                "3. Outlier Detection:"
            )
            st.write("   • During the testing phase, data points are projected onto the learned hyperplane. Points that fall on the side of the hyperplane opposite to the normal data are considered anomalies.")
            st.write(
                "4. Stochastic Gradient Descent (SGD):"
            )
            st.write("   • SGD is an optimization technique used to iteratively update the parameters of the model based on a small subset of the training data at each iteration. It's often employed to efficiently train large-scale models, including One-Class SVM.")

            st.subheader("Usefulness in Anomaly Detection")

            st.write("• Robust to Noise: One-Class SVM with SGD is robust to noise and outliers in the training data, as it aims to learn a representation of the majority class.")
            st.write("• Unsupervised Learning: It is an unsupervised learning method, requiring only normal data for training, making it suitable for scenarios where labelled anomalies are scarce.")
            st.write("• Flexibility: One-Class SVM is flexible in capturing the shape of normal data, allowing it to handle non-linear boundaries.")

            st.subheader("Considerations")

            st.write("• Assumption of Rare Anomalies: One-Class SVM assumes that anomalies are rare in the dataset, and it may not perform well if this assumption is violated.")
            st.write("• Parameter Tuning: The choice of hyperparameters, such as the width of the margin, is crucial for the algorithm's performance and may require tuning.")
            st.write("• Limited Anomaly Information: One-Class SVM does not distinguish between different types of anomalies; it only indicates whether a data point is considered normal or not.")

        with tab6:
            st.subheader("Autoencoder")

            st.write(
                "Autoencoders are a type of artificial neural network used for unsupervised learning and data compression. They are particularly useful for feature extraction and anomaly detection tasks. The basic idea behind autoencoders is to learn a compressed representation of the input data and then reconstruct it as accurately as possible.")

            st.subheader("Architecture of Autoencoder")
            st.write("An autoencoder consists of two main parts: the encoder and the decoder.")
            st.write(
                "1. Encoder: The encoder takes the input data and learns a compressed representation, also known as the encoding or latent space.")
            st.write(
                "2. Decoder: The decoder takes the encoded representation and reconstructs the original input data from it.")

            st.write(
                "The encoder and decoder are typically symmetric in structure, with the number of neurons decreasing in the encoder and increasing in the decoder.")

            st.subheader("Training an Autoencoder")
            st.write(
                "Autoencoders are trained using an unsupervised learning approach. The goal is to minimize the reconstruction error between the original input and the reconstructed output. This is typically done by minimizing a loss function, such as mean squared error (MSE) or binary cross-entropy (BCE).")

            st.subheader("Applications of Autoencoder")
            st.write("Autoencoders have various applications, including:")
            st.write(
                "- Dimensionality reduction: Learning compressed representations that capture the most important features of the data.")
            st.write(
                "- Anomaly detection: Detecting unusual or anomalous patterns in the data by comparing reconstruction errors.")
            st.write(
                "- Image denoising: Removing noise or artifacts from images by training the autoencoder to reconstruct clean images.")
            st.write("- Recommendation systems: Learning user preferences and generating personalized recommendations.")
            st.write("- Data generation: Generating new data samples similar to the training data.")

            st.subheader("Example: Image Denoising")
            st.write(
                "One application of autoencoders is image denoising. By training an autoencoder on noisy images and minimizing the reconstruction error, we can effectively remove the noise and reconstruct clean images.")

            st.image("https://miro.medium.com/v2/resize:fit:4266/1*QEmCZtruuWwtEOUzew2D4A.png", use_column_width=True, caption="Autoencoder Image Denoising")

            st.markdown("---")
            st.write(
                "In this blog post, we explored autoencoders, a type of artificial neural network used for unsupervised learning and data compression. Autoencoders consist of an encoder and a decoder, which learn a compressed representation of the input data and reconstruct it as accurately as possible.")
            st.write(
                "We discussed the training process of autoencoders, which involves minimizing the reconstruction error between the original input and the reconstructed output. Autoencoders have various applications, including dimensionality reduction, anomaly detection, image denoising, recommendation systems, and data generation.")
            st.write(
                "We also provided an example of image denoising using autoencoders, where the network learns to remove noise from noisy images and reconstruct clean images.")
            st.write(
                "By utilizing autoencoders, data scientists and researchers can effectively extract features, detect anomalies, denoise images, and generate new data samples. Autoencoders have wide-ranging applications and are particularly valuable in unsupervised learning scenarios.")
            st.write(
                "We hope this blog post has provided you with a clear understanding of autoencoders and their applications. Remember to explore further and apply autoencoders to different domains and datasets to unlock their full potential.")
            st.write("Happy autoencoding!")


        with tab3:
            # Create a graphlib graph object
            st.header("The following visualization shows one of the paths for a particular organization.")
            graph = graphviz.Digraph()
            graph.edge('PO_Creation', 'Final Released')
            graph.edge('Final Released', 'GRN Reversal')
            graph.edge('GRN Reversal', 'GRN')
           
            st.graphviz_chart(graph)

     


            # iframe_url = '<iframe title="process mining" width="1000" height="700" src="https://app.powerbi.com/view?r=eyJrIjoiNWE5ZDM0MDYtYmUwNC00ZjhiLTllOGMtNjFjNmY2M2M4YzkxIiwidCI6IjMyNTRjOGVlLWQxZDUtNDFmNy05ZTY5LTUxMzQxYjJhZWU3NCJ9&embedImagePlaceholder=true" frameborder="0" allowFullScreen="true"></iframe>'

            iframe_url = """
              <div class="marquee">
                <span style="color: #E3F4F4; background-color: #2b86d9;">It comes with pre-built statistical and machine learning models specifically designed to identify outliers in large-scale data.</span>
            </div>
            <center>
            <br>
                <a href="https://github.com/ravipratap366/LLM_chatbot">
                    <div class="cardGif 2" id="gif_card">
                    <div class="card_image_gif" id="gif_card">
                        <iframe title="process mining" width="100%" height="700" src="https://app.powerbi.com/view?r=eyJrIjoiNWE5ZDM0MDYtYmUwNC00ZjhiLTllOGMtNjFjNmY2M2M4YzkxIiwidCI6IjMyNTRjOGVlLWQxZDUtNDFmNy05ZTY5LTUxMzQxYjJhZWU3NCJ9&embedImagePlaceholder=true" frameborder="0" allowFullScreen="true"></iframe>'
                        </div>
                </a>
                </div>

            </center>
"""
            # Embed the Power BI report in the Streamlit app
            st.markdown(iframe_url, unsafe_allow_html=True)
    except Exception as e:
        raise CustomException(e, sys)




def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            if pdf.type == "application/pdf":
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif pdf.type == "text/plain":
                text += pdf.getvalue().decode("utf-8")
            else:
                raise Exception("Invalid file type. Please upload a valid PDF or text file.")
        return text
    except Exception as e:
        raise CustomException(e, sys)


def get_text_chunks(text):
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        raise CustomException(e, sys)


def get_vectorstore(text_chunks):
    try:
        # embeddings = OpenAIEmbeddings()
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        raise CustomException(e, sys)


def get_conversation_chain(vectorstore):
    try:
        # llm = ChatOpenAI()
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        raise CustomException(e, sys)



def handle_userinput(user_question):
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        raise CustomException(e, sys)


