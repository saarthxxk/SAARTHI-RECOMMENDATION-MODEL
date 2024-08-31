import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv('mentors.csv')

# Drop irrelevant columns
data = data.drop(["DOJ", "CURRENT DATE", "LEAVES USED", "LEAVES REMAINING"], axis=1)

# Combine features for TF-IDF
data['combined_features'] = data['UNIT'].fillna('') + ' ' + data['DESIGNATION'].fillna('') + ' ' + data['PAST EXP'].astype(str).fillna('')

# Create TF-IDF matrix
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Function to search by unit
def search_by_unit(unit, tfidf_matrix, data):
    # Filter by unit
    filtered_data = data[data['UNIT'].str.contains(unit, case=False, na=False)]
    
    # Get cosine similarities for the filtered data
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Sort the filtered data by 'PAST EXP'
    sorted_data = filtered_data.sort_values(by='PAST EXP', ascending=False)
    
    return sorted_data

# Streamlit app
st.title("SAARTHI RECOMMENDATION MODEL")

# Input from the user
unit_filter = st.text_input("Enter the unit to filter by")

if st.button("Search"):
    # Perform the search and display results
    results = search_by_unit(unit_filter, tfidf_matrix, data)
    st.write(results[['FIRST NAME', 'LAST NAME', 'UNIT', 'DESIGNATION', 'PAST EXP']])
st.write("TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a term in a document relative to a collection of documents, helping to identify relevant words.")
