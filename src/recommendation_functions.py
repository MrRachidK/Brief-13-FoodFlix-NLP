import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-13-FoodFlix-NLP")

import streamlit as st
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.d01_data_processing.data_processing import final_stopwords_list, foodfacts

@st.cache(allow_output_mutation=True)
def get_model(df, method):
    if method == "TF-IDF":
        model = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), stop_words = final_stopwords_list, min_df = 0)
    elif method == "CountVectorizer":
        model = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), stop_words = final_stopwords_list, min_df = 0)
    X = model.fit_transform(df['content'])
    return model, X

def get_results(model, X, user_input, number_similarity):
    user_matrix = model.transform([user_input])
    cosine_similarities = linear_kernel(user_matrix, X)
    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-(number_similarity+1):-1]
    similar_items = [(cosine_similarities[0][i], foodfacts['id'][i]) for i in similar_indices]
    results = similar_items

    return results

def get_nutrition_values(df, id):
    series = df.loc[df['id'] == id, ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g', 'fruits-vegetables-nuts_100g']]
    return series
