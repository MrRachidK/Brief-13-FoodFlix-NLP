import sys
sys.path.insert(0, "/home/apprenant/Documents/Brief-13-FoodFlix-NLP")

import streamlit as st
import src.sessionstate as SessionState
import pandas as pd 
from src.recommendation_functions import foodfacts, get_model, get_results, get_nutrition_values

st.title("""
**Moteur de recommandation FoodFlix**
"""
)

state = SessionState.get(position=10)

method = st.sidebar.radio("Quelle méthode voulez-vous utiliser ?", ("TF-IDF", "CountVectorizer"))

user_input = st.sidebar.text_input("Tapez le nom du produit que vous recherchez :")

state.position = st.sidebar.slider("Nombre de résultats :", 0, 50, state.position, 1)

st.sidebar.write('Filtres:')
option_c = st.sidebar.checkbox('Catégories')
option_i = st.sidebar.checkbox('Ingrédients')
option_s = st.sidebar.checkbox('Score nutritionnel')
option_v = st.sidebar.checkbox('Valeurs nutritionnelles')

model, X = get_model(foodfacts, method)

if user_input:
    results = get_results(model, X, user_input, state.position)
    for i in results:
        st.write("__Proposition n°{}__ :".format((results.index(i)+1)))
        st.write("_Nom du produit_ :", foodfacts['product_name'][i[1]], "(score : {})".format(round(i[0], ndigits=5)))
        if option_c:
            st.write("_Catégorie_ :", foodfacts['categories'][i[1]])
        if option_i:
            st.write("_Ingrédients_ :", foodfacts['ingredients_text'][i[1]])
        if option_s:
            st.write("_Score nutritionnel_ :", foodfacts['nutrition_grade_fr'][i[1]].upper())
        if option_v:
            st.table(get_nutrition_values(foodfacts, i[1]))
        st.write("--------------------------------------------------")



