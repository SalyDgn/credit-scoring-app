import pickle

import pandas as pd
import streamlit as st
from PIL import Image

# Charger le modèle ML
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Charger l'image
image = Image.open('Home_Credit_logo.svg.png')

# Redimensionner l'image
new_width = 300  # spécifiez la largeur souhaitée
new_height = int((new_width / image.width) * image.height)
resized_image = image.resize((new_width, new_height))

# Afficher l'image avec Streamlit
st.image(resized_image, use_column_width=True)

# Charger le dataset de test
test_data = pd.read_csv('test_data_final.csv', index_col=0)


# Fonction pour obtenir les informations du client et faire une prédiction
def get_client_data(client_id, data):
    client_data = data[data['SK_ID_CURR'] == client_id]
    if client_data.empty:
        return None
    return client_data


# Titre de l'applicationpip
st.title("Credit Scoring Application")

# Color background
page_bg_color = '''
<style>
body {
    background-color: #ADD8E6; /* couleur de fond (bleu clair dans cet exemple) */
}
</style>
'''
# Appliquer le CSS
st.markdown(page_bg_color, unsafe_allow_html=True)

# Champ de saisie pour l'identifiant du client
client_id = st.text_input("Please enter client ID; ex: 100001, 100005")

if client_id:
    client_id = int(client_id)
    client_data = get_client_data(client_id, test_data)

    if client_data is not None:
        st.subheader("Customer's features")
        st.write(client_data)

        # Préparer les données du client pour la prédiction
        features = client_data.drop(columns=['SK_ID_CURR'])

       # Faire la prédiction
        prediction = model.predict_proba(features)
        st.subheader("Predict Result")
        if prediction[0][0] > 0.5:
            st.markdown(f'<p style="color:green;">The customer is more likely to repay the loan at {prediction[0][0] * 100:.2f}% </p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="color:red;">The customer is a more likely to be a defaulter at {prediction[0][1] * 100:.2f}%</p>', unsafe_allow_html=True)
    else:
            st.write("Custumer not found")
