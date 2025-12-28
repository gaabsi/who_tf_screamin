import streamlit as st
import os

from evaluation import PredictEval

st.title("Who TF Screamin")

@st.cache_resource
def load_model():
    BASE_DIR = os.path.expanduser("~/who_tf_screamin")
    checkpoint_path = os.path.join(BASE_DIR, "models/version_0/version_0.ckpt")
    feature_extractor = "MIT/ast-finetuned-audioset-10-10-0.4593"
    CSV_PATH = os.path.join(BASE_DIR, "data/pokedex.csv")
    
    return PredictEval(
        checkpoint_path = checkpoint_path,
        feature_extractor = feature_extractor,
        csv_path = CSV_PATH
    )

uploaded_file = st.file_uploader("Upload un cri", type = ['mp3'])

if uploaded_file:
    st.audio(uploaded_file)
    
    if st.button("Prédire"):
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        predictor = load_model()
        nom_pred, _, conf, _ = predictor.predict(temp_path)
        
        st.write(f"**Pokémon:** {nom_pred}")
        st.write(f"**Confiance:** {conf*100:.1f}%")
        
        os.remove(temp_path)