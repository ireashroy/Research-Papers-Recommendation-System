# pip install tensorflow==2.15.0
# pip install torch==2.0.1
# pip install sentence_transformers==2.2.2
# pip install streamlit

# import necessary libraries
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras

# Load saved recommendation models
embeddings = pickle.load(open('models/embeddings.pkl','rb'))
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))

# Load saved prediction models
loaded_model = keras.models.load_model("models/model.h5")
with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)
with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)
with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)

# Custom functions for recommendation
def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list

# Functions for subject area prediction
def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    # Preprocess the abstract using the loaded text vectorizer
    preprocessed_abstract = vectorizer([abstract])

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_abstract)

    # Convert predictions to human-readable labels
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels

# Create Streamlit app interface
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Based App")

# Input for paper title and abstract
input_paper = st.text_input("Enter Paper Title...")
new_abstract = st.text_area("Paste Paper Abstract...")

# Recommendation button
if st.button("Recommend"):
    if input_paper and new_abstract:
        # Recommendation part
        recommend_papers = recommendation(input_paper)
        st.subheader("Recommended Papers")
        for paper in recommend_papers:
            st.write(paper)

        # Subject area prediction part
        st.write("===================================================================")
        predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
        st.subheader("Predicted Subject Areas")
        st.write(predicted_categories)
    else:
        st.error("Please enter both paper title and abstract.")
