import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Load anime data
Anime = pd.read_csv("Anime_Clean.csv")

# Initialize CountVectorizer
cv = CountVectorizer(max_features=13500, stop_words="english")
vectors = cv.fit_transform(Anime["Tags"]).toarray()

# Calculate cosine similarity
Similarity = cosine_similarity(vectors)

# Preprocess anime names for case-insensitive matching
Anime["Name_new"] = Anime["Name"].apply(lambda x: x.lower())
Anime["Name_new"] = Anime["Name_new"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
Anime["Name_new"] = Anime["Name_new"].apply(lambda x: " ".join(x.split()))
Anime["Name_new"] = Anime["Name_new"].apply(lambda x: x.replace(" ", "-"))

# Define Recommender function
def Recommender(user_input):
    Index_of_anime = Anime[Anime["Name"] == user_input].index[0]
    Similarity_score = Similarity[Index_of_anime]
    Sorted_scores = sorted(
        list(enumerate(Similarity_score)), reverse=True, key=lambda x: x[1]
    )[1:6]
    Recommended_Anime = []
    for i in Sorted_scores:
        Recommended_Anime.append(Anime.iloc[i[0]].Name)
    return Recommended_Anime

# Streamlit UI
def main():
    st.title("Anime Recommendation System")
    user_input = st.text_input("Enter an anime title:")
    if st.button("Get Recommendations"):
        if user_input:
            try:
                recommended_anime = Recommender(user_input)
                st.success("Top 5 Recommended Anime:")
                for anime in recommended_anime:
                    st.write(anime)
            except IndexError:
                st.error("Anime not found. Please try another one.")

if __name__ == "__main__":
    main()
