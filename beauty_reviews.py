import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")



# LOAD DATA

def load_data():
    df = pd.read_csv("data/beauty_reviews.csv")
    return df


# STREAM 1
# POS FILTERED TOKENS

def pos_filtered_tokens(text):

    doc = nlp(str(text))

    tokens = []

    for token in doc:

        if token.pos_ in ["NOUN", "ADJ"]:

            if not token.is_stop and token.is_alpha:

                tokens.append(
                    token.lemma_.lower()
                )

    return tokens



# STREAM 2
# WORD2VEC TOKENS

def lightly_cleaned_tokens(text):

    doc = nlp(str(text))

    tokens = []

    for token in doc:

        if not token.is_stop and token.is_alpha:

            tokens.append(
                token.lemma_.lower()
            )

    return tokens



# NER FUNCTION

def extract_entities(text):

    doc = nlp(str(text))

    entities = []

    for ent in doc.ents:
        entities.append(ent.text)

    return entities



# MAIN

if __name__ == "__main__":

    # Load Dataset
    df = load_data()

    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)

   
    # POS STREAM
   
    print("\nCreating POS stream...")

    df["pos_tokens"] = df["review_text"].apply(
        pos_filtered_tokens
    )

   
    # WORD2VEC STREAM

    print("\nCreating Word2Vec stream...")

    df["w2v_tokens"] = df["review_text"].apply(
        lightly_cleaned_tokens
    )

  
    # NER 
   
    print("\nExtracting entities...")

    df["entities"] = df["review_text"].apply(
        extract_entities
    )

    print("\nSample Entities:\n")

    print(
        df[
            ["entities"]
        ].head()
    )

  
    # TOP 20 ENTITIES
  
    all_entities = []

    for entity_list in df["entities"]:
        all_entities.extend(
            entity_list
        )

    entity_counter = Counter(
        all_entities
    )

    top20 = entity_counter.most_common(
        20
    )

    print("\nTop 20 Entities:\n")

    for entity, count in top20:

        print(
            f"{entity}: {count}"
        )


    # ENTITY BAR CHART
   
    names = [x[0] for x in top20]
    counts = [x[1] for x in top20]

    plt.figure(
        figsize=(12, 6)
    )

    sns.barplot(
        x=counts,
        y=names
    )

    plt.title(
        "Top 20 Mentioned Entities"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/entity_bar_chart.png"
    )

    plt.show()

    print(
        "\nEntity chart saved successfully!"
    )

 
    # TF-IDF CORPUS
   
    print(
        "\nCreating TF-IDF Corpus..."
    )

    df["pos_text"] = df[
        "pos_tokens"
    ].apply(
        lambda x: " ".join(x)
    )

    print(
        "\nSample POS Text:\n"
    )

    print(
        df["pos_text"].head()
    )

    # TF-IDF MATRIX

    print(
        "\nGenerating TF-IDF Matrix..."
    )

    vectorizer = TfidfVectorizer()

    X_tfidf = vectorizer.fit_transform(
        df["pos_text"]
    )

    print(
        "\nTF-IDF Matrix Shape:"
    )

    print(
        X_tfidf.shape
    )

    
    # FEATURE NAMES
   
    feature_names = (
        vectorizer.get_feature_names_out()
    )

    print(
        "\nFirst 20 TF-IDF Features:\n"
    )

    print(
        feature_names[:20]
    )

    # SAVE TF-IDF VECTORIZER
   
    with open(
        "outputs/tfidf_vectorizer.pkl",
        "wb"
    ) as f:

        pickle.dump(
            vectorizer,
            f
        )

    print(
        "\nTF-IDF Vectorizer Saved Successfully!"
    )

      
    # PHASE 5: COSINE SIMILARITY HEATMAP
   

    print("\nStarting Cosine Similarity Analysis...")

    # Select 50 random reviews
    sample_df = df.sample(
        n=50,
        random_state=42
    )

    sample_indices = sample_df.index

    # Get TF-IDF vectors of sampled reviews
    sample_tfidf = X_tfidf[
        sample_indices
    ]

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(
        sample_tfidf
    )

    print("\nSimilarity Matrix Shape:")
    print(similarity_matrix.shape)

    # Create labels with sentiment
    labels = []

    for i, sentiment in enumerate(
            sample_df["sentiment"]):

        if sentiment == 1:
            labels.append(f"P{i}")
        else:
            labels.append(f"N{i}")

    # Plot Heatmap
    plt.figure(figsize=(14, 12))

    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis"
    )

    plt.title(
        "Cosine Similarity Heatmap (50 Random Reviews)"
    )

    plt.xlabel("Reviews")
    plt.ylabel("Reviews")

    plt.tight_layout()

    plt.savefig(
        "outputs/cosine_similarity_heatmap.png"
    )

    plt.show()

    print(
        "\nCosine Similarity Heatmap Saved!"
    )

    # Save similarity matrix
    similarity_df = pd.DataFrame(
        similarity_matrix
    )

    similarity_df.to_csv(
        "outputs/cosine_similarity_matrix.csv",
        index=False
    )

    print(
        "Similarity Matrix CSV Saved!"
    )

    # ==================================
    # SAMPLE PROCESSED DATA
    # ==================================

    print(
        "\nProcessed Data Sample:\n"
    )

    print(
        df[
            [
                "review_text",
                "pos_tokens",
                "w2v_tokens",
                "pos_text"
            ]
        ].head()
    )


    
    # SAMPLE PROCESSED DATA
   
    print(
        "\nProcessed Data Sample:\n"
    )

    print(
        df[
            [
                "review_text",
                "pos_tokens",
                "w2v_tokens",
                "pos_text"
            ]
        ].head()
    )