import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

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

def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:

            vectors.append(
                model.wv[word]
            )

    if len(vectors) == 0:

        return np.zeros(
            model.vector_size
        )

    return np.mean(
        vectors,
        axis=0
    )
   #Phase 7.1: Topic Word Clouds
     
def create_topic_wordclouds(lda_model):
    num_topics = lda_model.num_topics

    plt.figure(figsize=(15, 10))

    for topic_id in range(num_topics):

        plt.subplot(
            (num_topics + 1) // 2,
            2,
            topic_id + 1
        )

        words = dict(
            lda_model.show_topic(
                topic_id,
                topn=20
            )
        )

        wc = WordCloud(
            width=400,
            height=300,
            background_color="white"
        )

        wc.generate_from_frequencies(
            words
        )

        plt.imshow(wc)

        plt.axis("off")

        plt.title(
            f"Topic {topic_id}"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/topic_wordclouds.png"
    )

    plt.show()
   
    


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

   
    # TF-IDF PCA VISUALIZATION


    print("\nReducing TF-IDF Dimensions...")

    svd = TruncatedSVD(
        n_components=50,
        random_state=42
        )

    tfidf_reduced = svd.fit_transform(
        X_tfidf
        )

    print(
        "Reduced Shape:",
        tfidf_reduced.shape
        )

    # PCA to 2D

    pca_tfidf = PCA(
        n_components=2,
        random_state=42
)

    tfidf_pca = pca_tfidf.fit_transform(
        tfidf_reduced
        )

    tfidf_plot_df = pd.DataFrame(
        {
            "PC1": tfidf_pca[:, 0],
            "PC2": tfidf_pca[:, 1],
            "sentiment": df["sentiment"]
            }
            )

    plt.figure(figsize=(10,8))

    sns.scatterplot(
        data=tfidf_plot_df,
        x="PC1",
        y="PC2",
        hue="sentiment",
        alpha=0.6
        )
    plt.title(
        "TF-IDF PCA Visualization"
        )

    plt.tight_layout()

    plt.savefig(
        "outputs/tfidf_pca.png"
        )

    plt.show()
    print(
        "TF-IDF PCA Plot Saved!"
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
 
    # PHASE 6: WORD2VEC + PCA
    

    print("\nTraining Word2Vec Model...")

    w2v_model = Word2Vec(
        sentences=df["w2v_tokens"],
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )

    print("Word2Vec Training Complete!")

    # Save Model

    w2v_model.save(
        "outputs/word2vec.model"
    )

    print("Word2Vec Model Saved!")

     
    # DOCUMENT EMBEDDINGS
    

    print(
        "\nCreating Document Embeddings..."
    )

    doc_vectors = np.array(
        [
            document_vector(
                tokens,
                w2v_model
            )
            for tokens in df["w2v_tokens"]
        ]
    )

    print(
        "Embedding Shape:",
        doc_vectors.shape
    )

    
    # PCA REDUCTION
     

    print(
        "\nApplying PCA..."
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_vectors = pca.fit_transform(
        doc_vectors
    )

   
    # PCA DATAFRAME
   

    pca_df = pd.DataFrame(
        {
            "PC1": pca_vectors[:, 0],
            "PC2": pca_vectors[:, 1],
            "sentiment": df["sentiment"]
        }
    )
 
    # PCA SCATTER PLOT
     

    plt.figure(
        figsize=(10, 8)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="sentiment",
        alpha=0.6
    )

    plt.title(
        "Word2Vec PCA Visualization"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/word2vec_pca.png"
    )

    plt.show()

    print(
        "Word2Vec PCA Plot Saved!"
    )

   
    # SAVE PCA DATA
    

    pca_df.to_csv(
        "outputs/word2vec_pca.csv",
        index=False
    )

    print(
        "Word2Vec PCA Data Saved!"
    )
   
    # PHASE 7: LDA TOPIC MODELING


        # ==================================
    # PHASE 7: LDA TOPIC MODELING
    # ==================================

    print("\nCreating Dictionary...")

    dictionary = corpora.Dictionary(
        df["pos_tokens"]
    )

    print(
        "Vocabulary Size:",
        len(dictionary)
    )

    print(
        "\nCreating Bag-of-Words Corpus..."
    )

    corpus = [
        dictionary.doc2bow(doc)
        for doc in df["pos_tokens"]
    ]

    print(
        "Corpus Size:",
        len(corpus)
    )

    # Topic counts to test
    topic_numbers = [5, 7, 10]

    lda_models = {}

    coherence_scores = {}

    # Train LDA Models

    for k in topic_numbers:

        print(
            f"\nTraining LDA with {k} topics..."
        )

        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=42,
            passes=10
        )

        lda_models[k] = lda

        coherence_model = CoherenceModel(
            model=lda,
            texts=df["pos_tokens"],
            dictionary=dictionary,
            coherence="c_v"
        )

        score = coherence_model.get_coherence()

        coherence_scores[k] = score

        print(
            f"Coherence Score = {score:.4f}"
        )

    # Select Best Model

    best_k = max(
        coherence_scores,
        key=coherence_scores.get
    )

    best_lda = lda_models[
        best_k
    ]

    print(
        f"\nBest Model: {best_k} Topics"
    )

    # Print Topics

    print(
        "\nBest Topics:\n"
    )

    topics = best_lda.print_topics(
        num_words=10
    )

    for topic in topics:
        print(topic)

    # Save Topics

    with open(
        "outputs/topics.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for topic in topics:

            f.write(
                str(topic) + "\n"
            )
    
    #WORD CLOUDS FOR TOPICS

    print("\nGenerating Topic Word Clouds...")
    create_topic_wordclouds(
        best_lda
        )
    
    print(
        "\nAssigning Dominant Topics..."
        )

    print("\nAssigning Dominant Topics...")

    dominant_topics = []

    for doc in corpus:
        topic_probs = best_lda.get_document_topics(
            doc
            )

    dominant_topic = max(
        topic_probs,
        key=lambda x: x[1]
        )[0]

    dominant_topics.append(
        dominant_topic
        )

    print(
        "Number of Topics Assigned:",
        len(dominant_topics)
        )

    df["dominant_topic"] = dominant_topics

    #7.3: 
    topic_sentiment = pd.crosstab(
    df["dominant_topic"],
    df["sentiment"],
    normalize="index"
    )

    topic_sentiment.columns = [
        "Negative",
        "Positive"
        ]

    print(
        "\nTopic Sentiment Distribution:\n"
        )

    print(
        topic_sentiment
        )

    topic_sentiment.to_csv(
        "outputs/topic_sentiment_table.csv"
        )
    
# PHASE 8: SENTIMENT CLASSIFICATION
    print("\nCreating Train-Test Split...")

    X = X_tfidf

    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
        )

    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

     # ==================================
# PHASE 8: SENTIMENT CLASSIFICATION
# ==================================

# Step 2
print("\nCreating Train-Test Split...")

X = X_tfidf
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 3
models = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

# STEP 4 STARTS HERE
results = []

plt.figure(figsize=(10,8))

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:,1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={roc_auc:.3f})"
    )

    print(
        f"Accuracy={accuracy:.4f}"
    ) 

    #ROC
    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
        )

    plt.xlabel(
        "False Positive Rate"
        )

    plt.ylabel(
        "True Positive Rate"
        )

    plt.title(
        "ROC Curve Comparison"
        )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/roc_curve_comparison.png"
        )

    plt.show()

#Comparison Table
    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC"
            ]
            )

    print(
        "\nClassifier Comparison:\n"
        )
    print(
        results_df
        )

    results_df.to_csv(
        "outputs/classifier_comparison.csv",
        index=False
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