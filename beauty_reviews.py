import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load SpaCy model
nlp = spacy.load("en_core_web_lg")


def load_data():
    df = pd.read_csv("data/beauty_reviews.csv")
    return df


# Stream 1: POS Filtered
def pos_filtered_tokens(text):

    doc = nlp(str(text))

    tokens = []

    for token in doc:

        if token.pos_ in ["NOUN", "ADJ"]:

            if not token.is_stop and token.is_alpha:

                tokens.append(token.lemma_.lower())

    return tokens


# Stream 2: Word2Vec Stream
def lightly_cleaned_tokens(text):

    doc = nlp(str(text))

    tokens = []

    for token in doc:

        if not token.is_stop and token.is_alpha:

            tokens.append(token.lemma_.lower())

    return tokens


# NER Function
def extract_entities(text):

    doc = nlp(str(text))

    entities = []

    for ent in doc.ents:
        entities.append(ent.text)

    return entities


if __name__ == "__main__":

    # Load Data
    df = load_data()

    # POS Stream
    print("Creating POS stream...")

    df["pos_tokens"] = df["review_text"].apply(
        pos_filtered_tokens
    )

    # Word2Vec Stream
    print("Creating Word2Vec stream...")

    df["w2v_tokens"] = df["review_text"].apply(
        lightly_cleaned_tokens
    )

    # NER
    print("Extracting entities...")

    df["entities"] = df["review_text"].apply(
        extract_entities
    )

    print("\nSample Entities:")
    print(df[["entities"]].head())

    # Collect all entities
    all_entities = []

    for entity_list in df["entities"]:
        all_entities.extend(entity_list)

    # Count frequencies
    entity_counter = Counter(all_entities)

    # Top 20 entities
    top20 = entity_counter.most_common(20)

    print("\nTop 20 Entities:\n")

    for entity, count in top20:
        print(f"{entity}: {count}")

    # Create Bar Chart
    names = [x[0] for x in top20]
    counts = [x[1] for x in top20]

    plt.figure(figsize=(12, 6))

    sns.barplot(
        x=counts,
        y=names
    )

    plt.title("Top 20 Mentioned Entities")

    plt.tight_layout()

    plt.savefig(
        "outputs/entity_bar_chart.png"
    )

    plt.show()

    # Show sample processed data
    print("\nProcessed Data Sample:\n")

    print(
        df[
            [
                "review_text",
                "pos_tokens",
                "w2v_tokens"
            ]
        ].head()
    )