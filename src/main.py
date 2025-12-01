# src/main.py
import pandas as pd
from joblib import dump

from text_prep import Text_prep
from bow import BoW
from plsa import PLSA
from fisher import Fisher_Vectorizer
from svm import SVM_Classifier

from scipy.sparse import csr_matrix

MAX_TRAIN = 10000   # Anzahl Tweets fürs PLSA/FK-SVM-Experiment
MAX_VAL   = 2000
MAX_TEST  = 2000

def main():
    print("Starting the application...")

    train_path = "data/train/sentiment140-train.csv"
    val_path   = "data/val/sentiment140-val.csv"
    test_path  = "data/test/sentiment140-test.csv"

    train_texts, y_train = load_file(train_path)
    val_texts,   y_val   = load_file(val_path)
    test_texts,  y_test  = load_file(test_path)

    if not train_texts or not val_texts or not test_texts:
        print("Failed to load one or more files. Exiting the application.")
        return

    print("File content loaded successfully...")

    # create smaller subsets
    train_texts = train_texts[:MAX_TRAIN]
    y_train     = y_train[:MAX_TRAIN]
    val_texts = val_texts[:MAX_VAL]
    y_val     = y_val[:MAX_VAL]
    test_texts = test_texts[:MAX_TEST]
    y_test     = y_test[:MAX_TEST]
    print("Datasets truncated to max sizes...")

    # -------- Preprocessing --------
    train_proc = Text_prep(train_texts)
    val_proc   = Text_prep(val_texts)
    test_proc  = Text_prep(test_texts)

    token_list  = train_proc.preprocess_list()
    val_tokens  = val_proc.preprocess_list()
    test_tokens = test_proc.preprocess_list()
    print("Tokens were created successfully...")

    # -------- Vokabular nur aus TRAIN --------
    freq_dict = train_proc.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")

    sorted_tokens_list = train_proc.sort_tokens(freq_dict)
    print("Dictionary is sorted descending...")
    print(sorted_tokens_list[:10])

    bow = BoW()
    word2idx, idx2word = bow.build_vocabulary(sorted_tokens_list)
    print("Vocabulary built successfully...")

    # -------- BoW-Matrizen --------
    X_train = bow.vectorize_list(token_list)
    X_val   = bow.vectorize_list(val_tokens)
    X_test  = bow.vectorize_list(test_tokens)
    print("BoW-Matrices created successfully...")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    # -------- PLSA: generatives Modell --------
    plsa = PLSA(num_topics=20, iterations=15)
    topic_train = plsa.fit(X_train)
    topic_val   = plsa.transform(X_val)
    topic_test  = plsa.transform(X_test)
    print("PLSA topic features created successfully...")
    print("topic_train shape:", topic_train.shape)

    # in Sparse-Form umwandeln, damit Fisher_Vectorizer wie bisher arbeiten kann
    T_train = csr_matrix(topic_train.astype("float32"))
    T_val   = csr_matrix(topic_val.astype("float32"))
    T_test  = csr_matrix(topic_test.astype("float32"))

    # -------- Fisher-Features im Themenraum --------
    fisher_vectorizer = Fisher_Vectorizer(alpha=1.0)
    Phi_train = fisher_vectorizer.fit_transform(T_train)
    Phi_val   = fisher_vectorizer.transform(T_val)
    Phi_test  = fisher_vectorizer.transform(T_test)
    print("Fisher-Features from PLSA created successfully...")

    # -------- SVM --------
    svm_classifier = SVM_Classifier(C=1.0, max_iter=2000)
    svm_classifier.train(Phi_train, y_train)
    print("SVM Classifier trained successfully...")

    print("Train labels:", len(y_train),
          "Val labels:", len(y_val),
          "Test labels:", len(y_test))

    val_acc  = svm_classifier.evaluate(Phi_val,  y_val)
    test_acc = svm_classifier.evaluate(Phi_test, y_test)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    # -------- Modell speichern --------
    model_bundle = {
        "bow": bow,
        "plsa": plsa,
        "fisher": fisher_vectorizer,
        "svm": svm_classifier.model,
        "idx2word": idx2word,
    }
    dump(model_bundle, "models/fisher_svm_sentiment.joblib")
    print("Model saved to models/fisher_svm_sentiment.joblib")


def load_file(file_path):
    """
    Lädt CSV mit Spalten 'polarity' und 'text' und gibt
    (texts, labels) zurück.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin-1')

        if "polarity" not in df.columns or "text" not in df.columns:
            raise ValueError(f"'polarity' oder 'text' Spalte fehlt in {file_path}")

        df["text"] = df["text"].fillna("").astype(str)
        df["polarity"] = df["polarity"].astype(int)

        texts  = df["text"].tolist()
        labels = df["polarity"].tolist()
        return texts, labels

    except Exception as e:
        print(f"An error occurred while loading the file '{file_path}': {e}")
        return [], []


if __name__ == "__main__":
    main()