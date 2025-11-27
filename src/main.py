# src/main.py
import pandas as pd
from joblib import dump, load

from text_prep import Text_prep
from bow import BoW
from fisher import Fisher_Vectorizer
from svm import SVM_Classifier

def main():
    '''
    main Pipeline to start text sentiment analysis.
    '''

    train_path = "data/train/sentiment140-train.csv"
    test_path  = "data/test/sentiment140-test.csv"
    val_path   = "data/val/sentiment140-val.csv"

    print("Starting the application...")
    train_texts, y_train = load_file(train_path)
    val_texts,   y_val   = load_file(val_path)
    test_texts,  y_test  = load_file(test_path)

    if not train_texts or not val_texts or not test_texts:
        print(f"Failed to load one or more files. Exiting the application.")
        return

    print("File content loaded successfully...")

    text_processor = Text_prep(train_texts)
    test_processor = Text_prep(test_texts)
    val_processor = Text_prep(val_texts)
    token_list = text_processor.preprocess_list()
    test_tokens = test_processor.preprocess_list()
    val_tokens = val_processor.preprocess_list()
    print("tokens were created successfully...")

    dict = text_processor.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")
    
    sorted_tokens_list = text_processor.sort_tokens(dict)
    print("Dictionary is sorted descending...")
    print(sorted_tokens_list[:10]) # print top 10 tokens (debugging)

    bow = BoW()
    word2idx, idx2word = bow.build_vocabulary(sorted_tokens_list)
    print("Vocabulary built successfully...")

    X_train = bow.vectorize_list(token_list)
    X_val   = bow.vectorize_list(val_tokens)
    X_test  = bow.vectorize_list(test_tokens)  
    print("BoW-Matrices created successfully...")
    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)

    fisher_vectorizer = Fisher_Vectorizer(alpha=1.0)
    Phi_train = fisher_vectorizer.fit_transform(X_train)
    Phi_val   = fisher_vectorizer.transform(X_val)
    Phi_test  = fisher_vectorizer.transform(X_test)
    print("Fisher-Features created successfully...")

    svm_classifier = SVM_Classifier(C=1.0, max_iter=2000)
    svm_classifier.train(Phi_train, y_train)
    print("SVM Classifier trained successfully...")

    print("Train labels:", len(y_train), "Val labels:", len(y_val), "Test labels:", len(y_test))

    svm_classifier = SVM_Classifier(C=1.0, max_iter=2000)
    print("Training SVM classifier on Fisher-Features...")
    svm_classifier.train(Phi_train, y_train)
    print("SVM classifier trained successfully...")

    # evaluate on validation and test sets
    val_acc = svm_classifier.evaluate(Phi_val, y_val)
    test_acc = svm_classifier.evaluate(Phi_test, y_test)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")

    model_bundle = {
        "bow": bow,
        "fisher": fisher_vectorizer,
        "svm": svm_classifier.model,
        "idx2word": idx2word,
    }

    dump(model_bundle, "models/fisher_svm_sentiment.joblib")
    print("Model saved to models/fisher_svm_sentiment.joblib")



def load_file(file_path):
    """
    Lädt eine CSV mit Spalten 'polarity' und 'text'
    und gibt zwei Listen zurück:
      - texts: List[str]
      - labels: List[int]
    """
    try:
        df = pd.read_csv(file_path, encoding='latin-1')

        if "polarity" not in df.columns or "text" not in df.columns:
            raise ValueError(f"'polarity' oder 'text' Spalte fehlt in {file_path}")

        df["text"] = df["text"].fillna("").astype(str)
        df["polarity"] = df["polarity"].astype(int)

        texts = df["text"].tolist()
        labels = df["polarity"].tolist()

        return texts, labels

    except Exception as e:
        print(f"An error occurred while loading the file '{file_path}': {e}")
        return [], []


if __name__ == "__main__":
    main()