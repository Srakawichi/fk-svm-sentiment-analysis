# src/test_model.py
import sys
from joblib import load
from text_prep import Text_prep
from scipy.sparse import csr_matrix

MODEL_PATH = "models/fisher_svm_sentiment.joblib"


def load_model(model_path=MODEL_PATH):
    """
    Lädt das gespeicherte Modell-Bundle:
    - bow: BoW-Instanz mit Vokabular (word2idx, idx2word)
    - plsa: trainiertes PLSA-Modell (P(w|z), P(z|d) vom Training)
    - fisher: Fisher_Vectorizer mit theta und I_diag (auf Topic-Raum trainiert)
    - svm: trainiertes SVM-Modell
    """
    try:
        bundle = load(model_path)
    except Exception as e:
        print(f"Error loading model from '{model_path}': {e}")
        sys.exit(1)

    bow = bundle["bow"]
    plsa = bundle["plsa"]
    fisher = bundle["fisher"]
    svm_model = bundle["svm"]

    return bow, plsa, fisher, svm_model


def predict_text(bow, plsa, fisher, svm_model, text: str) -> int:
    """
    Nimmt einen Roh-Text, wendet die gleiche Preprocessing-Pipeline an
    wie beim Training, und gibt das vorhergesagte Label (0/1) zurück.

    Pipeline:
      Text -> Preprocessing/Tokenisierung
           -> BoW-Vektor
           -> PLSA: P(z|d) für diesen Text
           -> Fisher-Features im Topic-Raum
           -> SVM-Prediction
    """

    # 1) Preprocessing + Tokenizing (Text_prep wie im Training benutzen)
    processor = Text_prep([text])
    token_list = processor.preprocess_list()  # Liste mit genau einem Eintrag
    tokens = token_list[0]

    # 2) BoW-Vektor für diesen einen Text (gleiche Vokabel wie im Training)
    X_user = bow.vectorize_list([tokens])  # csr_matrix mit Shape (1, vocab_size)

    # 3) PLSA: Doc-Topic-Verteilung P(z|d) für diesen Text
    #    plsa.transform erwartet eine csr_matrix mit dem gleichen Vokabular
    topic_user = plsa.transform(X_user)      # np.array mit Shape (1, num_topics)

    # 4) In Sparse-Form bringen, damit Fisher_Vectorizer wie im Training arbeitet
    T_user = csr_matrix(topic_user.astype("float32"))  # (1, num_topics)

    # 5) Fisher-Features im Topic-Raum
    Phi_user = fisher.transform(T_user)     # csr_matrix (1, num_topics)

    # 6) SVM-Prediction
    pred = svm_model.predict(Phi_user)[0]   # einzelnes Label (0 oder 1)

    return pred


def interactive_loop():
    """
    Lädt das Modell und startet eine interaktive Eingabe-Schleife,
    in der der User Texte eingibt und Sentiment-Vorhersagen bekommt.
    """
    print(f"Lade Modell aus '{MODEL_PATH}' ...")
    bow, plsa, fisher, svm_model = load_model()
    print("Modell erfolgreich geladen.\n")

    print("Interaktiver Sentiment-Test.")
    print("Gib einen Text ein und drücke Enter.")
    print("Tippe 'quit' oder 'exit', um zu beenden.\n")

    while True:
        try:
            user_text = input("Text: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBeende.")
            break

        if user_text.strip().lower() in ("quit", "exit"):
            print("Beende.")
            break

        if not user_text.strip():
            print("Bitte keinen leeren Text eingeben.\n")
            continue

        pred = predict_text(bow, plsa, fisher, svm_model, user_text)
        label = "NEGATIV" if pred == 0 else "POSITIV"

        print(f"-> Prediction: {label}\n")


if __name__ == "__main__":
    interactive_loop()