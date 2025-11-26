# src/main.py
import pandas as pd
from text_prep import Text_prep

def main():
    '''
    main Pipeline to start text sentiment analysis.
    '''

    print("Starting the application...")
    text_list = load_file("data/train/sentiment140-train.csv")
    print("File content loaded successfully...")
    text_processor = Text_prep(text_list)
    token_list = text_processor.preprocess_list()
    print("tokens were created successfully...")
    dict = text_processor.count_tokens_frequency(token_list)
    print("Token frequency counted successfully...")
    sorted_tokens_list = text_processor.sort_tokens(dict)
    print("Dictionary is sorted descending...")
    index2word = text_processor.index_to_word(sorted_tokens_list)
    word2index = text_processor.word_to_index(sorted_tokens_list)
    print("word/index lists created successfully...")
    print(sorted_tokens_list[:100])



def load_file(file_path):
    '''
    loads and prints the content of a file given its path.
    '''

    df = pd.read_csv(file_path,
                names=['polarity', 'text'],
                encoding='latin-1')

    content = df["text"].fillna("").astype(str).tolist()
    return content




if __name__ == "__main__":
    main()