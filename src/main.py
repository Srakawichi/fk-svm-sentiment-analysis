# src/main.py
from text_prep import Text_prep

def main():
    '''
    main Pipeline to start text sentiment analysis.
    '''

    print("Starting the application...")
    text = load_file("src/example.txt")
    print("File content loaded:")
    print(text)
    text_processor = Text_prep(text)
    tokens = text_processor.preprocess()
    print("Preprocessed tokens:")
    print(tokens)



def load_file(file_path):
    '''
    loads and prints the content of a file given its path.
    '''

    file = open(file_path, "r")
    content = file.read()
    file.close()
    return content


if __name__ == "__main__":
    main()