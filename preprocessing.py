import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
from spacy.lang.pt.stop_words import STOP_WORDS

nltk.download('stopwords', quiet=True)

CUSTOM_STOPWORDS = {'pra', 'pro', 'vou', 'deu', 'igual'}

MENTIONS_REGEX = re.compile(r'(@[A-Za-z0-9]+)|(_[A-Za-z0-9]+)')
SINGLE_LETTER_REGEX = re.compile(r'\s+[a-zA-Z]\s+')
PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_REGEX = re.compile(r'\s+')

STOP_WORDS_SET = {
    unidecode(word).lower()
    for word in STOP_WORDS.union(stopwords.words('portuguese')).union(CUSTOM_STOPWORDS)
}


def clean_text(text: str) -> str:
    """Preprocessamento e limpeza da letra."""
    text = text.replace('<div data-plugin="googleTranslate" id="lyrics">', '').replace('</div>', '').replace('<br/>', ' ')
    text = unidecode(text).lower()
    text = MENTIONS_REGEX.sub(" ", text)
    text = PUNCTUATION_REGEX.sub("", text)
    text = WHITESPACE_REGEX.sub(" ", text)
    text = ''.join([c for c in text if not c.isdigit()])
    text = SINGLE_LETTER_REGEX.sub(" ", text)
    tokens = [word for word in text.split() if word not in STOP_WORDS_SET]
    return ' '.join(tokens)


def read_songs(filepath="input/songs.csv") -> pd.DataFrame:
    return pd.read_csv(filepath, sep=";", index_col=0)


def filter_brazilian_songs(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['is_pt'] != 'Tradução ') & (df["category"] != "Fado")].copy()


def preprocess_lyrics(df: pd.DataFrame) -> pd.Series:
    return df['lyrics'].apply(clean_text)


def remove_duplicate_words(text: str) -> str:
    """Remove palavras duplicadas mantendo a ordem."""
    return ' '.join(dict.fromkeys(text.split()))


def add_text_length(text: str) -> int:
    return len(text)


def map_category(category: str) -> str:
    if category == 'Gospel/Religioso':
        return 'Religioso'
    elif category == 'Funk Carioca':
        return 'Funk'
    return category


def filter_categories_with_min_count(df: pd.DataFrame, min_count: int = 50) -> pd.DataFrame:
    return df.groupby('category').filter(lambda x: len(x) >= min_count)


def generate_histograms(df: pd.DataFrame, output_path="histogram.png"):
    categories = df['category'].unique()
    fig = plt.figure(figsize=(10, 10))
    for i, category in enumerate(categories):
        ax = fig.add_subplot(5, 3, i + 1)
        df[df['category'] == category]['len'].hist(bins=10, ax=ax)
        ax.set_title(category)
    fig.tight_layout()
    plt.savefig(output_path)


def save_processed_data(df: pd.DataFrame, output_path='output/songs.csv'):
    df.to_csv(output_path, index=False, sep=";")


def main():
    df = read_songs()
    df = filter_brazilian_songs(df)
    df['transformed'] = preprocess_lyrics(df)
    df['reduce'] = df['transformed'].apply(remove_duplicate_words)
    df['len'] = df['reduce'].apply(add_text_length)
    df['category'] = df['category'].apply(map_category)
    df = filter_categories_with_min_count(df)
    generate_histograms(df)
    save_processed_data(df)

if __name__ == "__main__":
    main()
