import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

def get_top_label(result):
    return result[0]["label"]


if __name__ == "__main__":

    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    pipe = pipeline(
        "text-classification",
        model= model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    songs = pd.read_csv("../output/songs.csv", sep=";")

    songs["n_tokens"] = songs["transformed_without_tags"].apply(lambda x: len(tokenizer.tokenize(x)))

    songs = songs[songs["n_tokens"] <= 500].copy()

    tqdm.pandas()

    songs["sentiment_result"] = songs["transformed_without_tags"].progress_apply(lambda x: pipe(x))

    songs["sentiment"] = songs["sentiment_result"].apply(get_top_label)

    print(songs[["transformed_without_tags", "sentiment"]])
