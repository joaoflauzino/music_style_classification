import pandas as pd
from huggingface_hub import InferenceClient
import re


def read_categories_examples(path: str = "output/songs.csv") -> dict:
    songs = pd.read_csv(path, sep=";")
    categories = songs['category'].unique().tolist()
    mode = get_avg_wordcount_by_genre()
    examples = {}
    for idx, cat in enumerate(categories):
        example = songs[songs['category'] == cat].iloc[0]['transformed']
        examples[f"Song {idx}"] = {"category": cat, "lyric": example[:300], "Mode": mode.get(cat)}
    return examples


def get_examples_to_classify(path: str = "output/songs.csv") -> tuple[dict, dict]:
    songs = pd.read_csv(path, sep=";")
    categories = songs['category'].unique().tolist()
    test_examples = {}
    labels = {}
    for idx, category in enumerate(categories):
        row = songs[songs['category'] == category].iloc[1]  # a segunda música de cada categoria
        lyric = row['transformed'][:500]
        test_examples[f"Song {idx}"] = lyric
        labels[f"Song {idx}"] = category
    return test_examples, labels


def parse_llm_response(response_text: str) -> dict:
    predictions = {}
    for line in response_text.strip().split("\n"):
        match = re.match(r"Song (\d+):\s*(.+)", line.strip())
        if match:
            song_id = f"Song {match.group(1)}"
            predicted_genre = match.group(2).strip()
            predictions[song_id] = predicted_genre
    return predictions


def get_avg_wordcount_by_genre(path: str = "output/songs.csv") -> dict:
    songs = pd.read_csv(path, sep=";")
    avg_len =  songs.groupby('category')['len'].agg(lambda x: x.mode().iloc[0]).astype(int)
    return dict(avg_len)


if __name__ == "__main__":

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #model_id = "google/gemma-3-27b-it"
    client = InferenceClient(model_id)

    train_examples = read_categories_examples()
    test_examples, ground_truth = get_examples_to_classify()

    prompt_usuario = f"""
    Você deve escolher **apenas UM** gênero musical que melhor representa **cada** letra. 
    Você tem essas opções: Axé, Black Music, Bossa Nova, Folk, Forró, Funk,
       Religioso, Infantil, Jovem Guarda, MPB, Pagode, Pop,
       Rap, Reggae

    Responda exatamente no seguinte formato (sem explicações ou variações):
    Song X: gênero

    Exemplo:
    Song 0: Samba
    Song 1: MPB
    ...

    Agora classifique as seguintes letras:\n
    {test_examples}
    """

    messages = [
        {
            "role": "system",
            "content": (
                        "Você é um especialista em música brasileira."
                        "Sua tarefa é identificar o gênero musical de letras de músicas."
                        "Use como referência os exemplos abaixo, considerando a letra e o número médio de palavras distintas por gênero (moda)."
                        "Sempre retorne **apenas UM gênero** por letra.\n\n"
                        f"{train_examples}"
                        )
        },
        {"role": "user", "content": prompt_usuario}
    ]

    response = client.chat.completions.create(
        messages=messages,
        temperature=1
    )

    llm_output = response.choices[0].message.content
    print("Resposta da LLM:\n", llm_output)

    predictions = parse_llm_response(llm_output)

    print("\nComparação LLM x Ground Truth:")
    correct = 0
    total = len(ground_truth)

    for song_id, true_label in ground_truth.items():
        pred_label = predictions.get(song_id, "N/A")
        status = "✅" if pred_label.lower() == true_label.lower() else "❌"
        print(f"{song_id}: Predito = {pred_label}, Real = {true_label} {status}")
        if status == "✅":
            correct += 1

    print(f"\nAcurácia: {correct}/{total} = {correct/total:.2%}")


## Analise de sentimento das músicas