import requests
import os


def download_file(model_name: str, url_: str) -> None:
    file_name = url_.split("/")[-1]

    request = requests.get(
        url_
    )

    with open(model_name + "//" + file_name, "wb") as file:
        file.write(request.content)

    print(f"File: {file_name} | Result {request.status_code}")


def download_model(model_name: str, urls: tuple) -> None:
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    for url in urls:
        download_file(model_name, url)


if __name__ == "__main__":
    download_model(
        "minilm-uncased-squad2",
        (
            "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/config.json",
            "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/flax_model.msgpack",
            "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/pytorch_model.bin",
            "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/special_tokens_map.json",
            "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/tokenizer_config.json",
            "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/training_args.bin",
            "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/vocab.txt",
        )
    )

    download_model(
        "wmt19-ru-en",
        (
            "https://huggingface.co/facebook/wmt19-ru-en/raw/main/config.json",
            "https://huggingface.co/facebook/wmt19-ru-en/raw/main/merges.txt",
            "https://huggingface.co/facebook/wmt19-ru-en/resolve/main/pytorch_model.bin",
            "https://huggingface.co/facebook/wmt19-ru-en/raw/main/tokenizer_config.json",
            "https://huggingface.co/facebook/wmt19-ru-en/raw/main/vocab-src.json",
            "https://huggingface.co/facebook/wmt19-ru-en/raw/main/vocab-tgt.json",
        )
    )

    download_model(
        "wmt19-en-ru",
        (
            "https://huggingface.co/facebook/wmt19-en-ru/raw/main/config.json",
            "https://huggingface.co/facebook/wmt19-en-ru/raw/main/merges.txt",
            "https://huggingface.co/facebook/wmt19-en-ru/resolve/main/pytorch_model.bin",
            "https://huggingface.co/facebook/wmt19-en-ru/raw/main/tokenizer_config.json",
            "https://huggingface.co/facebook/wmt19-en-ru/raw/main/vocab-src.json",
            "https://huggingface.co/facebook/wmt19-en-ru/raw/main/vocab-tgt.json",
        )
    )
