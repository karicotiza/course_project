import requests

URLS = [
    "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/config.json",
    "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/flax_model.msgpack",
    "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/pytorch_model.bin",
    "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/special_tokens_map.json",
    "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/tokenizer_config.json",
    "https://huggingface.co/deepset/minilm-uncased-squad2/resolve/main/training_args.bin",
    "https://huggingface.co/deepset/minilm-uncased-squad2/raw/main/vocab.txt",
]


def download(url_: str) -> None:
    file_name = url_.split("/")[-1]

    request = requests.get(
        url_
    )

    with open(file_name, "wb") as file:
        file.write(request.content)

    print(f"File: {file_name} | Result {request.status_code}")


for url in URLS:
    download(url)
