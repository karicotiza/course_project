import requests
import zipfile
import os


URLS = [
    "https://storage.googleapis.com/kaggle-data-sets/848918/1448164/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA"
    "-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221115%2Fauto%2Fstorage"
    "%2Fgoog4_request&X-Goog-Date=20221115T140803Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature"
    "=57665169989a412e3760c0b8e47956c9ea95d34e8e167bf23cea8ef62d20e1bcad27bc8bbc0b98ae3a35df783b562e21b9b3c536380141"
    "51768b27cc97beb037f9a02de9dd0589be4de172aa87337a5c77c9effe3dc1b2e5c1fdf35c24273bbd0b901403d8e77368181fe885291e2e"
    "1cc9c04aa887cb895153c2a82e5b3af7b215cb0377b0b636569352f59c66b06a2df777ed627921d5da4f888970d2aad50338278a0c16e4f8"
    "d8d440c865217cc37ac73fef598bed16faf0865a8293ae3b89768c1dc75cafd86846abe7bae20d31d376aa73090b0be2c1196e309c479c39"
    "ce0f8a42d225cc85848acb255757f6352f3e9f47ec23aad1758b32e3dee7c216c0",
    "https://storage.googleapis.com/kaggle-data-sets/189687/423331/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA2"
    "56&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221115%2Fauto%2Fstorage%2Fgoog4_"
    "request&X-Goog-Date=20221115T140839Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=a627f5f505"
    "120ca2021c0a05ee27e8312f67ea9192a29209a6f89f616e9aff55d1091b03e6f23ae351eaee1b6d108f566ab3c21e4b0ff896bb5211d021"
    "e22e82db47ccc5fac841c743b8af812556a76b29127e20d7cd92cb1b35c770d31d05b32d236e752704296f1f7c32ea8819d6e0e82b699b10"
    "276129063f26518fa47d0a2acfa9544a92a4d7e9f00d3cd3aadf49cc9fc5e5afe30134a61c722b136cf009bac1de4fa5dec42e8990c758c9"
    "a68fd97a1bd9d87023f1446c7f8c6c77c125e67d2fcdbb94a16fffa4bf14cf04c96553c9381cc68519419bc1fc0dc20ca8e448b568ce257f"
    "95d7248cc98772c078ffbb826a7fef5376da1e3329c641d534b140",
]


def download(url_: str) -> None:
    file_name = url_.split("/")[5]

    request = requests.get(
        url_
    )

    with open(file_name + ".zip", "wb") as file:
        file.write(request.content)

    with zipfile.ZipFile(file_name + ".zip", 'r') as zip:
        zip.extractall()

    os.remove(file_name + ".zip")

    print(f"File: {file_name} | Result {request.status_code}")


for url in URLS:
    download(url)
