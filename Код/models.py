import config
import transformers

from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: str) -> str:
        raise NotImplementedError


class WMT19RuEn(Model):
    def __init__(self):
        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.wmt19_ru_en)
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(config.wmt19_ru_en)

    def predict(self, context: str):
        translator = transformers.pipeline("translation", model=self.__model, tokenizer=self.__tokenizer)
        translate = translator(context)

        return translate[0].get("translation_text")


class OpusMTEnRu(Model):
    def __init__(self):
        self.__model = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.opus_mt_en_ru)
        self.__tokenizer = transformers.AutoTokenizer.from_pretrained(config.opus_mt_en_ru)

    def predict(self, context: str):
        translator = transformers.pipeline("translation", model=self.__model, tokenizer=self.__tokenizer)
        translate = translator(context)

        return translate[0].get("translation_text")


class MiniLM(Model):
    def __init__(self):
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(config.mini_lm_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.mini_lm_path)

    def predict(self, context: str):
        predictor = transformers.pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        data = {
            "question": config.question,
            "context": context,
        }

        prediction = predictor(data)

        return prediction.get("answer")
