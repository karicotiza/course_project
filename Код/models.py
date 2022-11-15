from abc import ABC, abstractmethod
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: str) -> str:
        raise NotImplementedError


class Question:
    question = "What industry does this company work in?"


class QuestionAnsweringModel(Model):
    @abstractmethod
    def __init__(self, path_to_model: str):
        self.model = AutoModelForQuestionAnswering.from_pretrained(path_to_model)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    @abstractmethod
    def predict(self, context: str) -> str:
        predictor = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        data = {
            "question": Question.question,
            "context": context,
        }

        prediction = predictor(data)

        return prediction.get("answer")


class MiniLM(QuestionAnsweringModel):
    def __init__(self):
        path_to_model = "C://Storage//Net//minilm-uncased-squad2"
        super(MiniLM, self).__init__(path_to_model)

    def predict(self, context: str):
        return super(MiniLM, self).predict(context)
