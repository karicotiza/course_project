from abc import ABC, abstractmethod
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class Model(ABC):
    @abstractmethod
    def __init__(self, path_to_model: str) -> None:
        self.model = AutoModelForQuestionAnswering.from_pretrained(path_to_model)
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    @abstractmethod
    def predict(self, context: str) -> str:
        raise NotImplementedError


class QuestionAnsweringModel(Model):
    def __init__(self, path_to_model: str):
        super().__init__(path_to_model)

    def predict(self, context: str):
        question = "What industry does this company work in ?"

        predictor = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        data = {
            "question": question,
            "context": context,
        }

        prediction = predictor(data)

        return prediction.get("answer")


model = QuestionAnsweringModel("C://Storage//Net//minilm-uncased-squad2")
prediction = model.predict(
    """
    Pizza Tempo is the largest chain of pizzerias in the Republic of Belarus: 
    24 cozy pizzerias are located in Minsk, Grodno, Gomel, Mogilev, Molodechno and Mozyr
    
    Tempo Pizza is the largest network of pizzeria in the Republic of Belarus: 
    24 cozy pizzeria are located in Minsk, Grodno, Gomel, Mogilev, Youth and Mozire
    """
)

print(prediction)
