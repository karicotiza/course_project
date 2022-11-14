from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


class Model:
    def __init__(self, path: str):
        self.model = AutoModelForQuestionAnswering.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def predict(self, context: str):
        question = "What industry does this company work in ?"

        predictor = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
        data = {
            "question": question,
            "context": context,
        }

        prediction = predictor(data)

        return prediction.get("answer")


model = Model("C://Storage//Net//minilm-uncased-squad2")
prediction = model.predict(
    """
    Pizza Tempo is the largest chain of pizzerias in the Republic of Belarus: 
    24 cozy pizzerias are located in Minsk, Grodno, Gomel, Mogilev, Molodechno and Mozyr
    """
)

print(prediction)
