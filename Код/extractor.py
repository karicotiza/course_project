import models


class Extractor:
    def __init__(self):
        self.__ru_en = models.WMT19RuEn()
        self.__en_ru = models.OpusMTEnRu()
        self.__mini_lm = models.MiniLM()

    def extract_industry(self, text: str) -> str:
        english = self.__ru_en.predict(text)
        industry = self.__mini_lm.predict(english)
        return self.__en_ru.predict(industry)