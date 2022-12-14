from extractor import Extractor

if __name__ == "__main__":
    extractor = Extractor()

    texts = {
        "Коммунарка": """
            СОАО «Коммунарка» — одна из крупнейших кондитерских фабрик кондитерской отрасли в Белоруссии.
        """,
        "Пицца Темпо": """
            Пицца Темпо – крупнейшая сеть пиццерий в Республике Беларусь: 
            24 уютные пиццерии расположились в городе Минске, Гродно, Гомеле, 
            Могилеве, Молодечно и Мозыре
        """,
        "Савушкин": """
            Компания «Савушкин продукт» – лидер молочной отрасли Беларуси 
            и один из ведущих производителей натуральной молочной продукции 
            Восточноевропейского региона.
        """,
        "Белита-Витэкс": """
            Сегодня ЗАО «ВИТЭКС» и СП «БЕЛИТА» ООО — признанные лидеры по 
            производству отечественной косметики, которые производят самый 
            широкий ассортимент косметической продукции среди отечественных 
            производителей – более 2500 видов, в том числе
        """,
        "БеларусьКалий": """
            Беларуськалий — белорусское предприятие-производитель калийных 
            минеральных удобрений; открытое акционерное общество. Расположено 
            в городе Солигорск в Минской области. Беларуськалий находится на 
            втором месте в СНГ по объёмам производства.
        """,
        "Динамо Минск": """
            Хоккейный клуб «Динамо-Минск» — это не только один из сильнейший 
            спортивных клубов Республики Беларусь, но и молодой, активно 
            развивающийся коллектив единомышленников. Поэтому мы всегда находимся 
            в поиске, в том числе молодых интересных специалистов, которым готовы 
            предложить достойные условия труда.
        """,
        "МинскХлебПром": """
            КУП "Минскхлебпром" - крупнейшее хлебопекарное предприятие Беларуси. 
            Оно состоит из шести территориально обособленных подразделений. Среднесуточная 
            мощность производства хлебобулочных изделий составляет 255 тонн, 
            кондитерских изделий - около 16 тонн, более 8 тонн продукции оборонного 
            назначения, порядка 10 тонн имбирных пряников и печенья. 
        """
    }

    for key, value in texts.items():
        print(f"{key}: {extractor.extract_industry(value).lower()}")
