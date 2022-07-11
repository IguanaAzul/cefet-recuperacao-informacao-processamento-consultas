from query.ranking_models import VectorRankingModel, BooleanRankingModel, OPERATOR
from index.structure import Index

if __name__ == "main":
    index = Index()
    index = index.read("wiki.idx")

    tipo = -1
    while True:
        try:
            tipo = int(input("Escolha seu RankingModel, digite o número correspondente\n0 - Boolean\n1 - VectorModel"))
            if tipo != 0 and tipo != 1:
                print("Entrada inválida, tente novamente")
            break
        except Exception as e:
            print("Entrada inválida, tente novamente", e)
            continue
    if tipo == 0:
        operator = 0
        while True:
            try:
                operator = int(input("Escolha o operador do seu RankingModel\n1 - AND\n2 - OR"))
                if operator != 1 and operator != 2:
                    print("Entrada inválida, tente novamente")
                break
            except Exception as e:
                print("Entrada inválida, tente novamente", e)
                continue
        ranking_model = BooleanRankingModel(OPERATOR(operator))
    else:
        ranking_model = VectorRankingModel(index)

    while True:
        try:
            query = str(input("Faça sua consulta"))
            break
        except Exception as e:
            print("Entrada inválida, tente novamente", e)
            continue


