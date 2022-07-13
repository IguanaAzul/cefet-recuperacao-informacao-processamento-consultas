from typing import List, Set, Mapping, Dict
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import (
    RankingModel,
    VectorRankingModel,
    IndexPreComputedVals,
    BooleanRankingModel,
    OPERATOR,
)
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner


class QueryRunner:
    def __init__(self, ranking_model: RankingModel, index: Index, cleaner: Cleaner):
        self.ranking_model = ranking_model
        self.index = index
        self.cleaner = cleaner

    def get_relevance_per_query(self) -> Dict[str, Set[int]]:
        """
        Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
        fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

        """
        dic_relevance_docs = {}
        for arquiv in ["belo_horizonte", "irlanda", "sao_paulo"]:
            with open(f"relevant_docs/{arquiv}.dat") as arq:
                dic_relevance_docs[arquiv] = set(arq.readline().split(","))
        return dic_relevance_docs

    def count_topn_relevant(
        self, n: int, respostas: List[int], doc_relevantes: Set[int]
    ) -> int:
        """
        Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
        Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
        Os documentos relevantes estão no parametro docRelevantes
        """
        return len(doc_relevantes.intersection(set(respostas[:n])))

    def compute_precision_recall(
        self, n: int, lst_docs: List[int], relevant_docs: Set[int]
    ) -> (float, float):
        interseccao = self.count_topn_relevant(n, lst_docs, relevant_docs)
        return interseccao / n, interseccao / len(relevant_docs)

    def get_query_term_occurence(self, query: str) -> Mapping[str, TermOccurrence]:
        """
        Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
        E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
        e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
        Coloque o docId como None.
        Caso o termo nao exista no indic, ele será desconsiderado.
        """
        # print(self.index)
        map_term_occur = {}
        query_pre = self.cleaner.preprocess_text(query)
        print(query_pre)
        for word in query_pre:
            try:
                map_term_occur[word] = (
                    TermOccurrence(None, self.index.get_term_id(word), 1)
                    if word not in map_term_occur
                    else TermOccurrence(
                        None,
                        self.index.get_term_id(word),
                        map_term_occur[word].term_freq + 1,
                    )
                )
            except:
                continue

        return map_term_occur

    def get_occurrence_list_per_term(
        self, terms: List
    ) -> Mapping[str, List[TermOccurrence]]:
        """
        Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
        Caso o termo nao exista, este termo possuirá uma lista vazia
        """
        return {term: self.index.get_occurrence_list(term) for term in terms}

    def get_docs_term(self, query: str) -> List[int]:
        """
        A partir do indice, retorna a lista de ids de documentos desta consulta
        usando o modelo especificado pelo atributo ranking_model
        """
        # Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
        dic_query_occur = self.get_query_term_occurence(query)

        # obtenha a lista de ocorrencia dos termos da consulta
        dic_occur_per_term_query = self.get_occurrence_list_per_term(
            self.cleaner.preprocess_text(query)
        )

        # utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
        return self.ranking_model.get_ordered_docs(
            dic_query_occur, dic_occur_per_term_query
        )

    @staticmethod
    def runQuery(
        query: str,
        indice: Index,
        cleaner: Cleaner,
        ranking_model: RankingModel,
    ):
        """
        Para um daterminada consulta `query` é extraído do indice `index` os documentos mais relevantes, considerando
        um modelo informado pelo usuário. O `indice_pre_computado` possui valores précalculados que auxiliarão na tarefa.
        Além disso, para algumas consultas, é impresso a precisão e revocação nos top 5, 10, 20 e 50. Essas consultas estão
        Especificadas em `map_relevantes` em que a chave é a consulta e o valor é o conjunto de ids de documentos relevantes
        para esta consulta.
        """
        time_checker = CheckTime()

        # PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
        # apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
        # abaixo, existem exemplos fixos.
        qr = QueryRunner(ranking_model, indice, cleaner)
        map_relevantes = qr.get_relevance_per_query()
        time_checker.print_delta("Query Creation")

        # Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
        resposta, _ = qr.get_docs_term(query)
        time_checker.print_delta("anwered with {len(respostas)} docs")

        # nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
        # se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
        # O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao
        key_query = query.lower().replace(" ", "_")
        if key_query in map_relevantes.keys():
            arr_top = [5, 10, 20, 50]
            for n in arr_top:
                precisao, revocacao = qr.compute_precision_recall(n, resposta, map_relevantes[key_query])
                print(f"Precisao {n}: {precisao}")
                print(f"Recall {n}: {revocacao}")
            print(resposta[:10])

    @staticmethod
    def main():
        index = Index()
        index = index.read("wiki.idx")

        check_time = CheckTime()
        cleaner = Cleaner(
            stop_words_file="stopwords.txt",
            language="portuguese",
            perform_stop_words_removal=True,
            perform_accents_removal=True,
            perform_stemming=False,
        )
        precomput = IndexPreComputedVals(index)
        check_time.print_delta("Precomputou valores")

        tipo = -1
        while True:
            try:
                tipo = int(
                    input(
                        "Escolha seu RankingModel, digite o número correspondente\n0 - Boolean\n1 - VectorModel\n"
                    )
                )
                if tipo != 0 and tipo != 1:
                    print("Entrada inválida, tente novamente\n")
                break
            except Exception as e:
                print("Entrada inválida, tente novamente\n", e)
                continue
        if tipo == 0:
            operator = 0
            while True:
                try:
                    operator = int(
                        input("Escolha o operador do seu RankingModel\n1 - AND\n2 - OR\n")
                    )
                    if operator != 1 and operator != 2:
                        print("Entrada inválida, tente novamente\n")
                    break
                except Exception as e:
                    print("Entrada inválida, tente novamente\n", e)
                    continue
            ranking_model = BooleanRankingModel(OPERATOR(operator))
        else:
            ranking_model = VectorRankingModel(precomput)
        query = ""
        while True:
            try:
                query = str(input("Faça sua consulta\n"))
                break
            except Exception as e:
                print("Entrada inválida, tente novamente\n", e)
                continue

        QueryRunner.runQuery(query, index, cleaner, ranking_model)
