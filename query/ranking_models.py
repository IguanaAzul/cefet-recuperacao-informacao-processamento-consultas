from typing import List
from abc import abstractmethod
from typing import List, Set, Mapping
from index.structure import TermOccurrence
import math
from enum import Enum


class IndexPreComputedVals:
    def __init__(self, index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        self.doc_count = self.index.document_count
        for key, entry in self.index.dic_index.items():
            occ_list = self.index.get_occurrence_list(key)
            for occ in occ_list:
                # print(occ)
                tfxidf = (
                    VectorRankingModel.tf_idf(
                        self.index.document_count, occ.term_freq, len(occ_list)
                    )
                    ** 2
                )
                self.document_norm[occ.doc_id] = (
                    self.document_norm[occ.doc_id] + tfxidf
                    if occ.doc_id in self.document_norm
                    else tfxidf
                )
        for doc in self.document_norm.keys():
            self.document_norm[doc] = self.document_norm[doc] ** 0.5


class RankingModel:
    @abstractmethod
    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]],
    ) -> (List[int], Mapping[int, float]):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método"
        )

    def rank_document_ids(self, documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key=lambda x: -documents_weight[x])
        return doc_ids


class OPERATOR(Enum):
    AND = 1
    OR = 2


# Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self, operator: OPERATOR):
        self.operator = operator

    def intersection_all(
        self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> List[int]:
        print(f"map_occ: {map_lst_occurrences}")
        if not map_lst_occurrences:
            return []
        first_term = list(map_lst_occurrences)[0]
        first_list = map_lst_occurrences[first_term]
        list_id = []
        for term_occ in first_list:
            list_id.append(term_occ.doc_id)
        set_ids = set(list_id)
        for term, lst_occurrences in map_lst_occurrences.items():
            # print(lst_occurrences)
            occ_list = []
            for term_occ in lst_occurrences:
                occ_list.append(term_occ.doc_id)
            set_ids = set_ids.intersection(set(occ_list))
        print(f"interseção: {set_ids}")
        return list(set_ids)

    def union_all(
        self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]
    ) -> List[int]:
        print(f"map_occ: {map_lst_occurrences}")
        if not map_lst_occurrences:
            return []
        first_term = list(map_lst_occurrences)[0]
        first_list = map_lst_occurrences[first_term]
        list_id = []
        for term_occ in first_list:
            list_id.append(term_occ.doc_id)
        set_ids = set(list_id)
        for term, lst_occurrences in map_lst_occurrences.items():
            occ_list = []
            for term_occ in lst_occurrences:
                occ_list.append(term_occ.doc_id)
            set_ids = set_ids.union(set(occ_list))
        print(f"união: {set_ids}")
        return list(set_ids)

    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        map_lst_occurrences: Mapping[str, List[TermOccurrence]],
    ) -> (List[int], Mapping[int, float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences), None
        else:
            return self.union_all(map_lst_occurrences), None


# Atividade 2
class VectorRankingModel(RankingModel):
    def __init__(self, idx_pre_comp_vals: IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term: int) -> float:
        return 1 + math.log2(freq_term) if freq_term > 0 else 0.0

    @staticmethod
    def idf(doc_count: int, num_docs_with_term: int) -> float:
        return math.log2(doc_count / num_docs_with_term)

    @staticmethod
    def tf_idf(doc_count: int, freq_term: int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        # print(f"TF:{tf} IDF:{idf} n_i: {num_docs_with_term} N: {doc_count}")
        return tf * idf

    def get_ordered_docs(
        self,
        query: Mapping[str, TermOccurrence],
        docs_occur_per_term: Mapping[str, List[TermOccurrence]],
    ) -> (List[int], Mapping[int, float]):
        documents_weight = {}

        for term, occ_list in docs_occur_per_term.items():
            if term not in query:
                continue
            wquery = self.tf_idf(
                self.idx_pre_comp_vals.doc_count, query[term].term_freq, len(occ_list)
            )
            for occ in occ_list:
                wdoc = self.tf_idf(
                    self.idx_pre_comp_vals.doc_count, occ.term_freq, len(occ_list)
                )
                documents_weight[occ.doc_id] = (
                    wdoc * wquery / self.idx_pre_comp_vals.document_norm[occ.doc_id]
                    if occ.doc_id not in documents_weight
                    else documents_weight[occ.doc_id]
                    + wdoc * wquery / self.idx_pre_comp_vals.document_norm[occ.doc_id]
                )
        # for key, value in documents_weight.items():
        #     documents_weight[key] = value / self.idx_pre_comp_vals.document_norm[key]
        return self.rank_document_ids(documents_weight), documents_weight
