from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
from collections import Counter
import os
from tqdm import tqdm


class Cleaner:
    def __init__(
        self,
        stop_words_file: str,
        language: str,
        perform_stop_words_removal: bool,
        perform_accents_removal: bool,
        perform_stemming: bool,
    ):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        # altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = {i: j for i, j in zip(in_table, out_table)}
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        soup = BeautifulSoup(html_doc, "html.parser")
        return soup.get_text()

    @staticmethod
    def read_stop_words(str_file) -> set:
        set_stop_words = set()
        with open(str_file, encoding="utf-8") as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        if self.perform_stop_words_removal:
            return term in self.set_stop_words
        else:
            return False

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        if self.perform_accents_removal:
            for i, j in self.accents_translation_table.items():
                term = term.replace(i, j)
            return term
        else:
            return term

    def preprocess_text(self, text: str) -> str or None:
        words = list()
        for word in word_tokenize(text.lower()):
            if self.is_stop_word(word) or word in self.set_punctuation:
                continue
            words.append(
                self.word_stem(self.remove_accents(word))
                if self.perform_stemming
                else self.remove_accents(word)
            )
        return words


class HTMLIndexer:
    cleaner = Cleaner(
        stop_words_file="./stopwords.txt",
        language="portuguese",
        perform_stop_words_removal=True,
        perform_accents_removal=True,
        perform_stemming=True,
    )

    def __init__(self, index):
        self.index = index

    def text_word_count(self, plain_text: str):
        dic_word_count = dict(Counter(self.cleaner.preprocess_text(plain_text)))
        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        for term, term_freq in self.text_word_count(
            self.cleaner.html_to_plain_text(text_html)
        ).items():
            self.index.index(term, doc_id, term_freq)

    def index_text_dir(self, path: str):
        for str_sub_dir in tqdm(os.listdir(path)):
            path_sub_dir = f"{path}/{str_sub_dir}"
            for filename in os.listdir(path_sub_dir):
                idx, format = filename.split(".")
                if format == "html":
                    filepath = f"{path_sub_dir}/{filename}"
                    with open(filepath, "r") as file:
                        html = file.read()
                        self.index_text(idx, html)
        self.index.finish_indexing()
