from indexer import HTMLIndexer, Cleaner
from structure import FileIndex

if __name__ == "__main__":
    HTMLIndexer.cleaner = Cleaner(
        stop_words_file="stopwords.txt",
        language="portuguese",
        perform_stop_words_removal=True,
        perform_accents_removal=True,
        perform_stemming=False,
    )
    obj_index = FileIndex()
    html_indexer = HTMLIndexer(obj_index)
    html_indexer.index_text_dir("./wiki")
