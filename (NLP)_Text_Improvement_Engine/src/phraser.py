from typing import List

import benepar
import spacy


class PhraseSplitter:
    def __init__(self):
        benepar.download('benepar_en3')
        self.splitter = spacy.load('en_core_web_md')
        self.splitter.add_pipe('benepar', config={'model': 'benepar_en3'})

    def split_sentence(self, sentence: str) -> List[str]:
        """
        Produces list of splitted sentence by phrases
        :param sentence: Sentence
        :return: List of sentences
        """
        sentence = sentence.replace("\n", "").replace("  ", " ")
        sentence = sentence.strip()

        doc = self.splitter(sentence)
        sent = list(doc.sents)[0]
        return list(sent._.children)
