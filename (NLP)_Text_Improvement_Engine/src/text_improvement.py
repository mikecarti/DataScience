from typing import List, Tuple, Any

from src.phraser import PhraseSplitter
from src.vector_db import VectorDataBase
from numpy import log
from src.generative import LLM, LLMAPI
from src.prompt import PROMPT


class TextImprovementEngine:
    def __init__(self, vector_db=VectorDataBase(), splitter=PhraseSplitter(), llm=LLM()):
        self.vector = vector_db
        self.k_nearest = 3
        self.splitter = splitter
        self.threshold = 0.63
        self.word_count_effect = 0.5
        self.norm_scaling_factor = None
        self.llm = llm
        self.prompt = PROMPT

    def improve_document(self, doc: str) -> None:
        """
        :param doc: str - some text for analysis
        :return:
        """
        sentences = doc.split(".")
        sentences = [sentence.replace("\n", " ") for sentence in sentences if len(sentence) >= 2]

        # for normalization
        sentence_word_count_max = max([sentence.count(" ") for sentence in sentences])
        self.norm_scaling_factor = log(sentence_word_count_max) ** self.word_count_effect

        suggestions = []
        for sent in sentences:
            suggestions += self._improve(sent)
        for sentence_suggestions in suggestions:
            if len(sentence_suggestions[1]) != 0:
                self._replace_phrase(sentence_suggestions)

    def _replace_phrase(self, sentence_suggestions: List[Tuple[str, List[str], List[float]]]) -> None:
        phrase, improvements, scores = sentence_suggestions

        improvements_prompt = " ".join([f"1) {i}" for i in improvements])
        model_answer = self.llm.generate(PROMPT.format(phrase=phrase, suggestions=improvements_prompt))
        improved_phrase = self._extract_suggestion_from_llm_output(model_answer)

        print(f"Phrase: {phrase}\nCandidates:")
        for improvement, score in zip(improvements, scores):
            print(f"[{score}]: {improvement}")
        print(f"\nSuggested phrase: {improved_phrase}\n", "-" * 100)

    def _extract_suggestion_from_llm_output(self, model_answer: str) -> str:
        return model_answer.split("Phrase:")[-1].split("\n")[0]

    def _improve(self, sentence: str) -> List[Tuple[str, List[str], List[float]]]:
        """
        Method for single sentences
        :param sentence: str
        :return:
        """
        phrases = self._split_by_phrases(sentence)
        suggestions = []
        for phrase in phrases:
            improvements, scores = self._find_relevant(phrase)
            suggestion = (phrase, [], [])

            for i, (improvement, score) in enumerate(zip(improvements, scores)):
                if score > self.threshold:
                    suggestion[1].append(improvement)
                    suggestion[2].append(score)
            if suggestion[1] != 0:
                suggestions.append(suggestion)
        return suggestions

    def _split_by_phrases(self, text: str) -> List[str]:
        phrases = self.splitter.split_sentence(text)
        phrases = [str(phrase) for phrase in phrases]
        # filter by phrases that are at least 2 words long
        phrases = [phrase for phrase in phrases if len(phrase.split(" ")) >= 2]

        return [text for text in phrases if text not in ('', ',')]

    def _find_relevant(self, phrase: str) -> Tuple[list[str], list[float]]:
        similar_docs = self.vector.db.similarity_search_with_relevance_scores(phrase, k=self.k_nearest)
        docs = [doc[0].page_content for doc in similar_docs]
        scores = [doc[1] for doc in similar_docs]
        scores = self._log_scale(phrase, scores)
        return docs, scores

    def _log_scale(self, phrase: str, scores: List[float]) -> List[float]:
        """
        multiply by log of words count, to fight problem of high cosine similarity
        on short sentences and low cosine similarity on big sentenes
        :param phrase: Analyzed phrase
        :param scores: List of cosine similarity scores
        :return: Rescaled list of cosine similarity scores
        """
        word_count = len(phrase.split(" "))
        scaled_scores = [(log(word_count) ** self.word_count_effect) * scr for scr in scores]
        # normalization
        return [score / self.norm_scaling_factor for score in scaled_scores]
