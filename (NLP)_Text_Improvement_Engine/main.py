from src.generative import LLM
from src.text_improvement import TextImprovementEngine

if __name__ == "__main__":
    engine = TextImprovementEngine()
    text_path = input("Welcome to the Text Improvement Engine! Input the path to text file: ")
    with open(text_path, mode="r") as f:
        sample_text = f.read()

    engine.improve_document(sample_text)
