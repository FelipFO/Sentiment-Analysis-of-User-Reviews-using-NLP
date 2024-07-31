import subprocess

import nltk

def devdownload_dependecies():
    nltk.download("stopwords")
    nltk.download("punkt")
    subprocess.run(["spacy", "download", "en_core_web_sm"])
