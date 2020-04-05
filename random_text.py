import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

def random_text():
    t = requests.get('https://randomtextgenerator.com/').text
    soup = BeautifulSoup(t, 'html.parser')
    paragraph = soup.body.find('textarea').contents[0]
    sentences = sent_tokenize(paragraph)
    return sentences