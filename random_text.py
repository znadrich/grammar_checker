import requests
from bs4 import BeautifulSoup

def random_text():
    t = requests.get('https://randomtextgenerator.com/').text
    soup = BeautifulSoup(t, 'html.parser')
    return soup.body.find('textarea').contents[0]