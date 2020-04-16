import nltk
import json
import numpy as np
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize, word_tokenize

from markov import MarkovChain
from utils import flag_phrases, trigrams, trigrams_words, verb_filter

def compare_sentences(mc, s1, s2):
    text = [s1, s2]
    tokenized = [word_tokenize(s) for s in text]
    snt_pos = nltk.pos_tag_sents(tokenized)
    preds = [mc.predict_proba(s) for s in snt_pos]
    phrase_preds = [trigrams(p) for p, w in preds]
    phrases = [trigrams_words(s) for s in tokenized]
    flags = [(p, flag_phrases(p, w)) for p, w in zip(phrase_preds, phrases)]
    for s, flag in zip(text, flags):
        print(s)
        print(flag)
    print()

def load_examples(path):
    with open(path, 'r') as f:      
        data = json.load(f)  
    return data['sentences']

def main():
    # Get our training data, Brown corpus
    b_sents = brown.sents()
    b_pos = nltk.pos_tag_sents(b_sents)

    # Filter to only verb phrases
    b_verbs = verb_filter(b_pos)

    # Fit our MarkovChain
    b_mc = MarkovChain(order=1)
    b_mc.fit(b_verbs)
    b_mc.normalize_transition_matrix()

    examples = load_examples('examples.json')
    
    for ex in examples:
        compare_sentences(mc=b_mc, s1=ex['good'], s2=ex['bad'])

if __name__ == '__main__':
    main()