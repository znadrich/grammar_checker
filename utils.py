import numpy as np

def verb_filter(pos):
    """
    Filters parts-of-speech tuples (words, pos) from `nltk.pos_tag_sents` to only
    include phrases of the form word-verb-word.
    
    Args:
        pos (list): List of tuples output by `nltk.pos_tag_sents`
    
    Returns:
        List: List of (word, pos) tuples, that are phrases of the form word-verb-word
    """
    verbs = []
    for pos in pos:
        for ix, wp in enumerate(pos):
            w, p = wp
            if p[0] == 'V':
                # if we're not out of range
                try:
                    phrase = [pos[ix-1], pos[ix], pos[ix+1]]
                    verbs.append(phrase)
                except Exception:
                    pass
    return verbs

def trigrams(p):
    """
    Turn 3 sequential Markov chain predictions into probability of a phrase
    
    Args:
        p (list): List of Markov chain predictions
    
    Returns:
        list: List of phrase probabilities
    """
    preds = []
    for ix, p_ix in enumerate(p):
        if ix == 0 or ix == len(p)-1:
            pass
        else:
            pred = p_ix * p[ix+1]
            preds.append(pred)
    return preds

def trigrams_words(s):
    """
    Turn a sentence into a series of trigram phrases
    
    Args:
        s (list): Tokenized sentence
    
    Returns:
        list: List of trigram phrases
    """
    words = []
    for ix, w_ix in enumerate(s):
        if ix == 0 or ix == len(s)-1:
            pass
        else:
            phrase = [s[ix-1], w_ix, s[ix+1]]
            phrase = ' '.join(phrase)
            words.append(phrase)
    return words

def flag_phrases(pr, words, thresh=.02):
    """
    Flag phrases that have a low probability of occuring, according to a Markov chain
    
    Args:
        pr (list): Phrase predictions
        words (list): Phrases
        thresh (float, optional): Flagging threshold. Defaults to .02.
    
    Returns:
        [type]: [description]
    """
    lwst = np.where((np.array(pr) < thresh) & (np.array(pr) > 0))
    rt_words = [words[l] for l in lwst[0]]
    return rt_words