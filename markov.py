from copy import deepcopy
import numpy as np

class MarkovChain:
    """
    Class to fit and apply Markov Chain transition matricies to parts of speech lists
    
    Args:
        order (int, optional): Order of the Markov Chain. Defaults to 1.
    """
    def __init__(self, order=1):
        self.order = order
        self.transition = {}
        self.init_vector = {}
        self.normalized_init = False
        self.normalized_transition = False
    
    def init_p(self, s):
        """
        Retrieves the first entry in the chain
        
        Args:
            s (list): Sequence to analyze
        
        Returns:
            tuple: Truncated s, parts of speech, word
        """
        s = deepcopy(s)
        w_l = []
        p_l = []

        for i in range(self.order):
            w_i, p_i = s.pop(0)
            w_l.append(w_i)
            p_l.append(p_i)

        p = '**'.join(p_l)
        w = ' '.join(w_l)

        return s, p, w

    def update_init_vector(self, s):
        """
        Update the initial vector counts
        
        Args:
            s (list): Sequence to analyze
        
        Returns:
            tuple: Truncated s, parts of speech
        """
        s, p, w = self.init_p(s=s)

        try:
            self.init_vector[p] += 1
        except KeyError:
            self.init_vector[p] = 1

        return s, p
    
    def update_prvs(self, prvs, new, delim='**'):
        """
        Update the "previous" sequence of order `self.order` to analyze
        
        Args:
            prvs (str): Previous sequence
            new (str): New term to append
            delim (str, optional): How `prvs` is delimited. Defaults to '**'.
        
        Returns:
            str: Updated previous sequence
        """
        prvs = prvs.split(delim)[1:]
        prvs.append(new)
        prvs = delim.join(prvs)
        return prvs

    def update_transition(self, p, prvs):
        """
        Update the transition matrix counts
        
        Args:
            p (str): Next part of speech
            prvs (str): Previous sequence of part of speech
        
        Returns:
            str: Updated sequence of part of speech
        """
        try:
            self.transition[prvs][p] += 1
        except KeyError:
            if prvs in self.transition.keys():                
                self.transition[prvs][p] = 1
            else:
                self.transition[prvs] = {p: 1}
        
        if self.order > 1:
            prvs = self.update_prvs(prvs=prvs, new=p, delim='**')
        else:
            prvs = p
            
        return prvs

    def normalize_transition_matrix(self):
        """
        Normalizes the rows of the transition matrix to be probabilities
        """
        if not self.normalized_transition:
            probs = {}
            for p, d in self.transition.items():
                tot = sum(d.values())
                probs[p] = {p2: v/tot for p2, v in d.items()}
            self.transition = probs
        else:
            AttributeError('Transition matrix already normalized')

    def normalize_init_vector(self):
        """
        Normalizes the initial vector to be probabilities
        """
        if not self.normalized_transition:
            init = self.init_vector
            self.init_vector = {p: v/sum(init.values()) for p, v in init.items()}
        else:
            AttributeError('Initial vector already normalized')

    def fit(self, pos):
        """
        Fit a transition matrix based on a given parts of speech corpus
        
        Args:
            pos (list): List of lists, pos tagged by NLTK pos tagger
        """
        cpy = deepcopy(pos)
        for s in cpy:
            if len(s) > self.order + 1:
                s, prvs = self.update_init_vector(s=s)
                for w, p in s:
                    prvs = self.update_transition(p=p, prvs=prvs)

        del cpy
        self.normalize_init_vector()
        self.normalize_transition_matrix()
    
    def predict_init_vector(self, s):
        """
        Apply the fitted initial vector to the input sequence
        
        Args:
            s (list): Sequence to analyze
        
        Returns:
            tuple: Truncated s, initial parts of speech, initial word
        """
        self.pr = []
        s, p, w = self.init_p(s=s)
        try:
            self.pr.append(self.init_vector[p])
        except KeyError:
            self.pr.append(0)
            
        self.words.append(w)
        
        return s, p, w

    def predict_transition(self, w, p, prvs, prvs_w):
        """
        Apply the fitted transition matrix to the input sequence 
        
        Args:
            w (str): Word
            p (str): Parts of speech
            prvs (str): Previous sequence of POS
            prvs_w (str): Previous sequence of words
        
        Returns:
            tuple: New sequence of POS, New sequence of words
        """
        try:
            self.pr.append(self.transition[prvs][p])
        except KeyError:
            self.pr.append(0)

        if self.order > 1:
            prvs = self.update_prvs(prvs=prvs, new=p, delim='**')
            prvs_w = self.update_prvs(prvs=prvs_w, new=w, delim=' ')
        else:
            prvs = p
            prvs_w = w
        
        self.words.append(prvs_w)

        return prvs, prvs_w

    def predict_proba(self, pos):
        """
        Predict the probabilities of a sequence of parts of speech
        
        Args:
            pos (list): List of (POS, word) tuples
        
        Returns:
            tuple: Probabilities of the sequence, words corresponding the lowest probability
        """
        s = deepcopy(pos)
        self.words = []
        sent_pr = []
        rt_words = []

        if len(s) > self.order + 1:
            s, prvs, prvs_w = self.predict_init_vector(s=s)
            
            for w, p in s:
                prvs, prvs_w = self.predict_transition(w=w, p=p, prvs=prvs, prvs_w=prvs_w)
        
            lwst = np.array(self.pr).argmin()
            rt_words.append(self.words[lwst])

        return self.pr, rt_words
        
        