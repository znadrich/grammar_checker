from copy import deepcopy
import numpy as np

class NGrams:
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

    def update_init_vector(self, p):
        """
        Update the initial vector counts
        
        Args:
            s (list): Sequence to analyze
        
        Returns:
            tuple: Truncated s, parts of speech
        """

        try:
            self.init_vector[p] += 1
        except KeyError:
            self.init_vector[p] = 1
    
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

    def update_transition(self, p, prvs, nxt):
        """
        Update the transition matrix counts
        
        Args:
            p (str): Next part of speech
            prvs (str): Previous sequence of part of speech
        
        Returns:
            str: Updated sequence of part of speech
        """
        key = '**'.join([prvs, p, nxt])
        try:
            self.transition[key] += 1
        except KeyError:
            self.transition[key] = 1         

    def calc_p_z(self):        
        tot = sum(self.transition.values())
        z_s = {key.split('**')[1] for key in self.transition.keys()}
        self.p_z = {z: 0 for z in z_s}
        for key, ct in self.transition.items():
            z = key.split('**')[1]
            self.p_z[z] += ct            
    
    def x_y(self, key):
        x = key.split('**')[0]
        y = key.split('**')[2]
        return x+'**'+y 

    def calc_p_x_y(self):
        tot = sum(self.transition.values())
        x_y_s = {self.x_y(key) for key in self.transition.keys()}
        self.p_x_y = {xy: 0 for xy in x_y_s}
        for key, ct in self.transition.items():
            xy = self.x_y(key)
            self.p_x_y[xy] += ct            

    def calc_p_z_given_x_y(self):
        tot = sum(self.transition.values())
        x_y_s = {self.x_y(key) for key in self.transition.keys()}
        self.p_z_given_x_y = {xy: {} for xy in x_y_s}
        for xy in self.p_x_y.keys():
            for key, ct in self.transition.items():
                if xy == self.x_y(key):
                    z = key.split('**')[1]
                    try:
                        self.p_z_given_x_y[xy][z] += ct
                    except KeyError:
                        self.p_z_given_x_y[xy][z] = ct

        for xy in self.p_z_given_x_y.keys():
            tot = sum(self.p_z_given_x_y[xy].values())
            for z in self.p_z_given_x_y[xy].keys():
                self.p_z_given_x_y[xy][z] /= tot

    def normalize_transition_matrix(self):
        """
        Normalizes the rows of the transition matrix to be probabilities
        """
        if not self.normalized_transition:
            self.calc_p_z()
            self.calc_p_x_y()
            self.calc_p_z_given_x_y()
            
            tot = sum(self.transition.values())
            probs = {key: value/tot for key, value in self.transition.items()}
            self.transition = probs            
            self.p_x_y = {xy: ct/tot for xy, ct in self.p_x_y.items()}
            self.p_z = {z: ct/tot for z, ct in self.p_z.items()}
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
                for ix, wp in enumerate(s):
                    len_s = len(s)
                    if ix == 0 or ix == len_s-1:
                        pass
                    else:
                        w, p = wp
                        prvs = s[ix-1][1]
                        nxt = s[ix+1][1]
                        self.update_transition(p=p, prvs=prvs, nxt=nxt)
                        self.update_init_vector(p=p)

        del cpy
        # self.normalize_init_vector()
        # self.normalize_transition_matrix()
    
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

    def predict_transition(self, s, ix):
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
        
        p = s[ix][1]
        prvs = s[ix-1][1]
        nxt = s[ix+1][1]
        key = '**'.join([prvs, p, nxt])
        try:
            denom = self.init_vector[p]
            self.pr.append(self.transition[key])
        except KeyError:
            self.pr.append(0)
        
        p = s[ix][0]
        prvs = s[ix-1][0]
        nxt = s[ix+1][0]
        self.words.append(' '.join([prvs, p, nxt]))

    def predict_transition_bayes(self, s, ix):
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
        
        z = s[ix][1]
        prvs = s[ix-1][1]
        nxt = s[ix+1][1]
        xy = '**'.join([prvs, nxt])
        try:
            num = self.p_z_given_x_y[xy][z] * self.p_x_y[xy]
            denom = self.p_z[z]
            self.pr.append(num/denom)
        except KeyError:
            self.pr.append(0)
        
        p = s[ix][0]
        prvs = s[ix-1][0]
        nxt = s[ix+1][0]
        self.words.append(' '.join([prvs, p, nxt]))

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
        self.pr = []
        sent_pr = []

        if len(s) > self.order + 1:
            for ix, wp in enumerate(s):
                len_s = len(s)
                if ix == 0 or ix == len_s-1:
                    pass
                else:
                    self.predict_transition_bayes(s=s, ix=ix)
    
        lwst = np.array(self.pr).argmin()
        rt_words = self.words[lwst]

        return self.pr, rt_words
        
        