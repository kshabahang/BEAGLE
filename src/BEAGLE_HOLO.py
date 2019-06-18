from GeneralVSAModel import Model
import numpy as np
import torch
from copy import deepcopy

def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

class BEAGLE_HOLO(Model):
    def __init__(self, params, hparams, QaAs=None):
        super(BEAGLE_HOLO, self).__init__(hparams, QaAs)

        self.params = params

        ###initialize environment vectors
        self.E = []
        ###initialize context vectors
        self.C = []
        ###initialize order vectors
        self.O = []
        self.N      = hparams["NFEATs"]
        self.K      = hparams["NGRAMS"]
        self.SD     = 1.0/np.sqrt(self.N)
        self.bind = cconv
        self.PHI  = np.random.normal(0.0, self.SD, self.N)

    def update_vocab(self, corpus_process):
        N  = self.N
        SD = self.SD

        for i in xrange(len(corpus_process)):
            if corpus_process[i] in self.I:
                self.wf[corpus_process[i]] += 1
            elif corpus_process[i] != '_':
                self.I[corpus_process[i]] = len(self.vocab)
                self.vocab.append(corpus_process[i])
                self.wf[corpus_process[i]] = 1
                ###construct environment vector
                self.E.append(np.random.normal(0.0, SD, N))
                self.C.append(np.zeros(N))
                self.O.append(np.zeros(N))
        self.V = len(self.vocab)

    def learn_context(self, window):
        '''we assume window is a list of tokens, that were already added to the vocabulary'''
        N = self.N
        I = self.I
        for i in xrange(len(window)):
            wi = window[i]
            for j in xrange(len(window)):
                if j != i:
                    wj = window[j]
                    self.C[I[wi]] += self.E[I[wj]]


    def normalize_context(self):
        for i in xrange(len(self.C)):
            self.C[i] = self.C[i]/np.linalg.norm(self.C[i])


    def ngram_bind(self, window, i):
        K = self.K #number of ngrams
        N = self.N #number of features
        L = len(window)
        CUBE = np.zeros((N, K, K))
        CUBE[:, 0, 0] = deepcopy(self.PHI)
        CUBE_str = np.zeros((K, K)).astype(str)
        CUBE_str[:] = ''
        CUBE_str[0, 0] = "(PHI)"
        v = np.zeros(N)
        for k in xrange(1, K): #for each ngram level
            left_bound = i - k - 1
            right_bound= i + k 
            m = 0
            if left_bound > 0:
                CUBE_str[k, m] = "{} * {}".format(window[i-k:i], "PHI")
                m += 1
            if right_bound < L:
                CUBE_str[k, m] = "{} * {}".format(window[i:i+k], "PHI")



















        return v, CUBE_str






if __name__ == "__main__":
    params = []
    hparams = {"NFEATs":1024,  "NGRAMS":5}

    beagle = BEAGLE_HOLO(params, hparams)
    sentence = "0 1 2 3 4 5 6"
    beagle.update_vocab(sentence.split())
    beagle.learn_context(sentence.split())

    for i in xrange(len(sentence.split())):
        a, b = beagle.ngram_bind(sentence.split(), i)
        print sentence
        print "s[i] = {}".format(sentence.split()[i])
        print b
        print

