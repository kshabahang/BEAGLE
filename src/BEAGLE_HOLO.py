from GeneralVSAModel import Model
import numpy as np
import torch
from copy import deepcopy
from progressbar import ProgressBar

def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def ccorr(a, b):
    '''
    Computes the circular correlation (inverse convolution) of the real-valued
    vector a with b.
    '''
    return cconv(np.roll(a[::-1], 1), b)


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
        self.SD     = 1.0/np.sqrt(self.N)

        f = open("../rsc/STOPLIST.txt", "r")
        STOPLIST = f.readlines()
        f.close()
        self.STOPLIST = [STOPLIST[i].strip() for i in xrange(len(STOPLIST))]

        self.PHI = np.random.normal(0.0, self.SD, self.N)
        perm    = np.array([i for i in xrange(self.N)])
        np.random.shuffle(perm)
        self.p1 = deepcopy(perm)
        np.random.shuffle(perm)
        self.p2 = deepcopy(perm)

#        self.E1_map = dict(zip([i for i in xrange(self.N)], np.random.permutation(self.N)))
#        self.E2_map = dict(zip([i for i in xrange(self.N)], np.random.permutation(self.N)))
#        self.D1_map = {self.E1_map[i]:i for i in xrange(self.N)} 
#        self.D2_map = {self.E2_map[i]:i for i in xrange(self.N)}
#    
#        self.E1 = lambda a : np.array([a[self.E1_map[i]] for i in xrange(self.N)])
#        self.E2 = lambda a : np.array([a[self.E2_map[i]] for i in xrange(self.N)])
#        self.D1 = lambda a : np.array([a[self.D1_map[i]] for i in xrange(self.N)])
#        self.D2 = lambda a : np.array([a[self.D2_map[i]] for i in xrange(self.N)])


    def bind(self, a, b):
        return cconv(a[self.p1], b[self.p2])

    def update_vocab(self, corpus_process):
        N  = self.N
        SD = self.SD
        corpus_process = corpus_process.split()

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
        '''we assume window is string of tokens, each already in the vocabulary'''
        window = window.split()
        N = self.N
        I = self.I
        WINDOW_LIM = self.hparams["CONTEXT_WINDOW"] + 1
        for i in xrange(len(window)):
            if window[i] not in self.STOPLIST:
                wi = window[i]
                for j in xrange(len(window)):
                    if j != i and window[j] not in self.STOPLIST and np.abs(i - j) < WINDOW_LIM:
                        wj = window[j]
                        self.C[I[wi]] += self.E[I[wj]]

    def learn_order(self, window):
        '''we assume window is a string of tokens, each already in the vocabulary'''
        window = window.split()
        I = self.I
        for i in xrange(len(window)):
            wi = window[i]
            self.O[I[wi]] += self.RP_bind(window, i)

    def RP_bind(self, window, i):
        o = np.zeros(self.N)
        I = self.I
        WINDOW_LIM = self.hparams["ORDER_WINDOW"] + 1
        for j in xrange(len(window)):
            if WINDOW_LIM > j - i > 0:
                o += self.perm_n(self.E[I[window[j]]], j-i ) #forward associations
            elif -WINDOW_LIM < j - i < 0:
                o += self.invperm_n(self.E[I[window[j]]], i - j)
        return o



    def normalize_context(self):
        for i in xrange(len(self.C)):
            vlen = np.linalg.norm(self.C[i])
            if vlen > 0:
                self.C[i] = self.C[i]/vlen  



    def normalize_order(self):
        for i in xrange(len(self.O)):
            vlen = np.linalg.norm(self.O[i])
            if vlen > 0:
                self.O[i] = self.O[i]/vlen  



    def compute_lexicon(self):
        '''combines context and order information
           assumes context and order vectors have been normalized'''
        I = self.I
        M = []
        for i in xrange(self.V):
            v = self.C[i] + self.O[i]
            vlen = np.linalg.norm(v)
            if vlen > 0:
                v = v/vlen
            M.append(v)
        self.M = M

    def sim_context(self, w):
        I = self.I
        if type(w) == str and w in self.vocab:
            v = self.C[I[w]]
        elif type(w) == np.ndarray:
            v = w
        else:
            print "ERROR: the token, {}, is not known".format(w)
            return -1

        strengths = sorted(zip(map(lambda u : np.dot(u, v), self.C), self.vocab))[::-1]
        for i in xrange(10):
            print round(strengths[i][0], 7), strengths[i][1]
        return strengths

    def perm_n(self, x, n):
        y = np.zeros(self.N)
        for i in xrange(self.N):
            y[(i+n)%self.N] = x[i]
        return y

    def invperm_n(self, y, n):
        x = np.zeros(self.N)
        for i in xrange(self.N):
            x[(i-n)] = y[i]
        return x

    def sim_order(self, w, n=0):
        I = self.I
        if type(w) == str and w in self.vocab:
            v = self.O[I[w]]
        elif type(w) == np.ndarray:
            v = w
        else:
            print "ERROR: the token, {}, is not known".format(w)
            return -1

        if n < 0:
            v = self.perm_n(v, -n)
        if n > 0:
            v = self.invperm_n(v, n)

        strengths =  sorted(zip(map(lambda u : np.dot(u, v), self.O), self.vocab))[::-1]
        for i in xrange(10):
            print round(strengths[i][0], 7), strengths[i][1]

        return strengths

    def sim(self, w):
        I = self.I
        if type(w) == str and w in self.vocab:
            v = self.M[I[w]]
        elif type(w) == np.ndarray:
            v = w
        else:
            print "ERROR: the token, {}, is not known".format(w)
            return -1

        strengths = sorted(zip(map(lambda u : np.dot(u, v), self.M), self.vocab))[::-1]
        for i in xrange(10):
            print round(strengths[i][0], 7), strengths[i][1]

        return strengths


if __name__ == "__main__":
    params = []
    hparams = {"NFEATs":1024,  "ORDER_WINDOW":7, "CONTEXT_WINDOW":7}

    beagle = BEAGLE_HOLO(params, hparams)

    ##load corpus
    f = open("../rsc/tasaClean.txt", "r")
    corpus = f.readlines()
    f.close()
    corpus = [corpus[i].strip() for i in xrange(len(corpus))]

    corpus = ["a b c d e f g", "A B C D E F G", "1 2 3 4 5 6 7 8", "the cat was sitting on the mat"]
    
    pbar = ProgressBar(maxval = len(corpus)).start()
    for i in xrange(len(corpus)):
        beagle.update_vocab(corpus[i])
        beagle.learn_context(corpus[i])
        beagle.learn_order(corpus[i])
        pbar.update(i+1)
    beagle.normalize_context()
    beagle.normalize_order()
    beagle.compute_lexicon()

    print "Testing retrieval using serial order with vector for letter D"

    for i in xrange(1,4):
        print "*"*72
        print "lag {}".format(i)
        print "Forward"
        a = beagle.sim_order("D", i)
        print ""
        print "Backward"
        a = beagle.sim_order("D", -i)

        

#    vocab = "A B C".split()
#    n = 1024
#    entries = {vocab[i]:np.random.normal(0.0, 1.0/np.sqrt(n), n) for i in xrange(len(vocab))}
#    sent = "A B C".split()
#    PHI = np.random.normal(0.0, 1.0/np.sqrt(n), n)
#
#    E1_map = dict(zip([i for i in xrange(n)], np.random.permutation(n)))
#    E2_map = dict(zip([i for i in xrange(n)], np.random.permutation(n)))
#    D1_map = {E1_map[i]:i for i in xrange(n)} 
#    D2_map = {E2_map[i]:i for i in xrange(n)}
#
#    E1 = lambda a : np.array([a[E1_map[i]] for i in xrange(n)])
#    E2 = lambda a : np.array([a[E2_map[i]] for i in xrange(n)])
#    D1 = lambda a : np.array([a[D1_map[i]] for i in xrange(n)])
#    D2 = lambda a : np.array([a[D2_map[i]] for i in xrange(n)])
#
#    
#    #encode order for B in position 1
#    idx = sent.index("B")
#    m_b = np.zeros(n)
#    m_b += cconv(E1(entries[sent[idx-1]]), E2(PHI)) #2-gram: A_
#    m_b += cconv(E1(PHI), E2(entries[sent[idx+1]])) #2-gram: _C
#    m_b += cconv(E1( cconv(E1(entries[sent[idx-1]]), E2(PHI)) ), E2(entries[sent[idx+1]])) #3-gram: A_C
#    
#    C_hat = D2(ccorr(E1(PHI), m_b)) #what vector succeeds B based on _C
#    print np.dot(C_hat, entries["C"])
#    C_hat = D2( ccorr( E1(cconv(E1(entries["A"]), E2(PHI))  ), m_b)) #what vector succeeds B, given A preceded it?
#    print np.dot(C_hat, entries["C"])



