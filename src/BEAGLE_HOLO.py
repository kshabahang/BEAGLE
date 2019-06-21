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





class BIND_NODE(object):
    def __init__(self, h, l = None, r = None, parent=None):
        self.h = h #height
        self.l = l #left-child
        self.r = r #right-child






class BIND(object):
    def __init__(self, K, s, i, PHI):
        self.K = K    # max number of ngrams
        self.s = s    # token sequence
        self.i = i    # index of target item to encode
        self.PHI= PHI # place-holder for the target item
        self.l = len(s) #number of tokens in the sequence
        self.root = BIND_NODE(0)

    def recurse_left(self, ):
        #

    def recurse_right(self, ):
        #









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
        for i in xrange(len(window)):
            if window[i] not in self.STOPLIST:
                wi = window[i]
                for j in xrange(len(window)):
                    if j != i and window[j] not in self.STOPLIST:
                        wj = window[j]
                        self.C[I[wi]] += self.E[I[wj]]

    def learn_order(self, window):
        '''we assume window is a string of tokens, each already in the vocabulary'''
        window = window.split()
        I = self.I
        for i in xrange(len(window)):
            wi = window[i]
            self.O[I[wi]] += self.ngram_bind(window, i)


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


    def ngram_bind(self, window, i):
        K = self.K #number of ngrams
        N = self.N #number of features
        I = self.I 
        L = len(window)
        bindings = {"PHI":deepcopy(self.PHI)}
        v = np.zeros(N)
        for n in xrange(2, K+1): #for each ngram level
            ngrams = zip(*[window[j:] for j in range(n)])
            ngrams_str = ''
            for j in xrange(len(ngrams)):
                if window[i] in ngrams[j]:
                    
                    ngram = list(ngrams[j])
                    nless1gram = ' '.join(ngram[:n-1])
                    if nless1gram in bindings: #recylce
                        bindings[' '.join(ngram)] = self.bind(bindings[nless1gram], self.E[I[ngram[n-1]]])
                    else: #create anew
                        w = np.zeros(N)
                        for l in xrange(len(ngram)-1):
                            if ngram[l] == window[i]:
                                v1 = bindings["PHI"]
                                a = "PHI"
                            else:
                                v1 = self.E[I[ngram[l]]]
                                a = ngram[l]
                            if ngram[l+1] == window[i]:
                                v2 = bindings["PHI"]
                                b = "PHI"
                            else:
                                v2 = self.E[I[ngram[l+1]]]
                                b = ngram[l]

                            w += self.bind(v1, v2)
                            ngrams_str += " ( {} * {} ) ".format(a, b)
                        wlen = np.linalg.norm(w)
                        if wlen > 0:
                            w = w/wlen
                        bindings[' '.join(ngram)] = w
                        v += w
                        print ngrams_str
        vlen = np.linalg.norm(v)
        if vlen > 0:
            return v/vlen
        else:
            return v

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



    def sim_order(self, w):
        I = self.I
        if type(w) == str and w in self.vocab:
            v = self.O[I[w]]
        elif type(w) == np.ndarray:
            v = w
        else:
            print "ERROR: the token, {}, is not known".format(w)
            return -1

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
    hparams = {"NFEATs":2048,  "NGRAMS":7}

    beagle = BEAGLE_HOLO(params, hparams)

    ##load corpus
    f = open("../rsc/BaseballWiki.txt", "r")
    corpus = f.readlines()
    f.close()
    corpus = [corpus[i].strip() for i in xrange(len(corpus))]
#    pbar = ProgressBar(maxval = len(corpus)).start()
#    for i in xrange(len(corpus)):
#        beagle.update_vocab(corpus[i])
#        beagle.learn_context(corpus[i])
#        beagle.learn_order(corpus[i])
#        pbar.update(i+1)
#    beagle.normalize_context()
#    beagle.normalize_order()
#    beagle.compute_lexicon()
#    sims = beagle.sim('batter')

        

    vocab = "A B C".split()
    n = 1024
    entries = {vocab[i]:np.random.normal(0.0, 1.0/np.sqrt(n), n) for i in xrange(len(vocab))}
    sent = "A B C".split()
    PHI = np.random.normal(0.0, 1.0/np.sqrt(n), n)

    E1_map = dict(zip([i for i in xrange(n)], np.random.permutation(n)))
    E2_map = dict(zip([i for i in xrange(n)], np.random.permutation(n)))
    D1_map = {E1_map[i]:i for i in xrange(n)} 
    D2_map = {E2_map[i]:i for i in xrange(n)}

    E1 = lambda a : np.array([a[E1_map[i]] for i in xrange(n)])
    E2 = lambda a : np.array([a[E2_map[i]] for i in xrange(n)])
    D1 = lambda a : np.array([a[D1_map[i]] for i in xrange(n)])
    D2 = lambda a : np.array([a[D2_map[i]] for i in xrange(n)])

    
    #encode order for B in position 1
    idx = sent.index("B")
    m_b = np.zeros(n)
    m_b += cconv(E1(entries[sent[idx-1]]), E2(PHI)) #2-gram: A_
    m_b += cconv(E1(PHI), E2(entries[sent[idx+1]])) #2-gram: _C
    m_b += cconv(E1( cconv(E1(entries[sent[idx-1]]), E2(PHI)) ), E2(entries[sent[idx+1]])) #3-gram: A_C
    
    C_hat = D2(ccorr(E1(PHI), m_b)) #what vector succeeds B based on _C
    print np.dot(C_hat, entries["C"])
    C_hat = D2( ccorr( E1(cconv(E1(entries["A"]), E2(PHI))  ), m_b)) #what vector succeeds B, given A preceded it?
    print np.dot(C_hat, entries["C"])



