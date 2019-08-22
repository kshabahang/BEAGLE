from GeneralVSAModel import Model
import numpy as np
#import torch
from copy import deepcopy
from progressbar import ProgressBar
import sys
import pickle


from scipy.io import FortranFile


def open_unformatted_mat(fName, m):
    '''
    uses the scipy routine FortranFile to import unformatted f90/f95 file dumps (float64)
    '''
    f = FortranFile(fName, 'r')
    mat = []
    for i in range(m):
        mat.append(f.read_reals(dtype=np.float64))
        
    return np.array(mat)


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
    def __init__(self, params, hparams, QaAs=None, E=None, vocab = None):
        super(BEAGLE_HOLO, self).__init__(hparams, QaAs)

        self.params = params

        ###initialize environment vectors
        if type(E) == list:
            self.E = E
            self.C = np.zeros((len(E), hparams["NFEATs"]))
            self.O = np.zeros((len(E), hparams["NFEATs"]))
        else:
            self.E = []
            self.C = []
            self.O = []
        if type(vocab) == list:
            for i in xrange(len(vocab)):
                self.vocab.append(vocab[i])
                self.I[vocab[i]] = i
        else:
            self.vocab = []
        ###initialize context vectors
        ###initialize order vectors
        self.N      = hparams["NFEATs"]
        self.SD     = 1.0/np.sqrt(self.N)

        f = open("../rsc/STOPLIST.txt", "r")
        STOPLIST = f.readlines()
        f.close()
        self.STOPLIST = list(set([STOPLIST[i].strip() for i in xrange(len(STOPLIST))])) #discard repeats

        self.PHI = np.random.normal(0.0, self.SD, self.N)

        c = np.zeros((self.hparams["ORDER_WINDOW"], self.hparams["ORDER_WINDOW"], self.N))
        c[0][0][:] = self.PHI #initialize convolution cube
        self.cube = c

        if self.hparams["bind"] == "permutation":
            self.bind = self.RP_bind #random permutation method
            self.PI = {0:np.arange(0, self.N)}

            PI_fw = np.array([(i+1)%self.N for i in xrange(self.N)])
            PI_bw = np.array([(i-1)%self.N for i in xrange(self.N)])
            for i in xrange(self.hparams["ORDER_WINDOW"]):
                PI = np.arange(0, self.N)
                self.PI[self.hparams["ORDER_WINDOW"] + i] = np.arange(0, self.N)[PI_fw]
            for i in xrange(self.hparams["ORDER_WINDOW"]):
                self.PI[self.hparams["ORDER_WINDOW"] - 2 - i] = np.arange(0, self.N)[PI_bw]
        else:
            self.bind = self.convolution_cube #circular convolution method

        perm    = np.array([i for i in xrange(self.N)])
        np.random.shuffle(perm)
        self.p1 = deepcopy(perm)
        np.random.shuffle(perm)
        self.p2 = deepcopy(perm)

        self.E1_map = dict(zip([i for i in xrange(self.N)], np.random.permutation(self.N)))
        self.E2_map = dict(zip([i for i in xrange(self.N)], np.random.permutation(self.N)))
        self.D1_map = {self.E1_map[i]:i for i in xrange(self.N)} 
        self.D2_map = {self.E2_map[i]:i for i in xrange(self.N)}
    
        self.E1 = lambda a : np.array([a[self.E1_map[i]] for i in xrange(self.N)])
        self.E2 = lambda a : np.array([a[self.E2_map[i]] for i in xrange(self.N)])
        self.D1 = lambda a : np.array([a[self.D1_map[i]] for i in xrange(self.N)])
        self.D2 = lambda a : np.array([a[self.D2_map[i]] for i in xrange(self.N)])


#    def bind(self, a, b):
#        return cconv(a[self.p1], b[self.p2])

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
        WINDOW_LIM = min(self.hparams["CONTEXT_WINDOW"] + 1, len(window))
        for i in xrange(len(window)):
            if window[i] not in self.STOPLIST:
                try:
                    wi = window[i]
                    for j in xrange(len(window)):
                        if j != i and window[j] not in self.STOPLIST and np.abs(i - j) < WINDOW_LIM:
                            wj = window[j]
                            self.C[I[wi]] += self.E[I[wj]]
                except Exception as e:
                    print e
                    continue

    def learn_order(self, window):
        '''we assume window is a string of tokens, each already in the vocabulary'''
        window = window.split()
        I = self.I
        for i in xrange(len(window)):
            try:
                wi = window[i]
                self.O[I[wi]] += self.bind(window, i)
            except Exception as e:
                print e
                continue

    def RP_bind(self, window, i):
        o = np.zeros(self.N)
        I = self.I
        WINDOW_LIM = min(self.hparams["ORDER_WINDOW"] + 1, len(window))
#        print "*><*"*42
#        print "Window limit = {}".format(WINDOW_LIM)
#        print "Window: " + " ".join(window)
#        print "target word: {} - at index {}".format(window[i], i)
        for j in xrange(len(window)):
#            print "j: {}".format(i)
            if -WINDOW_LIM < j - i < 0 or WINDOW_LIM > j - i > 0 and window[j] not in self.STOPLIST:
#                print "    Forward association: pem_n ( E({}), {}, j - i = {} )".format(window[j], j-i, j - i)
             #   o += self.perm_n(self.E[I[window[j]]], j-i ) #forward associations
                o += self.E[I[window[j]]][self.PI[j - i]]
            #elif -WINDOW_LIM < j - i < 0:
#                print "    Backward association: invperm_n (E({}), {}) --- j - i = {}".format(window[j], i - j, j - i)
            #    o += self.invperm_n(self.E[I[window[j]]], i - j)
        return o


    def convolution_cube_dbug(self, window, i):
        Nwd = len(window)
        K = self.hparams["ORDER_WINDOW"]
        #c = deepcopy(self.cube)
        c = np.zeros((5,5,1)).astype(str)
        c[0,0]="PHI"
        
        for j in xrange(1, K): # j'th window size
            for k in xrange(0, j+1): # k'th j-gram
                if (j == k):
                    print ">> j == k"
                    pos_idx = i + k
                    inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                    if inBounds:
                        bind = "E1(" + c[j-1][k-1][0] + ") * E2(" + window[pos_idx] + ")"
                elif (k == 0):
                    print ">> k == 0"
                    pos_idx = i - j
                    inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                    if inBounds:
                        bind = "E1(" + window[pos_idx] + ") * E2(" + c[j-1][k][0] + ")"
                    else:
                        print ">> otherwise"
                        pos_idx = i + k - j #+ 1
                        max_idx = pos_idx + j
                        inBounds = (pos_idx >= 0) and (max_idx < Nwd)
                        if inBounds:
                            bind = "E1(" + sentence[pos_idx] + ") * E2(" + c[j-1][k][0] + ")"
            
                if inBounds:
                    print bind
                    c[j][k][0] = bind
        return c

    def convolution_cube(self, window, i):
        Nwd = len(window)
        K = self.hparams["ORDER_WINDOW"]
        c = deepcopy(self.cube)
        work = np.zeros(self.N)

        for j in xrange(1, K): # j'th window size
            for k in xrange(0, j+1): # k'th j-gram
                if (j == k):
                    pos_idx = i + k
                    inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                    if inBounds:
                        work[:] += cconv(self.E1(c[j-1][k-1][:]), self.E2(self.E[self.I[window[pos_idx]]]))
                elif (k == 0):
                    pos_idx = i - j
                    inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                    if inBounds:
                        work[:] += cconv(self.E1(self.E[self.I[window[pos_idx]]]), self.E2(c[j-1][k][:]))
                    else:
                        pos_idx = i + k - j 
                        max_idx = pos_idx + j
                        inBounds = (pos_idx >= 0) and (max_idx < Nwd)
                        if inBounds:
                            work[:] += cconv(self.E1(self.E[self.I[window[pos_idx]]]), self.E2(c[j-1][k][:]))

        return work


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

        strengths = sorted(zip(map(lambda u : vcos(u, v), self.C), self.vocab))[::-1]
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


#        if n < 0:
#            v = self.perm_n(v, -n)
#        if n > 0:
#            v = self.invperm_n(v, n)
        #
        if n == 0:
            strengths =  sorted(zip(map(lambda u : vcos(u, v), self.O), self.vocab))[::-1]
        else:
            strengths =  sorted(zip(map(lambda u : vcos(u, v[self.PI[n]]), self.E), self.vocab))[::-1]
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

        strengths = sorted(zip(map(lambda u : vcos(u, v), self.M), self.vocab))[::-1]
        for i in xrange(10):
            print round(strengths[i][0], 7), strengths[i][1]

        return strengths

def vcos(u,v):
    udotv = u.dot(v)
    if udotv != 0:
        return udotv/(np.linalg.norm(u)*np.linalg.norm(v))
    else:
        return 0

def open_npz(npzfile):
    return list(np.load(npzfile).items()[0][1])


if __name__ == "__main__":
    params = []
    hparams = {"NFEATs":30000,  "ORDER_WINDOW":1, "CONTEXT_WINDOW":2, "bind":"permutation"}
    windowSlide = True
    toStem = True
    toTest = False
    getOrder = True
    getContext= True
    ##load corpus
    REP = "RP"

    if toStem:
        from nltk.stem import WordNetLemmatizer

    MODE = sys.argv[1]

    if MODE == "help":
        print "\n".join(["<MODE> and ARGS: \n", 
                         "init <corpus path> <env vec path> | init <corpus path> <env vec path> <ref env vec path>", 
                         "train <corpus path> <env vec path> <current chunk idx> <num chunks>",
                         "compile <env vec path> <context vector paths> <order vector paths> <num chunks>",
                         "run <env vec path> <context vector paths> <order vector paths>"])

    if MODE == "init":
        corpus_path = sys.argv[2]
        env_vec_path = sys.argv[3] #path for dumping environment vecs
        if len(sys.argv) > 4:
            #assume we have path to vocab and environment vectors, but we may want to augment them AND they're in <ref env vec path>
            env_vec_ref= sys.argv[4]
            E_ref = list(open_npz("../rsc/{}/env.npz".format(env_vec_ref)))
            f = open("../rsc/{}/vocab.txt".format(env_vec_ref))
            vocab = f.readlines()
            f.close()

            vocab_ref = [vocab[i].strip() for i in xrange(len(vocab))]

        else:
            E_ref = []
            vocab_ref = []


        f = open("../rsc/{}/corpus_ready.txt".format(corpus_path), "r")
        corpus = f.readlines()
        f.close()

        vocab_intersect = list(set(" ".join(corpus).split()) & set(vocab_ref))

        print "{} environmental vectors from referent vocab intersect with current...".format(len(vocab_intersect))

        E = []
        vocab = []

        for i in xrange(len(vocab_ref)):
            if vocab_ref[i] in vocab_intersect:
                E.append(E_ref[i])
                vocab.append(vocab_ref[i])
        
        vocab0 = deepcopy(vocab) + list(set(" ".join(corpus).split()))

        ###all the previous entries in the beginning
        ###now we need to add the complement of the intersect
        vocab  = vocab_intersect + list(set(" ".join(corpus).split()) - set(vocab_intersect) ) # second term is empty if no ref specified


        N = hparams["NFEATs"]
        SD = 1/np.sqrt(N)
        f = open("../rsc/{}/vocab.txt".format(env_vec_path), "w")
        print "Generating environmental vectors..."
        pbar = ProgressBar(maxval=len(vocab)).start()
        for i in xrange(len(vocab)):
            if vocab[i] not in vocab_intersect: #initialize new env-vector
                if REP == "RG": #Random Gaussian
                    E.append(np.random.normal(0, SD, N))
                elif REP == "RP": #Random Permutation
                    u = np.hstack([np.zeros(hparams["NFEATs"]-60), np.ones(30), -1*np.ones(30)])
                    np.random.shuffle(u)
                    
                    E.append(u)#/np.linalg.norm(u)) #normalize

            f.write(vocab[i]+"\n")
            pbar.update(i+1)
        print "Dumping to disk..."
        f.close()


        np.savez_compressed("../rsc/{}/env.npz".format(env_vec_path), np.array(E))


    elif MODE == "train":
        corpus_path = sys.argv[2]
        env_vec_path = sys.argv[3] #path to environment vecs
        idx = int(sys.argv[4]) #current chunk
        CHU = int(sys.argv[5]) #number of chunks

        f = open("../rsc/{}".format(corpus_path + "/corpus_ready.txt"), "r")
        corpus = f.readlines()
        f.close()
        L = len(corpus)/CHU

        corpus = [corpus[i].strip() for i in xrange(len(corpus))][idx*L:(idx+1)*L]
        E = open_npz("../rsc/{}/env.npz".format(env_vec_path))
#        E = list(open_unformatted_mat("../rsc/NOVELS/env_novels.unf", 39076))
#        f = open("../rsc/NOVELS/word_list.txt", "r")
        f = open("../rsc/{}/vocab.txt".format(env_vec_path))
        vocab = f.readlines()
        f.close()
#        vocab = [vocab[i].split()[0] for i in xrange(len(vocab))]
        vocab = [vocab[i].strip() for i in xrange(len(vocab))]

        beagle = BEAGLE_HOLO(params, hparams, E = E, vocab = vocab)

        if windowSlide:
            corpus = ' '.join(corpus).split()
            window_size = max([hparams["CONTEXT_WINDOW"], hparams["ORDER_WINDOW"]])
            pbar = ProgressBar(maxval = len(corpus)/window_size).start()
            for i in xrange(len(corpus)/window_size):
                window = ' '.join(corpus[i*window_size:(i+1)*window_size])
                if getContext:
                    beagle.learn_context(window)
                if getOrder:
                    beagle.learn_order(window)
                pbar.update(i+1)
        else:
            pbar = ProgressBar(maxval = len(corpus)).start()
            for i in xrange(len(corpus)):
                #beagle.update_vocab(corpus[i])
                if getContext:
                    beagle.learn_context(corpus[i])
                if getOrder:
                    beagle.learn_order(corpus[i])
                pbar.update(i+1)
        if getContext:
            beagle.normalize_context()
        if getOrder:
            beagle.normalize_order()
        if getContext and getOrder:
            beagle.compute_lexicon()

        if getContext:
            np.savez_compressed("context_CHU{}.npz".format(idx), np.array(beagle.C))

        if getOrder:
            np.savez_compressed("order_ORD{}.npz".format(idx), np.array(beagle.O))


    elif MODE == "compile":
        env_vec_path = sys.argv[2]  #path to environment vecs
        source_context = sys.argv[3] #source of vectors to compile
        source_order = sys.argv[4]
        CHU = int(sys.argv[5])
        E = open_npz("../rsc/{}/env.npz".format(env_vec_path))
#        E = list(open_unformatted_mat("../rsc/NOVELS/env_novels.unf", 39076))

        f = open("../rsc/{}/vocab.txt".format(env_vec_path), "r")
#        f = open("../rsc/NOVELS/word_list.txt", "r")
        vocab = f.readlines()
        f.close()
#        vocab = [vocab[i].split()[0] for i in xrange(len(vocab))]
        vocab = [vocab[i].strip() for i in xrange(len(vocab))]

        beagle = BEAGLE_HOLO(params, hparams, E = E, vocab = vocab)
        if getOrder:
            O = open_npz("../rsc/{}/order_ORD0.npz".format(source_order))

        if getContext:
            C = open_npz("../rsc/{}/context_CHU0.npz".format(source_context))
        
        if getOrder:
            print "Compiling order "
            pbar = ProgressBar(maxval = CHU).start()
            for i in xrange(1, CHU):
                Oi = open_npz("../rsc/{}/order_ORD{}.npz".format(source_order, i))
                for j in xrange(len(Oi)):
                    O[j] += Oi[j]
                pbar.update(i+1)

            np.savez_compressed("../rsc/{}/order.npz".format(source_order), np.array(O))



        if getContext:
            print "Compiling context "
            pbar = ProgressBar(maxval = CHU).start() 
            for j in xrange(1, CHU): 
                Cj = open_npz("../rsc/{}/context_CHU{}.npz".format(source_context, j))
                C[j] += Cj[j]
                pbar.update(i+1)

            np.savez_compressed("../rsc/{}/context.npz".format(source_context), np.array(C))
 
    elif MODE == "run":
         env_vec_path = sys.argv[2]
         source_context = sys.argv[3] #source of vectors to compile
         source_order = sys.argv[4]

         E = open_npz("../rsc/{}/env.npz".format(env_vec_path))
#         E = list(open_unformatted_mat("../rsc/NOVELS_ENV/novels_env.unf", 39076))

#         f = open("../rsc/word_list_novels.txt", "r")
         f = open("../rsc/{}/vocab.txt".format(env_vec_path), "r")
         vocab = f.readlines()
         f.close()
#         vocab = [vocab[i].split()[0] for i in xrange(len(vocab))]
         vocab = [vocab[i].strip() for i in xrange(len(vocab))]
 
         beagle = BEAGLE_HOLO(params, hparams, E = E, vocab = vocab)

         beagle.O = open_npz("../rsc/{}/order.npz".format(source_order))

         beagle.C = open_npz("../rsc/{}/context.npz".format(source_context))

         beagle.normalize_order()
         beagle.normalize_context()
         
         heading = "Word before" + "Word after" + "Context-only"
         cues = ["KING", "PRESIDENT", "WAR", "SEA"]
         to_print = []
         for cue in cues:

             s = cue + "\n"
             cue_bw = beagle.sim_order(cue.lower(), 1)
             cue_fw = beagle.sim_order(cue.lower(), -1)
             cue_ct = beagle.sim_context(cue.lower())
             for i in xrange(5):
                 s += "{} {} ".format(cue_bw[i][1], round(cue_bw[i][0], 2) )
                 s += "{} {} ".format(cue_fw[i][1], round(cue_fw[i][0], 2) )
                 s += "{} {} ".format(cue_ct[i][1], round(cue_ct[i][0], 2) )
                 s += "\n"
             to_print.append(s)
         print "       ".join(heading)
         for i in xrange(len(cues)):
             print "-"*72
             print to_print[i]
            
































