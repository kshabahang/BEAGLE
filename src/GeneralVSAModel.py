import numpy as np
from numpy import *
#import torch


class Model(object):
    def __init__(self, hparams, QaAs = None):

        self.hparams = hparams

        self.I     = {}
        self.vocab = []
        self.V     = 0
        self.banks = {}
        self.wf    = {}
        self.QaAs  = QaAs


    def eval(self, verbose=False):
        idx_predict = self.hparams["idx_predict"]
        #print
        accuracy = {}
        targ_ranks={}
        under_the_hood = {}
        for cond in self.QaAs.keys():
            acc   = []
            ranks = []
            states = []
            if verbose:
                print()
                print( "%10s %2s %10s %20s %10s %2s %10s" % ('correct', '', 'report', '', 'syntagmatic', '', 'paradigmatic'))
                print ("%10s %2s %10s %20s %10s %2s %10s" % ('-------', '', '------', '', '-----------', '', '------------'))
                print()
            for i in range(len(self.QaAs[cond])):
                q, correct = self.QaAs[cond][i]

                self.update_vocab(q.split())

                X_o, X_s, Y_o, Y_s = self.prep_input(q.split(), idx_predict)


                report, top_syn, top_par = self.report(X_o, X_s)

                acc.append(report == correct)

                ranks.append(self.get_rank(correct))


#                               states.append([q, pari, syni, (report, correct)]

                if verbose:
                    print ("%10s %2s %10s %20s %10s %2s %10s" % (correct, '', report, '', top_syn, '', top_par))
                # print "{} {}                  {} x {}".format(correct, report, top_syn, top_par)
            if verbose:
                print()
                print (cond.replace('_', ' ') + ': {}/{}'.format(sum(acc), len(acc)))
                print()
            accuracy[cond] = acc
            under_the_hood[cond] = states
            targ_ranks[cond] = ranks

        return under_the_hood, accuracy, targ_ranks

    def get_rank(self, correct):
        if self.hparams["mode"] == "numpy":
            rank = list(argsort((self.echo)[:self.V])[::-1]).index(self.I[correct])
        elif self.hparams["mode"] == "torch":
            rank = list(argsort((self.echo.numpy())[:self.V])[::-1]).index(self.I[correct])
        elif self.hparams["mode"] == "gpu":
            rank = list(argsort((self.echo.cpu().numpy())[:self.V])[::-1]).index(self.I[correct])
        return rank


    def sent2vec(self, input_sentence):
        from numpy import zeros
        '''vectorize a given input sentence for feeding into the model'''
        numSlots = self.hparams['numSlots']

        if self.hparams["mode"] == "numpy":
            o = zeros(self.V0*numSlots) #order information
            s = zeros(self.V0) #syntagmatic informaiton
        elif self.hparams["mode"] == "torch":
            o = torch.zeros(self.V0*numSlots, dtype = self.hparams["dtype"])
            s = torch.zeros(self.V0, dtype = self.hparams["dtype"])
        elif self.hparams["mode"] == "gpu":
            o = torch.zeros(self.V0*numSlots, dtype = self.hparams["dtype"]).cuda()
            s = torch.zeros(self.V0, dtype = self.hparams["dtype"]).cuda()


        iShiftS = max(len(input_sentence) - numSlots,   0) #iShiftS > 0 implies len(input_sentence) > numSlots
        for i in range(min(len(input_sentence), numSlots)):
            w_i = input_sentence[iShiftS + i]
            if w_i != '_':
                o[i*self.V0 + self.I[w_i]] = 1
                s[self.I[w_i]] = 1

#               if input_sentence[iShiftS] != '_':
#                       assert o[self.I[input_sentence[iShiftS]]] == 1, 'iShiftS {}; iShiftK {}'.format(iShiftS, iShiftK)

        return o, s

    def prep_input(self, window, idx_predict=-1):
        from numpy import zeros, matrix, argmax, random

        numSlots = self.hparams['numSlots']

        Y, X_s = self.sent2vec(window)
        if self.hparams["mode"] == "numpy":
            X_o = zeros(self.V0*numSlots)
        elif self.hparams["mode"] == "torch":
            X_o = torch.zeros(self.V0*numSlots, dtype=self.hparams["dtype"])
        elif self.hparams["mode"] == "gpu":
            X_o = torch.zeros(self.V0*numSlots, dtype=self.hparams["dtype"]).cuda()


        X_o += Y

        self.X_o = X_o
        self.X_s = X_s
        self.Y_s = X_s
        self.Y   = Y

        return self.X_o, self.X_s, self.Y, self.Y_s

    def save_vocab(self, path):
        import pickle
        #saves vocab details
        f = open(path + 'I.pkl', 'wb')
        pickle.dump(self.I, f) #we can get vocab and V with this so both are redundant
        f.close()

        f = open(path + 'wf.pkl', 'wb')
        pickle.dump(self.wf, f)
        f.close()

        f = open(path + 'vocab.pkl', 'wb')
        pickle.dump(self.vocab, f)
        f.close()

        f = open(path + 'details.dat', 'w')
        f.write('V0: {}'.format(self.V0) + '\n')
        f.close()


    def load_vocab(self, path):
        import pickle
        #load vocab details
        f = open(path + 'I.pkl', 'rb')
        self.I = pickle.load(f)
        f.close()

        f = open(path + 'wf.pkl', 'rb')
        self.wf = pickle.load(f)
        f.close()

        f = open(path + 'vocab.pkl', 'rb')
        self.vocab = pickle.load(f)
        f.close()

        f = open(path + 'details.dat', 'r')
        details = f.readlines()
        f.close()
        self.V0 = int(details[0].split(':')[1])


        self.V = len(self.I)

    def probe(self, sentence, showRec = False):
        X_order, X_syntag = self.sent2vec(sentence.split())
        report = self.report(X_order, X_syntag)
        if showRec:
            self.print_inplace(self.recurrence_frames, header = sentence)
        return report
    #-----------------------diagnostics------------------------#


    def view_banks(self, k, mute = False):
        '''print the top k most active words across all banks'''
        numBanks = self.hparams['numBanks']
        bank_labels = self.hparams['bank_labels']
        to_print = []
        for i in range(numBanks):
            s = ''
            bank_data = self.banks[bank_labels[i]]

            #print bank_data
            for j in range(len(bank_data[0][:k])):
                if bank_data[1][j]  >= 0.00000000001:
                    s += '%10s %5f ' % (bank_data[0][j], bank_data[1][j])
            to_print.append(bank_labels[i] + ' ' +  s)
        if mute:
            return to_print
        else:
            print ('\n'.join(to_print))
    def view_bank(self):
        from numpy import argsort, array
        bank = self.echo[:self.V][0]
        isort = array(argsort(bank)[::-1])
        bank = bank[isort]
        s = ''
        for j in range(5):
            if isort[j] < self.V and bank[j] >= 0.0000001:
                s += '%10s %5f ' % (self.vocab[isort[j]], bank[j])
        print (s)

    def sort_banks(self, echo = None):
        from numpy import argsort, array
        '''sort all the banks in non-ascending order of activations'''
        numBanks = self.hparams['numBanks']
        bank_labels = self.hparams['bank_labels']
        if type(echo) != ndarray:
            if self.hparams["mode"] == "gpu":
                echo = self.echo_full.cpu().numpy()
            elif self.hparams["mode"] == "torch":
                echo = self.echo_full.numpy()
        for i in range(numBanks):
            bank = echo[i*self.V0:(i+1)*self.V0][:self.V]
            isort = argsort(bank)[::-1]
            self.banks[bank_labels[i]] = [array(self.vocab)[isort], bank[isort].astype(float)]
    def print_inplace(self, lines, header = '', t = 1):
        import curses, time
        screen = curses.initscr()
        try:
            for i in range(self.hparams["numBanks"] + 3):
                screen.addstr(i, 0, '*')
            screen.addstr(0, 0, header)
            for j in range(2, len(lines)):
                for i in range(len(lines[j])):
                    screen.addstr(i+2, 0, lines[j][i])
                    screen.refresh()
                time.sleep(t)
        except:
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            return
        curses.nocbreak()
        curses.echo()
        curses.endwin()
