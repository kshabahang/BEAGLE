from progressbar import ProgressBar
import sys


def count(corpus):
    pbar = ProgressBar(maxval = len(corpus)).start()
    wf = {}
    for i in xrange(len(corpus)):
        line = corpus[i].split()
        for j in xrange(len(line)):
            if line[j] in wf:
                wf[line[j]] += 1
            else:
                wf[line[j]] = 0

        pbar.update(i+1)

    return wf

def freq_cut(corpus, fcut, wf):
    pbar = ProgressBar(maxval = len(corpus)).start()
    corpus_new = []
    cuts = []
    for i in xrange(len(corpus)):
        line = corpus[i].split()
        line_new = []
        for j in xrange(len(line)):
            if wf[line[j]] < fcut:
                line_new.append(line[j])
            elif line[j] not in cuts:
                cuts.append(line[j])
        pbar.update(i+1)
        if len(line_new) > 1:
            corpus_new.append(' '.join(line_new))
    return corpus_new, cuts


if __name__ == "__main__":
    corpus_path = sys.argv[1]
    fcut = int(sys.argv[2])

    f = open(corpus_path, "r")
    corpus = f.readlines()
    f.close()
    print "Counting words..."
    wf = count(corpus)

    print "Getting rid of words with frequency higher than {}".format(fcut)
    corpus, cuts = freq_cut(corpus, fcut, wf)

    print "Saving corpus..."
    f = open("corpus_ready.txt", "w")
    f.write("\n".join(corpus))
    f.close()

    print "Saving cuts..."
    f = open("highWF.txt", "w")
    f.write("\n".join(cuts))
    f.close()
