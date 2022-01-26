from nltk.stem import WordNetLemmatizer
from progressbar import ProgressBar
import sys

def lemmatize(corpus):
    lemmatizer = WordNetLemmatizer()
    pbar = ProgressBar(maxval=len(corpus)).start()
    corpus_new = []
    for i in xrange(len(corpus)):
        corpus_new.append(' '.join(map(lambda w : lemmatizer.lemmatize(w),corpus[i].strip().split())))
        pbar.update(i+1)
    return corpus_new

if __name__ == "__main__":
    corpus_path = sys.argv[1]
    f = open(corpus_path, "r")
    corpus = f.readlines()
    f.close()
    corpus = lemmatize(corpus)
    f = open("corpus_lemmatized.txt", "w")
    f.write("\n".join(corpus))
    f.close()
    
