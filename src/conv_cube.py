import numpy as np



def convolution_cube1(i, Nwd):
    c = np.zeros((5,5,1)).astype(str)
    c[0][0][0] = PHI
    for j in xrange(1, 5): # j'th window size
        for k in xrange(0, j+1): # k'th j-gram
            if (j == k):
#                print ">> j == k"
                pos_idx = i + k
                inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                if inBounds:
                    bind = "E1(" + c[j-1][k-1][0] + ") * E2(" + sentence[pos_idx] + ")"
            elif (k == 0):
#                print ">> k == 0"
                pos_idx = i - j
                inBounds = (pos_idx >= 0) and (pos_idx < Nwd)
                if inBounds:
                    bind = "E1(" + sentence[pos_idx] + ") * E2(" + c[j-1][k][0] + ")"
            else:
#                print ">> otherwise"
                pos_idx = i + k - j #+ 1
                max_idx = pos_idx + j
                inBounds = (pos_idx >= 0) and (max_idx < Nwd)
                if inBounds:
                    bind = "E1(" + sentence[pos_idx] + ") * E2(" + c[j-1][k][0] + ")"

            if inBounds:
                print bind
                c[j][k][0] = bind
    return c


if __name__ == "__main__":
    global sentence
    global visual
    global work
    global PHI






    sentence = ["A", "B", "C", "D"]
    PHI = "PHI"
#    for i in xrange(len(sentence)):
    if True:
        i = 3
        print "*"*72
        print sentence[i]
        c = convolution_cube1(i, len(sentence))
#    print c
