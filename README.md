# BEAGLE
code for implementing the BEAGLE model by Mewhort and Jones (2007), in addition to an alternative binding approach for constructing the serial-order vectors, using Random Permutaitons (Sahlgren, Holst, & Kanerva, 2008).


More specifically, the code was used to construct vectors for the following paper...


Osth, A. F., Heathcote, A., Mewhort, D. J. K., & Shabahang, K., (in review). Global semantic similarity effects in recognition memory: Insights from BEAGLE representations and the diffusion decision model


...

Note that parts of the code may be redundant due to change of approach.  For instance, instead of shifting vectors on-the-fly when binding via random permutations, I later changed the code so that we pre-compute the permutation vectors and use Numpy's vector indexing approach to speed up the process.  I have tested the code against typical examples to ensure correctness, but perhaps you'll notice something that I missed.


If this work was useful to you and you want to somehow thank my work, donations are welcome:
<p>ETH: 0x5831aa28D2378Ae5333f57B3C2d8FeC3C736eEeb</p>
<p>XMR: 44q99xTChW3B8dNykAGRza66TRZi2wpnAZtj2FuGwwL9H8shiXJYwgcicGf529uufyRDBMsLTLXAcKWohQRHvvdfUw4fWm2</p>
<p>DODGE: DEhsBqavQmY2i7RgZQCsjXeTY9kceuy454</p>
<p>LTC: ltc1qq9gdv7tpmwutdxvap05t049rvm96qtmmmtshs2</p>

