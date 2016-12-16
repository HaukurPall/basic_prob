import math

def binary_entropy(p):
    return entropy(1-p) + entropy(p)

def entropy(p):
    return p*math.log2(1/p)

rq = entropy(1/10.0)*1/3.0+entropy(1/21.0)*2/3.0*0.5+entropy(1/5.0)*2/3.0*0.5
print("{}".format(rq))
