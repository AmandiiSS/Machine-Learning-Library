import decisiontree
import math
import random
from collections import Counter

#This function returns a subset Sb of all examples with number associated from a list of num
def select_examples_b(S, random_numbers):
    Sb = []
    for num in random_numbers:
        Sb.append(S[num])
    return Sb

#S is the set of examples. The weights will be changed there directly.
#T is the number of times we iterate the algorithm
def Bagged_decisiontree(S,Attributes,Label,gain,depth,T):
    #In this variable we will save each Tree and vote associated to it from each iteration
    Trees = []
    for t in range(T):
        random_numbers = []
        for i in range(len(S)):
            random_numbers.append(random.randrange(0,len(S)))
        Sb = select_examples_b(S, random_numbers)
        Tree_t = decisiontree.ID3(Sb, Attributes, Label, gain, depth)
        Trees.append(Tree_t)
    return Trees
        

#Trees = list of [Tree]
#s example to test
#Label the two possible labels
def prediction_Bagging(Trees,s):
    pred = []
    for tree in Trees:
        prediction_tree = decisiontree.prediction(tree, s)
        pred.append(prediction_tree)
    return common_prediction(pred)


def common_prediction(pred):
    predictionsAndValues = Counter(pred)
    return max(predictionsAndValues, key=predictionsAndValues.get)
