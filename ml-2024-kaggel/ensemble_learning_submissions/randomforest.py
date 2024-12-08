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
def RandomForest(S,Attributes,Label,gain,depth,T,FeatureSetSize):
    #In this variable we will save each Tree and vote associated to it from each iteration
    Trees = []
    for t in range(T):
        random_numbers = []
        for i in range(len(S)):
            random_numbers.append(random.randrange(0,len(S)))
        Sb = select_examples_b(S, random_numbers)
        Tree_t = RandTreeLearn(Sb, Attributes, Label, gain, depth, FeatureSetSize)
        Trees.append(Tree_t)
    return Trees
        

#Trees = list of [Tree]
#s example to test
#Label the two possible labels
def prediction_RandomForest(Trees,s):
    pred = []
    for tree in Trees:
        prediction_tree = decisiontree.prediction(tree, s)
        pred.append(prediction_tree)
    return common_prediction(pred)


def common_prediction(pred):
    predictionsAndValues = Counter(pred)
    return max(predictionsAndValues, key=predictionsAndValues.get)



def RandTreeLearn(S, Attributes, Label, Gain = "information", MaxDepth = 10, FeatureSetSize=4):
    sameLabel = decisiontree.same_label(S)
    commonLabel = decisiontree.common_label(S)
    #If all examples have the same label or Attibutes is empty or MaxDepth = 0 return a leaf node
    if sameLabel or MaxDepth == 0 or Attributes == {}:
        #Return a leaf node with the most common label
        return decisiontree.Node("",commonLabel)
    else:
        #We use the type of gain selected to choose the A from a random subset of Attributes that best splits S
        AttributesRandomSubset = {}
        randomAttributes = random.sample(list(Attributes.items()),min(FeatureSetSize, len(Attributes)))
        for attribute in randomAttributes:
            AttributesRandomSubset[attribute[0]] = attribute[1]
        #Best A is a tuple of the key (name of the attribute) and the value (list of all possible values)
        BestA = decisiontree.best_attribute(S, AttributesRandomSubset, Gain)
        #Create a Root Node for the tree
        root_node = decisiontree.Node(BestA[0],"")
        #For each possible value v that A can take we: 
            #1. Add a new tree branch orresponding to A=v
            #2. Let Sv be the subset of examples with A=v:
                #If Sv is empty: we create a leaf node with the most common value of Label in S
                #Else below this branch, add the subtree ID3(Sv, Attibutes-{A}, Label, Gain, MaxDepth-1)
        newAttributes = Attributes.copy()
        del newAttributes[BestA[0]]
        for v in BestA[1]:
            Sv = decisiontree.select_examples_v(S, BestA[0], v)
            if Sv == []:
                root_node.add_decision_branch(v, decisiontree.Node("",decisiontree.common_label(S)))
            else:
                root_node.add_decision_branch(v, RandTreeLearn(Sv,newAttributes, Label, Gain, MaxDepth-1,FeatureSetSize))
        #Return Root Node
        return root_node