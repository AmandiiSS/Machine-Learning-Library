#All import necessaries
import math
from collections import Counter
#import sys
#print(sys.getrecursionlimit())
#sys.setrecursionlimit(1500)

#Class: Node
#The @attribute will represent the "type" of the node.
# label is the end result prediction 
class Node: 
    attr: str
    label_result: str
    #branches should be dictionaries with the attribute value as a key and a Node as a children
    #{key = attr_name/type, val = Node}
    branches = {}
    def __init__(self, attribute: str = "", label: str = ""):
        self.attr = attribute
        self.label_result = label 
        self.branches = {}
    def add_decision_branch(self, attr_value: str, child):
        self.branches[attr_value] = child
    

#S is the set of examples (list (set of examples) of list (example) of ditionary (attributes and their "values" for the example), strings (labels), weight (float))
#Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
#Label are the possible lables (list)
#Gain is the type of gain we want to use to select the best attribute (string):
#   either information, me or gini
#MaxDepth is the maximum depth we want the tree to have (integer)

def ID3_weights(S, Attributes, Label, Gain = "information", MaxDepth = 10):
    sameLabel = same_label(S)
    commonLabel = common_label(S)
    #If all examples have the same label or Attibutes is empty or MaxDepth = 0 return a leaf node
    if sameLabel or MaxDepth == 0 or Attributes == {}:
        #Return a leaf node with the most common label
        return Node("",commonLabel)
    else:
        #We use the type of gain selected to choose the A from Attributes that best splits S
        #Best A is a tuple of the key (name of the attribute) and the value (list of all possible values)
        BestA = best_attribute(S, Attributes, Gain)
        #Create a Root Node for the tree
        root_node = Node(BestA[0],"")
        #For each possible value v that A can take we: 
            #1. Add a new tree branch orresponding to A=v
            #2. Let Sv be the subset of examples with A=v:
                #If Sv is empty: we create a leaf node with the most common value of Label in S
                #Else below this branch, add the subtree ID3_weigths(Sv, Attibutes-{A}, Label, Gain, MaxDepth-1)
        newAttributes = Attributes.copy()
        del newAttributes[BestA[0]]
        for v in BestA[1]:
            Sv = select_examples_v(S, BestA[0], v)
            if Sv == []:
                root_node.add_decision_branch(v, Node("",common_label(S)))
            else:
                root_node.add_decision_branch(v, ID3_weights(Sv,newAttributes, Label, Gain, MaxDepth-1))
        #Return Root Node
        return root_node


#This function returns True if all examples have the same label and False if not
def same_label(S):
    #local
    firstLabel = S[0][1]
    #Search all training data
    for s in S:
        if s[1] != firstLabel:
            return False
    return True


#This function returns the most common label among the ones from the set of examples S
def common_label(S):
    #This probably does not work as intended
    labelList = []
    for s in S:
        labelList.append(s[1])
    labelsAndValues = {}
    for s in S:
        if s[1] in labelsAndValues:
            labelsAndValues[s[1]] += s[2]
        else:
            labelsAndValues[s[1]] = s[2]
    return max(labelsAndValues, key=labelsAndValues.get)


#This function selects and returns the best attribute to split the set of examples S
def best_attribute(S, Attributes, Gain):
    #For each attribute, calculate each gain associated to the attribute and we want the one with max gain
    maxGain = 0
    maxA = list(Attributes.items())[0] #Get one random attribute to be the default one
    for A in Attributes.items(): #A is a tuple (key=name_attribute, value=list of possible values for the attribute)
        if Gain == "gini":
            attributeGain= giniGain(S,A)
        elif Gain == "me":
            attributeGain = meGain(S,A)
        else:
            attributeGain = informationGain(S,A)

        if attributeGain > maxGain:
            maxGain = attributeGain
            maxA = A
    return maxA #maxA is a tuple (key=name_attribute, value=list of possible values for the attribute)

#This function calculates the lenght of a group of examples S taking into account the weights
def S_length_weigth(S):
    S_length = 0
    for s in S:
        S_length += s[2]
    return S_length

#This function calculates and returns the information gain of the set of examples S of attribute A
def informationGain(S, A):
    #Calculate the general entropy
    generalEntropy = entropy(S)
    infogain = generalEntropy
    #Get the examples with attribute == A
    for v in A[1]:
        Sv = select_examples_v(S, A[0], v)
        Sv_len = S_length_weigth(Sv)
        S_len = S_length_weigth(S)
        if Sv != []:
            infogain -= (Sv_len/S_len)*entropy(Sv)
    return infogain

#This function calculates and returns the majority error gain of the set of examples S of attribute A
def meGain(S, A):
    #Calculate the general majority error
    generalMajorityError = majorityError(S)
    megain = generalMajorityError
    #Get the examples with attribute == A
    for v in A[1]:
        Sv = select_examples_v(S, A[0], v)
        Sv_len = S_length_weigth(Sv)
        S_len = S_length_weigth(S)
        if Sv != []:
            megain -= (Sv_len/S_len)*majorityError(Sv)
    return megain
    

#This function calculates and returns the gini gain of the set of examples S of attribute A
def giniGain(S, A):
    #Calculate the general majority error
    generalGiniIndex = giniIndex(S)
    ginigain = generalGiniIndex
    #Get the examples with attribute == A
    for v in A[1]:
        Sv = select_examples_v(S, A[0], v)
        Sv_len = S_length_weigth(Sv)
        S_len = S_length_weigth(S)
        if Sv != []:
            ginigain -= (Sv_len/S_len)*giniIndex(Sv)
    return ginigain
    

#This function calculates and returns the entropy of the set of examples S
def entropy(S):
    #labelDistribution is a dictionary with labels as keys and the number of that labels among S as values
    labelDistribution = label_distribution(S)
    totalLabels = sum(labelDistribution.values())
    H = 0
    for i in list(labelDistribution.values()):
        H += -(i/totalLabels)*math.log2(i/totalLabels)
    return H


#This function calculates and returns the gini index of the set of examples S
def giniIndex(S):
    #labelDistribution is a dictionary with labels as keys and the number of that labels among S as values
    labelDistribution = label_distribution(S)
    totalLabels = sum(labelDistribution.values())
    not_gi = 0
    for i in list(labelDistribution.values()):
        not_gi += (i/totalLabels)**2
    return 1-not_gi


#This function calculates and returns the majority error of the set of examples S
def majorityError(S):
    #labelDistribution is a dictionary with labels as keys and the number of that labels among S as values
    labelDistribution = label_distribution(S)
    totalLabels = sum(labelDistribution.values())
    me = 1- max(labelDistribution.values())/totalLabels
    return me


#This function returns a dictionary with labels as keys and the number of times the labels appear among S as values
# def label_distribution(S):
#     labelList = []
#     for s in S:
#         labelList.append(s[1])
#     labelsAndValues = Counter(labelList)
#     return labelsAndValues
def label_distribution(S):
    labelsAndValues = {}
    for s in S:
        if s[1] in labelsAndValues:
            labelsAndValues[s[1]] += s[2]
        else:
            labelsAndValues[s[1]] = s[2]
    return labelsAndValues


#This function returns a subset Sv of all examples with A=v among S
def select_examples_v(S, attribute_name, value):
    Sv = []
    for s in S:
        if s[0][attribute_name] == value:
            Sv.append(s)
    return Sv



#This recursive function returns a prediction for the example_test given, which is just a dictionary
def prediction(Tree:Node, example_test):
    #Base case
    if Tree.label_result != "" and Tree.label_result != None:
        return Tree.label_result
    #Recursion
    value = example_test[Tree.attr]
    label_predicted: str = prediction(Tree.branches[value],example_test)
    return label_predicted