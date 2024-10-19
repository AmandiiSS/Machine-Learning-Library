INSTRUCTIONS TO USE THE ENSEMBLE LEARNING ALGORITHMS:
-------------------------------------------------------

%%%%%%% Adaboost algorithm %%%%%%%

(It returns a list of list [vote,tree])
AdaBoost_decisiontree(S,Attributes,Label,Gain,MaxDepth,T)
where:
1. S is the set of examples. The weights will be changed there directly.
2. Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
3. Label are the possible lables (list)
4. Gain is the type of gain we want to use to select the best attribute (string): either information, me or gini
5. MaxDepth is the maximum depth we want the tree to have (integer)
6. T is the number of times we iterate the algorithm

prediction_AdaBoost(Trees_and_votes,s, Label)
where:
1. Trees_and_votes is a list of list [vote,tree]
2. s is the example from which we want to make the prediction
3. Label is a list of the possible lables


%%%%%%% Bagged algorithm %%%%%%%

(It returns a list of trees)
Bagged_decisiontree(S,Attributes,Label,Gain,MaxDepth,T)
where:
1. S is the set of examples
2. Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
3. Label are the possible lables (list)
4. Gain is the type of gain we want to use to select the best attribute (string): either information, me or gini
5. MaxDepth is the maximum depth we want the tree to have (integer)
6. T is the number of times we iterate the algorithm

prediction_Bagging(Trees,s)
where:
1. Trees is a list of trees
2. s is the example from which we want to make the prediction

%%%%%%% Random Forest algorithm %%%%%%%

(It returns a list of random trees)
RandomForest(S,Attributes,Label,Gain,MaxDepth,T,FeatureSetSize)
where:
1. S is the set of examples
2. Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
3. Label are the possible lables (list)
4. Gain is the type of gain we want to use to select the best attribute (string): either information, me or gini
5. MaxDepth is the maximum depth we want the tree to have (integer)
6. T is the number of times we iterate the algorithm
7. FeatureSetSize is the size of the set of features that the algorithm randomly chooses to choose the Best attribute

prediction_RandomForest(Trees,s)
where:
1. Trees is a list of random trees
2. s is the example from which we want to make the prediction

RandTreeLearn(S, Attributes, Label, Gain, MaxDepth, FeatureSetSize)
it's a function that just returns a single random tree,
1. S is the set of examples
2. Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
3. Label are the possible lables (list)
4. Gain is the type of gain we want to use to select the best attribute (string): either information, me or gini
5. MaxDepth is the maximum depth we want the tree to have (integer)
6. FeatureSetSize is the size of the set of features that the algorithm randomly chooses to choose the Best attribute

%%%%%%% Decision Tree Weights algorithm %%%%%%%
It is just the same algorithm than the regular decision tree but it supports fractional examples.
