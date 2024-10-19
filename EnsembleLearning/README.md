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
