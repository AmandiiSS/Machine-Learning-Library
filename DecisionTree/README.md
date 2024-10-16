INSTRUCTIONS TO USE THE DECISION TREE ALGORITHM:
------------------------------------------------
I you call function ID3, it returns an object called Node, wich is the top node of the tree we are creating. 

%%%%%%% Class NODE %%%%%%%
Each node has associated in it:
1. The name of the attribute corresponding to that node if it is a node or empty if that node is a leaf.
2. A label corresponding to that leaf if its a leaf, and this is empty if the node is not a leaf.
3. A group of branches, you can access each one using the attribute value associated with each branch and every one of them has a node associated with them.
Branches are dictionaries with the attribute value as a key and a Node as a children

%%%%%%% ID3 algorithm %%%%%%%
You can call the algorithm by using:
ID3(S, Attributes, Label, Gain, MaxDepth)
where:
1. S is the set of examples (list (set of examples) of list (example) of ditionary (attributes and their "values" for the example) and strings (labels))
2. Attributes is the set of attributes available (dictionary key=attributes and values is a list of all values that the attribute key can have)
3. Label are the possible lables (list)
4. Gain is the type of gain we want to use to select the best attribute (string): either information, me or gini
5. MaxDepth is the maximum depth we want the tree to have (integer)