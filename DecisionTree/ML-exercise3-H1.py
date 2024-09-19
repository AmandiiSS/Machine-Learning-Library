import decisiontree

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list):
    example = [{"age":example_list[0],"maint":example_list[1],"doors":example_list[2],"persons":example_list[3],"lug_boot":example_list[4],"safety":example_list[5]},example_list[6]]
    return example

#Construction and definition of the Attributes and Label parameters
Attributes = {
    "age":["bigger","less"],
    "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["unknown","secondary","primary","tertiary"],
    "default":["yes","no"],
    "balance":["bigger","less"],
    "housing":["yes","no"],
    "loan":["yes","no"],
    "contact":["unknown","telephone","cellular"],
    "day":["bigger","less"],
    "month":["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "duration":["bigger","less"],
    "campaign":["bigger","less"],
    "pdays":["bigger","less","no"],
    "previous":["bigger","less"],
    "poutcome":["unknown","other","failure","success"],
}
Label = ["yes","no"]
#Construction of the S parameter
S = []
CSVfile = 'bank/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        S.append(add_example_to_S(example_list))
        #example is a list of all the attributes


#MaxDepth = 6
#Gain = 'info'
Tree_6_info = decisiontree.ID3(S, Attributes, Label, "info", 6)
print("Tree_6_info done")
#MaxDepth = 5
#Gain = 'info'
Tree_5_info = decisiontree.ID3(S, Attributes, Label, "info", 5)
print("Tree_5_info done")
#MaxDepth = 4
#Gain = 'info'
Tree_4_info = decisiontree.ID3(S, Attributes, Label, "info", 4)
print("Tree_4_info done")
#MaxDepth = 3
#Gain = 'info'
Tree_3_info = decisiontree.ID3(S, Attributes, Label, "info", 3)
print("Tree_3_info done")
#MaxDepth = 2
#Gain = 'info'
Tree_2_info = decisiontree.ID3(S, Attributes, Label, "info", 2)
print("Tree_2_info done")
#MaxDepth = 1
#Gain = 'info'
Tree_1_info = decisiontree.ID3(S, Attributes, Label, "info", 1)
print("Tree_1_info done")
###############################################
#MaxDepth = 6
#Gain = 'me'
Tree_6_me = decisiontree.ID3(S, Attributes, Label, "me", 6)
print("Tree_6_me done")
#MaxDepth = 5
#Gain = 'me'
Tree_5_me = decisiontree.ID3(S, Attributes, Label, "me", 5)
print("Tree_5_me done")
#MaxDepth = 4
#Gain = 'me'
Tree_4_me = decisiontree.ID3(S, Attributes, Label, "me", 4)
print("Tree_4_me done")
#MaxDepth = 3
#Gain = 'me'
Tree_3_me = decisiontree.ID3(S, Attributes, Label, "me", 3)
print("Tree_3_me done")
#MaxDepth = 2
#Gain = 'me'
Tree_2_me = decisiontree.ID3(S, Attributes, Label, "me", 2)
print("Tree_2_me done")
#MaxDepth = 1
#Gain = 'me'
Tree_1_me = decisiontree.ID3(S, Attributes, Label, "me", 1)
print("Tree_1_me done")
###############################################
#MaxDepth = 6
#Gain = 'gini'
Tree_6_gini = decisiontree.ID3(S, Attributes, Label, "gini", 6)
print("Tree_6_gini done")
#MaxDepth = 5
#Gain = 'gini'
Tree_5_gini = decisiontree.ID3(S, Attributes, Label, "gini", 5)
print("Tree_5_gini done")
#MaxDepth = 4
#Gain = 'gini'
Tree_4_gini = decisiontree.ID3(S, Attributes, Label, "gini", 4)
print("Tree_4_gini done")
#MaxDepth = 3
#Gain = 'gini'
Tree_3_gini = decisiontree.ID3(S, Attributes, Label, "gini", 3)
print("Tree_3_gini done")
#MaxDepth = 2
#Gain = 'gini'
Tree_2_gini = decisiontree.ID3(S, Attributes, Label, "gini", 2)
print("Tree_2_gini done")
#MaxDepth = 1
#Gain = 'gini'
Tree_1_gini = decisiontree.ID3(S, Attributes, Label, "gini", 1)
print("Tree_1_gini done")

