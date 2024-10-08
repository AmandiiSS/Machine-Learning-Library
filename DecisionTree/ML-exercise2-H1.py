import decisiontree

def add_example_to_S(example_list):
    example = [{"buying":example_list[0],"maint":example_list[1],"doors":example_list[2],"persons":example_list[3],"lug_boot":example_list[4],"safety":example_list[5]},example_list[6]]
    return example

#Construction and definition of the Attributes and Label parameters
Attributes = {
    "buying":["vhigh","high","med","low"],
    "maint":["vhigh","high","med","low"],
    "doors":["2","3","4","5more"],
    "persons":["2","4","more"],
    "lug_boot":["small","med","big"],
    "safety":["low","med","high"]
}
Label = ["unacc","acc","good","vgood"]
#Construction of the S parameter
S = []
CSVfile = 'car/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        S.append(add_example_to_S(example_list))
        #example is a list of all the attributes

######################################################## TRAINING
#MaxDepth = 6
#Gain = 'info'
Tree_6_info = decisiontree.ID3(S, Attributes, Label, "info", 6)
# print("Tree_6_info done")
#MaxDepth = 5
#Gain = 'info'
Tree_5_info = decisiontree.ID3(S, Attributes, Label, "info", 5)
# print("Tree_5_info done")
#MaxDepth = 4
#Gain = 'info'
Tree_4_info = decisiontree.ID3(S, Attributes, Label, "info", 4)
# print("Tree_4_info done")
# MaxDepth = 3
# Gain = 'info'
Tree_3_info = decisiontree.ID3(S, Attributes, Label, "info", 3)
# print("Tree_3_info done")
#MaxDepth = 2
#Gain = 'info'
Tree_2_info = decisiontree.ID3(S, Attributes, Label, "info", 2)
# print("Tree_2_info done")
#MaxDepth = 1
#Gain = 'info'
Tree_1_info = decisiontree.ID3(S, Attributes, Label, "info", 1)
# print("Tree_1_info done")
##############################################
#MaxDepth = 6
#Gain = 'me'
Tree_6_me = decisiontree.ID3(S, Attributes, Label, "me", 6)
# print("Tree_6_me done")
#MaxDepth = 5
#Gain = 'me'
Tree_5_me = decisiontree.ID3(S, Attributes, Label, "me", 5)
# print("Tree_5_me done")
#MaxDepth = 4
#Gain = 'me'
Tree_4_me = decisiontree.ID3(S, Attributes, Label, "me", 4)
# print("Tree_4_me done")
#MaxDepth = 3
#Gain = 'me'
Tree_3_me = decisiontree.ID3(S, Attributes, Label, "me", 3)
# print("Tree_3_me done")
#MaxDepth = 2
#Gain = 'me'
Tree_2_me = decisiontree.ID3(S, Attributes, Label, "me", 2)
# print("Tree_2_me done")
#MaxDepth = 1
#Gain = 'me'
Tree_1_me = decisiontree.ID3(S, Attributes, Label, "me", 1)
# print("Tree_1_me done")
###############################################
#MaxDepth = 6
#Gain = 'gini'
Tree_6_gini = decisiontree.ID3(S, Attributes, Label, "gini", 6)
# print("Tree_6_gini done")
#MaxDepth = 5
#Gain = 'gini'
Tree_5_gini = decisiontree.ID3(S, Attributes, Label, "gini", 5)
# print("Tree_5_gini done")
#MaxDepth = 4
#Gain = 'gini'
Tree_4_gini = decisiontree.ID3(S, Attributes, Label, "gini", 4)
# print("Tree_4_gini done")
#MaxDepth = 3
#Gain = 'gini'
Tree_3_gini = decisiontree.ID3(S, Attributes, Label, "gini", 3)
# print("Tree_3_gini done")
#MaxDepth = 2
#Gain = 'gini'
Tree_2_gini = decisiontree.ID3(S, Attributes, Label, "gini", 2)
# print("Tree_2_gini done")
#MaxDepth = 1
#Gain = 'gini'
Tree_1_gini = decisiontree.ID3(S, Attributes, Label, "gini", 1)
# print("Tree_1_gini done")

######################################################## PREDICT AND COMPARE TEST DATA SETS
print("Prediction error for the test data set: ")

total_exaples_to_test = 728
pred_error_info = [0,0,0,0,0,0]
pred_error_me = [0,0,0,0,0,0]
pred_error_gini = [0,0,0,0,0,0]
CSVfile = 'car/test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_to_test = line.strip().split(',')
        test = {"buying":example_to_test[0],"maint":example_to_test[1],"doors":example_to_test[2],"persons":example_to_test[3],"lug_boot":example_to_test[4],"safety":example_to_test[5]}
        # info
        if (decisiontree.prediction(Tree_1_info,test) != example_to_test[6]):
            pred_error_info[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_info,test) != example_to_test[6]):
            pred_error_info[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_info,test) != example_to_test[6]):
            pred_error_info[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_info,test) != example_to_test[6]):
            pred_error_info[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_info,test) != example_to_test[6]):
            pred_error_info[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_info,test) != example_to_test[6]):
            pred_error_info[5] += 1/total_exaples_to_test
        # me
        if (decisiontree.prediction(Tree_1_me,test) != example_to_test[6]):
            pred_error_me[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_me,test) != example_to_test[6]):
            pred_error_me[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_me,test) != example_to_test[6]):
            pred_error_me[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_me,test) != example_to_test[6]):
            pred_error_me[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_me,test) != example_to_test[6]):
            pred_error_me[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_me,test) != example_to_test[6]):
            pred_error_me[5] += 1/total_exaples_to_test
        # gini
        if (decisiontree.prediction(Tree_1_gini,test) != example_to_test[6]):
            pred_error_gini[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_gini,test) != example_to_test[6]):
            pred_error_gini[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_gini,test) != example_to_test[6]):
            pred_error_gini[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_gini,test) != example_to_test[6]):
            pred_error_gini[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_gini,test) != example_to_test[6]):
            pred_error_gini[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_gini,test) != example_to_test[6]):
            pred_error_gini[5] += 1/total_exaples_to_test


print("Predicted error with information gain: ")
print(pred_error_info)
print("Predicted error with me gain: ")
print(pred_error_me)
print("Predicted error with gini gain: ")
print(pred_error_gini)


######################################################## PREDICT AND COMPARE TEST DATA SETS
print("Prediction error for the train data set: ")

total_exaples_to_test = 1000
pred_error_info = [0,0,0,0,0,0]
pred_error_me = [0,0,0,0,0,0]
pred_error_gini = [0,0,0,0,0,0]
CSVfile = 'car/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_to_test = line.strip().split(',')
        test = {"buying":example_to_test[0],"maint":example_to_test[1],"doors":example_to_test[2],"persons":example_to_test[3],"lug_boot":example_to_test[4],"safety":example_to_test[5]}
        # info
        if (decisiontree.prediction(Tree_1_info,test) != example_to_test[6]):
            pred_error_info[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_info,test) != example_to_test[6]):
            pred_error_info[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_info,test) != example_to_test[6]):
            pred_error_info[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_info,test) != example_to_test[6]):
            pred_error_info[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_info,test) != example_to_test[6]):
            pred_error_info[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_info,test) != example_to_test[6]):
            pred_error_info[5] += 1/total_exaples_to_test
        # me
        if (decisiontree.prediction(Tree_1_me,test) != example_to_test[6]):
            pred_error_me[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_me,test) != example_to_test[6]):
            pred_error_me[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_me,test) != example_to_test[6]):
            pred_error_me[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_me,test) != example_to_test[6]):
            pred_error_me[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_me,test) != example_to_test[6]):
            pred_error_me[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_me,test) != example_to_test[6]):
            pred_error_me[5] += 1/total_exaples_to_test
        # gini
        if (decisiontree.prediction(Tree_1_gini,test) != example_to_test[6]):
            pred_error_gini[0] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_2_gini,test) != example_to_test[6]):
            pred_error_gini[1] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_3_gini,test) != example_to_test[6]):
            pred_error_gini[2] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_4_gini,test) != example_to_test[6]):
            pred_error_gini[3] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_5_gini,test) != example_to_test[6]):
            pred_error_gini[4] += 1/total_exaples_to_test
        if (decisiontree.prediction(Tree_6_gini,test) != example_to_test[6]):
            pred_error_gini[5] += 1/total_exaples_to_test


print("Predicted error with information gain: ")
print(pred_error_info)
print("Predicted error with me gain: ")
print(pred_error_me)
print("Predicted error with gini gain: ")
print(pred_error_gini)