import decisiontree
import time
import statistics

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous):
    if float(example_list[0]) >= media_age:
        age_value = "biggereq" 
    else:
        age_value = "less"
    if float(example_list[5]) >= media_balance:
        balance_value = "biggereq" 
    else:
        balance_value = "less"
    if float(example_list[9]) >= media_day:
        day_value = "biggereq" 
    else:
        day_value = "less"
    if float(example_list[11]) >= media_duration:
        duration_value = "biggereq" 
    else:
        duration_value = "less"
    if float(example_list[12]) >= media_campaign:
        campaign_value = "biggereq" 
    else:
        campaign_value = "less"
    if float(example_list[12]) == -1:
        pdays_value = "no" 
    elif float(example_list[12]) >= media_pdays:
        pdays_value = "biggereq"
    else:
        pdays_value = "less"
    if float(example_list[13]) >= media_previous:
        previous_value = "biggereq" 
    else:
        previous_value = "less"
    
    example = [{"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]},example_list[16]]
    return example

#Construction and definition of the Attributes and Label parameters
Attributes = {
    "age":["biggereq","less"],
    "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["unknown","secondary","primary","tertiary"],
    "default":["yes","no"],
    "balance":["biggereq","less"],
    "housing":["yes","no"],
    "loan":["yes","no"],
    "contact":["unknown","telephone","cellular"],
    "day":["biggereq","less"],
    "month":["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "duration":["biggereq","less"],
    "campaign":["biggereq","less"],
    "pdays":["biggereq","less","no"],
    "previous":["biggereq","less"],
    "poutcome":["unknown","other","failure","success"],
}
Label = ["yes","no"]

#Getting the media for each numerical value
age=[]
balance=[]
day=[]
duration=[]
campaign=[]
pdays=[]
previous=[]

CSVfile = 'bank/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        age.append(float(example_list[0]))
        balance.append(float(example_list[5]))
        day.append(float(example_list[9]))
        duration.append(float(example_list[11]))
        campaign.append(float(example_list[12]))
        if float(example_list[13]) != -1:
            pdays.append(float(example_list[13]))
        previous.append(float(example_list[14]))

media_age = statistics.median(age)
media_balance = statistics.median(balance)
media_day = statistics.median(day)
media_duration = statistics.median(duration)
media_campaign = statistics.median(campaign)
media_pdays = statistics.median(pdays)
media_previous = statistics.median(previous)
        

#Construction of the S parameter
S = []
CSVfile = 'bank/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        S.append(add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous))
      



# #MaxDepth = 6
# #Gain = 'info'
# Tree_6_info = decisiontree.ID3(S, Attributes, Label, "info", 6)
# print("Tree_6_info done")
# #MaxDepth = 5
# #Gain = 'info'
# Tree_5_info = decisiontree.ID3(S, Attributes, Label, "info", 5)
# print("Tree_5_info done")
# #MaxDepth = 4
# #Gain = 'info'
# Tree_4_info = decisiontree.ID3(S, Attributes, Label, "info", 4)
# print("Tree_4_info done")
# #MaxDepth = 3
# #Gain = 'info'
# Tree_3_info = decisiontree.ID3(S, Attributes, Label, "info", 3)
# print("Tree_3_info done")
# #MaxDepth = 2
# #Gain = 'info'
# Tree_2_info = decisiontree.ID3(S, Attributes, Label, "info", 2)
# print("Tree_2_info done")
# #MaxDepth = 1
# #Gain = 'info'
# Tree_1_info = decisiontree.ID3(S, Attributes, Label, "info", 1)
# print("Tree_1_info done")
# ###############################################
# #MaxDepth = 6
# #Gain = 'me'
# Tree_6_me = decisiontree.ID3(S, Attributes, Label, "me", 6)
# print("Tree_6_me done")
# #MaxDepth = 5
# #Gain = 'me'
# Tree_5_me = decisiontree.ID3(S, Attributes, Label, "me", 5)
# print("Tree_5_me done")
# #MaxDepth = 4
# #Gain = 'me'
# Tree_4_me = decisiontree.ID3(S, Attributes, Label, "me", 4)
# print("Tree_4_me done")
# #MaxDepth = 3
# #Gain = 'me'
# Tree_3_me = decisiontree.ID3(S, Attributes, Label, "me", 3)
# print("Tree_3_me done")
# #MaxDepth = 2
# #Gain = 'me'
# Tree_2_me = decisiontree.ID3(S, Attributes, Label, "me", 2)
# print("Tree_2_me done")
# #MaxDepth = 1
# #Gain = 'me'
# Tree_1_me = decisiontree.ID3(S, Attributes, Label, "me", 1)
# print("Tree_1_me done")
# ###############################################
# #MaxDepth = 6
# #Gain = 'gini'
# Tree_6_gini = decisiontree.ID3(S, Attributes, Label, "gini", 6)
# print("Tree_6_gini done")
# #MaxDepth = 5
# #Gain = 'gini'
# Tree_5_gini = decisiontree.ID3(S, Attributes, Label, "gini", 5)
# print("Tree_5_gini done")
# #MaxDepth = 4
# #Gain = 'gini'
# Tree_4_gini = decisiontree.ID3(S, Attributes, Label, "gini", 4)
# print("Tree_4_gini done")
# #MaxDepth = 3
# #Gain = 'gini'
# Tree_3_gini = decisiontree.ID3(S, Attributes, Label, "gini", 3)
# print("Tree_3_gini done")
# #MaxDepth = 2
# #Gain = 'gini'
# Tree_2_gini = decisiontree.ID3(S, Attributes, Label, "gini", 2)
# print("Tree_2_gini done")
# #MaxDepth = 1
# #Gain = 'gini'
# Tree_1_gini = decisiontree.ID3(S, Attributes, Label, "gini", 1)
# print("Tree_1_gini done")

