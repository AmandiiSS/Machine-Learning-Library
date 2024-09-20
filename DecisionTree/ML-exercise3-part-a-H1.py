import decisiontree
import time
import statistics

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous):
    if float(example_list[0]) > media_age:
        age_value = "biggereq" 
    else:
        age_value = "less"
    if float(example_list[5]) > media_balance:
        balance_value = "biggereq" 
    else:
        balance_value = "less"
    if float(example_list[9]) > media_day:
        day_value = "biggereq" 
    else:
        day_value = "less"
    if float(example_list[11]) > media_duration:
        duration_value = "biggereq" 
    else:
        duration_value = "less"
    if float(example_list[12]) > media_campaign:
        campaign_value = "biggereq" 
    else:
        campaign_value = "less"
    if float(example_list[12]) == -1:
        pdays_value = "no" 
    elif float(example_list[12]) > media_pdays:
        pdays_value = "biggereq"
    else:
        pdays_value = "less"
    if float(example_list[13]) > media_previous:
        previous_value = "biggereq" 
    else:
        previous_value = "less"
    
    example = [{"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]},example_list[16]]
    return example

# def number_to_cartegories(example_list):
#     example = example_list.copy()
#     return example



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
      

# Let's now run the code and get the results

print("Prediction error for the trining data set: ")

total_examples = 5000
pred_error_info = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pred_error_me = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pred_error_gini = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
gain_list = ["info", "me", "gini"]
depth_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for gain in gain_list:
    for depth in depth_list:
        Tree = decisiontree.ID3(S, Attributes, Label, gain, depth)
        #print("Tree "+ str(depth) +" "+ gain +" done")
        CSVfile = 'bank/train.csv'
        with open(CSVfile, 'r') as f:
            for line in f:
                example_to_test = line.strip().split(',')
                if float(example_to_test[0]) > media_age:
                    age_value = "biggereq" 
                else:
                    age_value = "less"
                if float(example_to_test[5]) > media_balance:
                    balance_value = "biggereq" 
                else:
                    balance_value = "less"
                if float(example_to_test[9]) > media_day:
                    day_value = "biggereq" 
                else:
                    day_value = "less"
                if float(example_to_test[11]) > media_duration:
                    duration_value = "biggereq" 
                else:
                    duration_value = "less"
                if float(example_to_test[12]) > media_campaign:
                    campaign_value = "biggereq" 
                else:
                    campaign_value = "less"
                if float(example_to_test[12]) == -1:
                    pdays_value = "no" 
                elif float(example_to_test[12]) > media_pdays:
                    pdays_value = "biggereq"
                else:
                    pdays_value = "less"
                if float(example_to_test[13]) > media_previous:
                    previous_value = "biggereq" 
                else:
                    previous_value = "less"
                test = {"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]}

                predicted_label = decisiontree.prediction(Tree,test)
                if predicted_label == "yes":
                    print("YAAAAY")
                if (predicted_label != example_to_test[16]):
                    if (gain == "info"):
                        pred_error_info[depth-1] += 1/total_examples
                    elif (gain == "me"):
                        pred_error_me[depth-1] += 1/total_examples
                    elif (gain == "gini"):
                        pred_error_gini[depth-1] += 1/total_examples
    

print("Predicted error with information gain: ")
print(pred_error_info)
print("Predicted error with me gain: ")
print(pred_error_me)
print("Predicted error with gini gain: ")
print(pred_error_gini)



print("Prediction error for the test data set: ")

total_examples = 5000
pred_error_info = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pred_error_me = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
pred_error_gini = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
gain_list = ["info", "me", "gini"]
depth_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for gain in gain_list:
    for depth in depth_list:
        Tree = decisiontree.ID3(S, Attributes, Label, gain, depth)
        CSVfile = 'bank/test.csv'
        with open(CSVfile, 'r') as f:
            for line in f:
                example_to_test = line.strip().split(',')
                if float(example_to_test[0]) > media_age:
                    age_value = "biggereq" 
                else:
                    age_value = "less"
                if float(example_to_test[5]) > media_balance:
                    balance_value = "biggereq" 
                else:
                    balance_value = "less"
                if float(example_to_test[9]) > media_day:
                    day_value = "biggereq" 
                else:
                    day_value = "less"
                if float(example_to_test[11]) > media_duration:
                    duration_value = "biggereq" 
                else:
                    duration_value = "less"
                if float(example_to_test[12]) > media_campaign:
                    campaign_value = "biggereq" 
                else:
                    campaign_value = "less"
                if float(example_to_test[12]) == -1:
                    pdays_value = "no" 
                elif float(example_to_test[12]) > media_pdays:
                    pdays_value = "biggereq"
                else:
                    pdays_value = "less"
                if float(example_to_test[13]) > media_previous:
                    previous_value = "biggereq" 
                else:
                    previous_value = "less"
                test = {"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]}

                predicted_label = decisiontree.prediction(Tree,test)
                if (predicted_label != example_to_test[16]):
                    if (gain == "info"):
                        pred_error_info[depth-1] += 1/total_examples
                    elif (gain == "me"):
                        pred_error_me[depth-1] += 1/total_examples
                    elif (gain == "gini"):
                        pred_error_gini[depth-1] += 1/total_examples
    
       

print("Predicted error with information gain: ")
print(pred_error_info)
print("Predicted error with me gain: ")
print(pred_error_me)
print("Predicted error with gini gain: ")
print(pred_error_gini)






# Tree_8_info = decisiontree.ID3(S, Attributes, Label, "info", 8)
# print("Prediction error for the test data set: ")
# total_examples = 5000
# pred_error_info = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# CSVfile = 'bank/test.csv'
# with open(CSVfile, 'r') as f:
#     for line in f:
#         example_to_test = line.strip().split(',')
#         if float(example_to_test[0]) >= media_age:
#             age_value = "biggereq" 
#         else:
#             age_value = "less"
#         if float(example_to_test[5]) >= media_balance:
#             balance_value = "biggereq" 
#         else:
#             balance_value = "less"
#         if float(example_to_test[9]) >= media_day:
#             day_value = "biggereq" 
#         else:
#             day_value = "less"
#         if float(example_to_test[11]) >= media_duration:
#             duration_value = "biggereq" 
#         else:
#             duration_value = "less"
#         if float(example_to_test[12]) >= media_campaign:
#             campaign_value = "biggereq" 
#         else:
#             campaign_value = "less"
#         if float(example_to_test[12]) == -1:
#             pdays_value = "no" 
#         elif float(example_to_test[12]) >= media_pdays:
#             pdays_value = "biggereq"
#         else:
#             pdays_value = "less"
#         if float(example_to_test[13]) >= media_previous:
#             previous_value = "biggereq" 
#         else:
#             previous_value = "less"
#         test = {"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]}
#         if decisiontree.prediction(Tree_8_info,test) == "yes":
#             print("hola")
#         # info
#         if (decisiontree.prediction(Tree_8_info,test) != example_to_test[16]):
#             pred_error_info[7] += 1/total_examples

# print(pred_error_info)