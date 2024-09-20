import decisiontree
import time
import statistics

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous):
    if float(example_list[0]) > media_age:
        age_value = "bigger" 
    else:
        age_value = "lesseq"
    if float(example_list[5]) > media_balance:
        balance_value = "bigger" 
    else:
        balance_value = "lesseq"
    if float(example_list[9]) > media_day:
        day_value = "bigger" 
    else:
        day_value = "lesseq"
    if float(example_list[11]) > media_duration:
        duration_value = "bigger" 
    else:
        duration_value = "lesseq"
    if float(example_list[12]) > media_campaign:
        campaign_value = "bigger" 
    else:
        campaign_value = "lesseq"
    if float(example_list[12]) > media_pdays:
        pdays_value = "bigger"
    else:
        pdays_value = "lesseq"
    if float(example_list[13]) > media_previous:
        previous_value = "bigger" 
    else:
        previous_value = "lesseq"
    
    example = [{"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]},example_list[16]]
    return example

# def number_to_cartegories(example_list):
#     example = example_list.copy()
#     return example



#Construction and definition of the Attributes and Label parameters
Attributes = {
    "age":["bigger","lesseq"],
    "job":["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["unknown","secondary","primary","tertiary"],
    "default":["yes","no"],
    "balance":["bigger","lesseq"],
    "housing":["yes","no"],
    "loan":["yes","no"],
    "contact":["unknown","telephone","cellular"],
    "day":["bigger","lesseq"],
    "month":["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "duration":["bigger","lesseq"],
    "campaign":["bigger","lesseq"],
    "pdays":["bigger","lesseq"],
    "previous":["bigger","lesseq"],
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
                    age_value = "bigger" 
                else:
                    age_value = "lesseq"
                if float(example_to_test[5]) > media_balance:
                    balance_value = "bigger" 
                else:
                    balance_value = "lesseq"
                if float(example_to_test[9]) > media_day:
                    day_value = "bigger" 
                else:
                    day_value = "lesseq"
                if float(example_to_test[11]) > media_duration:
                    duration_value = "bigger" 
                else:
                    duration_value = "lesseq"
                if float(example_to_test[12]) > media_campaign:
                    campaign_value = "bigger" 
                else:
                    campaign_value = "lesseq"
                if float(example_to_test[12]) > media_pdays:
                    pdays_value = "bigger"
                else:
                    pdays_value = "lesseq"
                if float(example_to_test[13]) > media_previous:
                    previous_value = "bigger" 
                else:
                    previous_value = "lesseq"
                test = {"age":age_value,"job":example_to_test[1],"marital":example_to_test[2],"education":example_to_test[3],"default":example_to_test[4],"balance":balance_value,"housing":example_to_test[6],"loan":example_to_test[7],"contact":example_to_test[8],"day":day_value,"month":example_to_test[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_to_test[15]}

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
                    age_value = "bigger" 
                else:
                    age_value = "lesseq"
                if float(example_to_test[5]) > media_balance:
                    balance_value = "bigger" 
                else:
                    balance_value = "lesseq"
                if float(example_to_test[9]) > media_day:
                    day_value = "bigger" 
                else:
                    day_value = "lesseq"
                if float(example_to_test[11]) > media_duration:
                    duration_value = "bigger" 
                else:
                    duration_value = "lesseq"
                if float(example_to_test[12]) > media_campaign:
                    campaign_value = "bigger" 
                else:
                    campaign_value = "lesseq"
                if float(example_to_test[12]) > media_pdays:
                    pdays_value = "bigger"
                else:
                    pdays_value = "lesseq"
                if float(example_to_test[13]) > media_previous:
                    previous_value = "bigger" 
                else:
                    previous_value = "lesseq"
                test = {"age":age_value,"job":example_to_test[1],"marital":example_to_test[2],"education":example_to_test[3],"default":example_to_test[4],"balance":balance_value,"housing":example_to_test[6],"loan":example_to_test[7],"contact":example_to_test[8],"day":day_value,"month":example_to_test[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_to_test[15]}

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
