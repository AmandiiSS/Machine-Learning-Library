import statistics
import adaboost
import time

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
    if float(example_list[12]) == -1:
        pdays_value = "no" 
    elif float(example_list[12]) > media_pdays:
        pdays_value = "bigger"
    else:
        pdays_value = "lesseq"
    if float(example_list[13]) > media_previous:
        previous_value = "bigger" 
    else:
        previous_value = "lesseq"
    
    example = [{"age":age_value,"job":example_list[1],"marital":example_list[2],"education":example_list[3],"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":example_list[8],"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_list[15]},example_list[16]]
    return example


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
    "pdays":["bigger","lesseq","no"],
    "previous":["bigger","lesseq"],
    "poutcome":["unknown","other","failure","success"],
}
Label = ["no","yes"]

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

#Include the initial weights in the examples 
weights_0 = 1/len(age)
for s in S:
    s.append(weights_0)

# Let's now run the code and get the results
depth = 2
gain = "information"
total_examples = 5000

training_errors = []
test_errors = []

Trees_and_votes = adaboost.AdaBoost_decisiontree(S,Attributes,Label,gain,depth,500)

for i in range(500):
    print("Iteration: ", i)
    ########################### TRAINING DATA
    error_i_train = 0
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
            if float(example_to_test[13]) == -1:
                pdays_value = "no" 
            elif float(example_to_test[13]) > media_pdays:
                pdays_value = "bigger"
            else:
                pdays_value = "lesseq"
            if float(example_to_test[14]) > media_previous:
                previous_value = "bigger" 
            else:
                previous_value = "lesseq"
            test = {"age":age_value,"job":example_to_test[1],"marital":example_to_test[2],"education":example_to_test[3],"default":example_to_test[4],"balance":balance_value,"housing":example_to_test[6],"loan":example_to_test[7],"contact":example_to_test[8],"day":day_value,"month":example_to_test[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_to_test[15]}

            #Select the trees and votes up to i
            Trees_and_votes_i = Trees_and_votes[:i+1]
            predicted_label = adaboost.prediction_AdaBoost(Trees_and_votes_i,test,Label)
            if (predicted_label != example_to_test[16]):
                error_i_train += 1/total_examples

    ########################### TEST DATA
    error_i_test = 0
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
            if float(example_to_test[13]) == -1:
                pdays_value = "no" 
            elif float(example_to_test[13]) > media_pdays:
                pdays_value = "bigger"
            else:
                pdays_value = "lesseq"
            if float(example_to_test[14]) > media_previous:
                previous_value = "bigger" 
            else:
                previous_value = "lesseq"
            test = {"age":age_value,"job":example_to_test[1],"marital":example_to_test[2],"education":example_to_test[3],"default":example_to_test[4],"balance":balance_value,"housing":example_to_test[6],"loan":example_to_test[7],"contact":example_to_test[8],"day":day_value,"month":example_to_test[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":example_to_test[15]}

            Trees_and_votes_i = Trees_and_votes[:i+1]
            predicted_label = adaboost.prediction_AdaBoost(Trees_and_votes_i,test,Label)
            if (predicted_label != example_to_test[16]):
                error_i_test += 1/total_examples

    training_errors.append(error_i_train)
    test_errors.append(error_i_test)


print(training_errors)
print(test_errors)