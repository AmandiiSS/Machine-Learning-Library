#import concurrent.futures
import statistics
import bagged
import time
import decisiontree
import random

def One_bagged_decisiontree(S,Attributes,Label,gain,depth):
    #In this variable we will save each Tree and vote associated to it from each iteration
    random_numbers = []
    for i in range(5000):
        random_numbers.append(random.randrange(0,5000))
    Sb = select_examples_b(S, random_numbers)
    OneMoreTrees = decisiontree.ID3(Sb, Attributes, Label, gain, depth)
    return OneMoreTrees


#This function returns a subset Sb of all examples with number associated from a list of num
def select_examples_b(S, random_numbers):
    Sb = []
    for num in random_numbers:
        Sb.append(S[num])
    return Sb

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

# Let's now run the code and get the results

depth = 50
gain = "information"
total_examples = 1000

training_errors = []
test_errors = []
Bagged_Predictors = []
Single_Trees = []

for i in range(100):
    print("Iteration: ", i)

    #Randomly select a 1000 examples without reposition from S
    S_rand = random.sample(S, 1000)

    Trees_bagged = bagged.Bagged_decisiontree(S_rand,Attributes,Label,gain,depth,500)
    Bagged_Predictors.append(Trees_bagged)
    Single_Trees.append(Trees_bagged[0])

########################### TEST DATA
bias_st_all = []
samp_var_st_all = []
bias_bt_all = []
samp_var_bt_all = []

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

        if example_to_test[16] == "no":
            real_label = -1
        else:
            real_label = 1
        # SINGLE TREE
        average_pred_st_i = 0
        for st in Single_Trees:
            predicted_label_sing_tree = decisiontree.prediction(st, test) 
            if predicted_label_sing_tree == "no":
                average_pred_st_i -= 1
            else:
                average_pred_st_i += 1
        average_pred_st_i = average_pred_st_i/100
        bias_st_i = pow(average_pred_st_i-real_label,2)
        var_sum_st_i = 0
        for st in Single_Trees:
            predicted_label_sing_tree = decisiontree.prediction(st, test) 
            if predicted_label_sing_tree == "no":
                var_sum_st_i += pow((-1-average_pred_st_i),2)
            else:
                var_sum_st_i += pow((1-average_pred_st_i),2)
        samp_var_st_i = (1/99)*var_sum_st_i
        bias_st_all.append(bias_st_i)
        samp_var_st_all.append(samp_var_st_i)

        #BAGGED TREES
        average_pred_bt_i = 0
        for bt in Bagged_Predictors:
            predicted_label_bagg_tree = bagged.prediction_Bagging(bt, test) 
            if predicted_label_bagg_tree == "no":
                average_pred_bt_i -= 1
            else:
                average_pred_bt_i += 1
        average_pred_bt_i = average_pred_bt_i/100
        bias_bt_i = pow(average_pred_bt_i-real_label,2)
        var_sum_bt_i = 0
        for bt in Bagged_Predictors:
            predicted_label_bagg_tree = bagged.prediction_Bagging(bt, test) 
            if predicted_label_bagg_tree == "no":
                var_sum_bt_i += pow((-1-average_pred_bt_i),2)
            else:
                var_sum_bt_i += pow((1-average_pred_bt_i),2)
        samp_var_bt_i = (1/99)*var_sum_bt_i
        bias_bt_all.append(bias_bt_i)
        samp_var_bt_all.append(samp_var_bt_i)
        

#The final bias and variance for the single trees
bias_st_final = sum(bias_st_all)/len(bias_st_all)
sample_var_st_final = sum(samp_var_st_all)/len(samp_var_st_all)
expected_error_wrt_st = bias_st_final + sample_var_st_final
#The final bias and variance for the bagged trees
bias_bt_final = sum(bias_bt_all)/len(bias_bt_all)
sample_var_bt_final = sum(samp_var_bt_all)/len(samp_var_bt_all)
expected_error_wrt_bt = bias_bt_final + sample_var_bt_final

print("Single tree results:")
print("Bias: ", bias_st_final)
print("Variance: ", sample_var_st_final)
print("Expected error wrt: ", expected_error_wrt_st)

print("Bagged trees results:")
print("Bias: ", bias_bt_final)
print("Variance: ", sample_var_bt_final)
print("Expected error wrt: ", expected_error_wrt_bt)



