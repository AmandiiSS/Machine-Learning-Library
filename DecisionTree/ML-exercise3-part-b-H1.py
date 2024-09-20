import decisiontree
import time
import statistics
from collections import Counter

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous,common_job,common_education,common_contact,common_poutcome):
    if float(example_list[0]) > media_age:
        age_value = "bigger" 
    else:
        age_value = "lesseq"
    # if example_list[1] == "unkown":
    #     example_list[1] = common_attribute_value()
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
    if example_list[1] == "unknown":
        job_value = common_job
    else:
        job_value = example_list[1]
    if example_list[3] == "unknown":
        education_value = common_education
    else:
        education_value = example_list[3]
    if example_list[8] == "unknown":
        contact_value = common_contact
    else:
        contact_value = example_list[8]
    if example_list[15] == "unknown":
        poutcome_value = common_poutcome
    else:
        poutcome_value = example_list[15]
    
    example = [{"age":age_value,"job":job_value,"marital":example_list[2],"education":education_value,"default":example_list[4],"balance":balance_value,"housing":example_list[6],"loan":example_list[7],"contact":contact_value,"day":day_value,"month":example_list[10],"duration":duration_value,"campaign":campaign_value,"pdays":pdays_value,"previous":previous_value,"poutcome":poutcome_value},example_list[16]]
    return example



#Construction and definition of the Attributes and Label parameters
Attributes = {
    "age":["bigger","lesseq"],
    "job":["admin.","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"],
    "marital":["married","divorced","single"],
    "education":["secondary","primary","tertiary"],
    "default":["yes","no"],
    "balance":["bigger","lesseq"],
    "housing":["yes","no"],
    "loan":["yes","no"],
    "contact":["telephone","cellular"],
    "day":["bigger","lesseq"],
    "month":["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    "duration":["bigger","lesseq"],
    "campaign":["bigger","lesseq"],
    "pdays":["bigger","lesseq","no"],
    "previous":["bigger","lesseq"],
    "poutcome":["other","failure","success"],
}
Label = ["yes","no"]

#Getting the media for each numerical value and getting the most common value for the attributes that can have a missing value "unknown"
age=[]
job=[]
education=[]
balance=[]
contact=[]
day=[]
duration=[]
campaign=[]
pdays=[]
previous=[]
poutcome=[]


CSVfile = 'bank/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        age.append(float(example_list[0]))
        if example_list[1] != "unknown":
            job.append(example_list[1])
        if example_list[3] != "unknown":
            education.append(example_list[3])
        balance.append(float(example_list[5]))
        if example_list[8] != "unknown":
            contact.append(example_list[8])
        day.append(float(example_list[9]))
        duration.append(float(example_list[11]))
        campaign.append(float(example_list[12]))
        if float(example_list[13]) != -1:
            pdays.append(float(example_list[13]))
        previous.append(float(example_list[14]))
        if example_list[15] != "unknown":
            poutcome.append(example_list[15])

media_age = statistics.median(age)
media_balance = statistics.median(balance)
media_day = statistics.median(day)
media_duration = statistics.median(duration)
media_campaign = statistics.median(campaign)
media_pdays = statistics.median(pdays)
media_previous = statistics.median(previous)

def common_attribute(attribute_values):
    attributeValuesAndValues = Counter(attribute_values)
    return max(attributeValuesAndValues, key=attributeValuesAndValues.get)

common_job=common_attribute(job)
common_education=common_attribute(education)
common_contact=common_attribute(contact)
common_poutcome=common_attribute(poutcome)
        

#Construction of the S parameter
S = []
CSVfile = 'bank/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        S.append(add_example_to_S(example_list,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous,common_job,common_education,common_contact,common_poutcome))
      

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
                test = add_example_to_S(example_to_test,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous,common_job,common_education,common_contact,common_poutcome)[0]
                
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

job=[]
education=[]
contact=[]
poutcome=[]

CSVfile = 'bank/test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        if example_list[1] != "unknown":
            job.append(example_list[1])
        if example_list[3] != "unknown":
            education.append(example_list[3])
        if example_list[8] != "unknown":
            contact.append(example_list[8])
        if example_list[15] != "unknown":
            poutcome.append(example_list[15])

common_job=common_attribute(job)
common_education=common_attribute(education)
common_contact=common_attribute(contact)
common_poutcome=common_attribute(poutcome)


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
                test = add_example_to_S(example_to_test,media_age,media_balance,media_day,media_duration,media_campaign,media_pdays,media_previous,common_job,common_education,common_contact,common_poutcome)[0]

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
