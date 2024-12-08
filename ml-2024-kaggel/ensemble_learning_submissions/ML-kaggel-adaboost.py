import decisiontree
import statistics
from collections import Counter
import csv
import numpy
import adaboost

#Question: the change from numerical attributes to categorical (or binary) has to
#be done inside the algorithm or it can be done outside of it?

def add_example_to_S(example_list,quantiles_age,quantiles_fnlwgt,common_educationnum,quantiles_capitalgain,quantiles_capitalloss,quantiles_hoursperweek,workclass_0,workclass_1,education_0,education_1,maritalstatus_0,maritalstatus_1,occupation_0,occupation_1,relationship_0,relationship_1,race_0,race_1,sex_0,sex_1,nativecountry_0,nativecountry_1):
    #age
    if example_list[0] != "?":
        if float(example_list[0]) > quantiles_age[2]:
            age_value = "fourth" 
        elif float(example_list[0]) > quantiles_age[1]:
            age_value = "third"
        elif float(example_list[0]) > quantiles_age[0]:
            age_value = "second"
        else:
            age_value = "first"
    else:
        age_value = "second"
    #workclass
    if example_list[1] == "?":
        if example_list[14] == 0:
            workclass_value = common_attribute(workclass_0)
        else:
            workclass_value = common_attribute(workclass_1)
    else:
        workclass_value = example_list[1]
    #fnlwgt
    if example_list[2] != "?":
        if float(example_list[2]) > quantiles_fnlwgt[2]:
            fnlwgt_value = "fourth" 
        elif float(example_list[2]) > quantiles_fnlwgt[1]:
            fnlwgt_value = "third"
        elif float(example_list[2]) > quantiles_fnlwgt[0]:
            fnlwgt_value = "second"
        else:
            fnlwgt_value = "first"
    else:
        fnlwgt_value = "second"
    #education
    if example_list[3] == "?":
        if example_list[14] == 0:
            education_value = common_attribute(education_0)
        else:
            education_value = common_attribute(education_1)
    else:
        education_value = example_list[3]
    #education-num
    if example_list[4] == "?":
        educationnum_value = common_educationnum
    else:
        educationnum_value = example_list[4]
    #marital-status
    if example_list[5] == "?":
        if example_list[14] == 0:
            maritalstatus_value = common_attribute(maritalstatus_0)
        else:
            maritalstatus_value = common_attribute(maritalstatus_1)
    else:
        maritalstatus_value = example_list[5]
    #occupation
    if example_list[6] == "?":
        if example_list[14] == 0:
            occupation_value = common_attribute(occupation_0)
        else:
            occupation_value = common_attribute(occupation_1)
    else:
        occupation_value = example_list[6]
    #relationship
    if example_list[7] == "?":
        if example_list[14] == 0:
            relationship_value = common_attribute(relationship_0)
        else:
            relationship_value = common_attribute(relationship_1)
    else:
        relationship_value = example_list[7]
    #race
    if example_list[8] == "?":
        if example_list[14] == 0:
            race_value = common_attribute(race_0)
        else:
            race_value = common_attribute(race_1)
    else:
        race_value = example_list[8]
    #sex
    if example_list[9] == "?":
        if example_list[14] == 0:
            sex_value = common_attribute(sex_0)
        else:
            sex_value = common_attribute(sex_1)
    else:
        sex_value = example_list[9]
    #capital-gain
    if example_list[10] != "?":
        if float(example_list[10]) > quantiles_capitalgain[2]:
            capitalgain_value = "fourth" 
        elif float(example_list[10]) > quantiles_capitalgain[1]:
            capitalgain_value = "third"
        elif float(example_list[10]) > quantiles_capitalgain[0]:
            capitalgain_value = "second"
        else:
            capitalgain_value = "first"
    else:
        capitalgain_value = "second"
    #capital-loss
    if example_list[11] != "?":
        if float(example_list[11]) > quantiles_capitalloss[2]:
            capitalloss_value = "fourth" 
        elif float(example_list[11]) > quantiles_capitalloss[1]:
            capitalloss_value = "third"
        elif float(example_list[11]) > quantiles_capitalloss[0]:
            capitalloss_value = "second"
        else:
            capitalloss_value = "first"
    else:
        capitalloss_value = "second"
    #hours-per-week
    if example_list[12] != "?":
        if float(example_list[12]) > quantiles_hoursperweek[2]:
            hoursperweek_value = "fourth" 
        elif float(example_list[12]) > quantiles_hoursperweek[1]:
            hoursperweek_value = "third"
        elif float(example_list[12]) > quantiles_hoursperweek[0]:
            hoursperweek_value = "second"
        else:
            hoursperweek_value = "first"
    else:
        hoursperweek_value = "second"
    #native-country
    if example_list[13] == "?":
        if example_list[14] == 0:
            nativecountry_value = common_attribute(nativecountry_0)
        else:
            nativecountry_value = common_attribute(nativecountry_1)
    else:
        nativecountry_value = example_list[13]

    
    example = [{"age":age_value,
                "workclass":workclass_value,
                "fnlwgt":fnlwgt_value,
                "education":education_value,
                "education-num":educationnum_value,
                "marital-status":maritalstatus_value,
                "occupation":occupation_value,
                "relationship":relationship_value,
                "race":race_value,
                "sex":sex_value,
                "capital-gain":capitalgain_value,
                "capital-loss":capitalloss_value,
                "hours-per-week":hoursperweek_value,
                "native-country":nativecountry_value},
                example_list[14]]
    return example



#Construction and definition of the Attributes and Label parameters
Attributes = {
    #continous
    "age":["first","second","third","fourth"],
    "workclass":["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov","Without-pay","Never-worked"],
    #continous
    "fnlwgt":["first","second","third","fourth"],
    "education":["Bachelors","Some-college","11th","HS-grad","Prof-school","Assoc-acdm","Assoc-voc","9th","7th-8th","12th","Masters","1st-4th","10th","Doctorate","5th-6th","Preschool"],
    "education-num":["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"],
    "marital-status":["Married-civ-spouse","Divorced","Never-married","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"],
    "occupation":["Tech-support","Craft-repair","Other-service","Sales","Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv","Armed-Forces"],
    "relationship":["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"],
    "race":["White","Asian-Pac-Islander","Amer-Indian-Eskimo","Other","Black"],
    "sex":["Female","Male"],
    #continous
    "capital-gain":["first","second","third","fourth"],
    #continous
    "capital-loss":["first","second","third","fourth"],
    #continous
    "hours-per-week":["first","second","third","fourth"],
    "native-country":["United-States","Cambodia","England","Puerto-Rico","Canada","Germany","Outlying-US(Guam-USVI-etc)","India","Japan","Greece","South","China","Cuba","Iran","Honduras","Philippines","Italy","Poland","Jamaica","Vietnam","Mexico","Portugal","Ireland","France","Dominican-Republic","Laos","Ecuador","Taiwan","Haiti","Columbia","Hungary","Guatemala","Nicaragua","Scotland","Thailand","Yugoslavia","El-Salvador","Trinadad&Tobago","Peru","Hong","Holand-Netherlands"],
}
Label = ["0","1"]

#Getting information for each numerical value and getting the most common value for the attributes that can have a missing value "unknown"
#numerical (we will substitute the ? with the median)
age=[]
fnlwgt=[]
educationnum=[]
capitalgain=[]
capitalloss=[]
hoursperweek=[]
#categorical (we will substitute the ? with the most common value)
workclass_0=[]
workclass_1=[]
education_0=[]
education_1=[]
maritalstatus_0=[]
maritalstatus_1=[]
occupation_0=[]
occupation_1=[]
relationship_0=[]
relationship_1=[]
race_0=[]
race_1=[]
sex_0=[]
sex_1=[]
nativecountry_0=[]
nativecountry_1=[]



CSVfile = 'data/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        if example_list[0] != "?":
            age.append(float(example_list[0]))
        if example_list[1] != "?":
            if example_list[14] ==0:
                workclass_0.append(example_list[1])
            else:
                workclass_1.append(example_list[1])
        if example_list[2] != "?":
            fnlwgt.append(float(example_list[2]))
        if example_list[3] != "?":
            if example_list[14] ==0:
                education_0.append(example_list[3])
            else:
                education_1.append(example_list[3]) 
        if example_list[4] != "?":
            educationnum.append(example_list[4])
        if example_list[5] != "?":
            if example_list[14] ==0:
                maritalstatus_0.append(example_list[5])
            else:
                maritalstatus_1.append(example_list[5])
        if example_list[6] != "?":
            if example_list[14] ==0:
                occupation_0.append(example_list[6])
            else:
                occupation_1.append(example_list[6])
        if example_list[7] != "?":
            if example_list[14] ==0:
                relationship_0.append(example_list[7])
            else:
                relationship_1.append(example_list[7])
        if example_list[8] != "?":
            if example_list[14] ==0:
                race_0.append(example_list[8])
            else:
                race_1.append(example_list[8])
        if example_list[9] != "?":
            if example_list[14] ==0:
                sex_0.append(example_list[9])
            else:
                sex_1.append(example_list[9])
        if example_list[10] != "?":
            capitalgain.append(float(example_list[10]))
        if example_list[11] != "?":
            capitalloss.append(float(example_list[11]))
        if example_list[12] != "?":
            hoursperweek.append(float(example_list[12]))
        if example_list[13] != "?":
            if example_list[14] ==0:
                nativecountry_0.append(example_list[13])
            else:
                nativecountry_1.append(example_list[13])
        
    
quantiles_age = [0,0,0]
quantiles_age[0] = numpy.quantile(age,0.25)
quantiles_age[1] = statistics.median(age)
quantiles_age[2] = numpy.quantile(age,0.75)
quantiles_fnlwgt = [0,0,0]
quantiles_fnlwgt[0] = numpy.quantile(fnlwgt,0.25)
quantiles_fnlwgt[1] = statistics.median(fnlwgt)
quantiles_fnlwgt[2] = numpy.quantile(fnlwgt,0.75)
quantiles_capitalgain = [0,0,0]
quantiles_capitalgain[0] = numpy.quantile(capitalgain,0.25)
quantiles_capitalgain[1] = statistics.median(capitalgain)
quantiles_capitalgain[2] = numpy.quantile(capitalgain,0.75)
quantiles_capitalloss = [0,0,0]
quantiles_capitalloss[0] = numpy.quantile(capitalloss,0.25)
quantiles_capitalloss[1] = statistics.median(capitalloss)
quantiles_capitalloss[2] = numpy.quantile(capitalloss,0.75)
quantiles_hoursperweek = [0,0,0]
quantiles_hoursperweek[0] = numpy.quantile(hoursperweek,0.25)
quantiles_hoursperweek[1] = statistics.median(hoursperweek)
quantiles_hoursperweek[2] = numpy.quantile(hoursperweek,0.75)


def common_attribute(attribute_values):
    attributeValuesAndValues = Counter(attribute_values)
    return max(attributeValuesAndValues, key=attributeValuesAndValues.get)


# common_workclass=common_attribute(workclass)
# common_education=common_attribute(education)
common_educationnum=common_attribute(educationnum)
# common_maritalstatus=common_attribute(maritalstatus)
# common_occupation=common_attribute(occupation)
# common_relationship=common_attribute(relationship)
# common_race=common_attribute(race)
# common_sex=common_attribute(sex)
# common_nativecountry=common_attribute(nativecountry)
        

#Construction of the S parameter
S = []
CSVfile = 'data/train.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        S.append(add_example_to_S(example_list,quantiles_age,quantiles_fnlwgt,common_educationnum,quantiles_capitalgain,quantiles_capitalloss,quantiles_hoursperweek,workclass_0,workclass_1,education_0,education_1,maritalstatus_0,maritalstatus_1,occupation_0,occupation_1,relationship_0,relationship_1,race_0,race_1,sex_0,sex_1,nativecountry_0,nativecountry_1))
      

# Let's now run the code and get the results

# workclass=[]
# education=[]
# educationnum=[]
# maritalstatus=[]
# occupation=[]
# relationship=[]
# race=[]
# sex=[]
# nativecountry=[]

# CSVfile = 'data/test.csv'
# with open(CSVfile, 'r') as f:
#     for line in f:
#         example_list = line.strip().split(',')
#         example_list.pop(0)
#         if example_list[1] != "?":
#             workclass.append(example_list[1])
#         if example_list[3] != "?":
#             education.append(example_list[3])
#         if example_list[4] != "?":
#             educationnum.append(example_list[4])
#         if example_list[5] != "?":
#             maritalstatus.append(example_list[5])
#         if example_list[6] != "?":
#             occupation.append(example_list[6])
#         if example_list[7] != "?":
#             relationship.append(example_list[7])
#         if example_list[8] != "?":
#             race.append(example_list[8])
#         if example_list[9] != "?":
#             sex.append(example_list[9])
#         if example_list[13] != "?":
#             nativecountry.append(example_list[13])

# common_workclass=common_attribute(workclass)
# common_education=common_attribute(education)
# common_educationnum=common_attribute(educationnum)
# common_maritalstatus=common_attribute(maritalstatus)
# common_occupation=common_attribute(occupation)
# common_relationship=common_attribute(relationship)
# common_race=common_attribute(race)
# common_sex=common_attribute(sex)
# common_nativecountry=common_attribute(nativecountry)


gain = "gini"
max_depth = 2
data = []

#Include the initial weights in the examples 
weights_0 = 1/len(age)
for s in S:
    s.append(weights_0)


Trees_and_votes = adaboost.AdaBoost_decisiontree(S,Attributes,Label,gain,max_depth,50)
print("Tree built!")
CSVfile = 'data/test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_to_test = line.strip().split(',')
        identifier=example_to_test[0]
        example_to_test.pop(0)
        example_to_test.append("0") #This is only for the next function to work (and not have to rewrite everything again), but its not going to affect the results (we add a random "0" at the end as a "label")
        test = add_example_to_S(example_to_test,quantiles_age,quantiles_fnlwgt,common_educationnum,quantiles_capitalgain,quantiles_capitalloss,quantiles_hoursperweek,workclass_0,workclass_1,education_0,education_1,maritalstatus_0,maritalstatus_1,occupation_0,occupation_1,relationship_0,relationship_1,race_0,race_1,sex_0,sex_1,nativecountry_0,nativecountry_1)[0]
        predicted_label = adaboost.prediction_AdaBoost(Trees_and_votes,test,Label)
        data.append({"ID":identifier,"Prediction":predicted_label})
        #print(data)
        
with open('predictions/submission_Amanda_SS_adaboost_T50_2_gini_differentintervals_quantiles_?common-attrib-same-lab.csv', 'w', newline='') as file:
    fields = ["ID","Prediction"]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data)

       


