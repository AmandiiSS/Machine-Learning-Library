pred1 = []
CSVfile = 'predictions/submission_Amanda_SS_decisiontree_3_gini_differentintervals_educationnum.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        pred1.append(example_list[1])

pred2 = []
CSVfile = 'predictions/submission_Amanda_SS_decisiontree_3_gini_differentintervals_educationnum_commonlabels_test.csv'
with open(CSVfile, 'r') as f:
    for line in f:
        example_list = line.strip().split(',')
        pred2.append(example_list[1])

diff = 0
for i in pred1:
    if i != pred2[1]:
        diff += 1

print(diff)
print(len(pred1))
print((diff/len(pred1))*100)
