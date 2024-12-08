import decisiontree_weigths
import math

#S is the set of examples. The weights will be changed there directly.
#T is the number of times we iterate the algorithm
def AdaBoost_decisiontree(S,Attributes,Label,gain,depth,T):
    #In this variable we will save each Tree and vote associated to it from each iteration
    Trees_and_votes = []
    for t in range(T):
        Tree_t = decisiontree_weigths.ID3_weights(S, Attributes, Label, gain, depth)
        error_t = 0
        for s in S:
            pred_t = decisiontree_weigths.prediction(Tree_t, s[0])
            if pred_t != s[1]:
                error_t += s[2]
        if (error_t == 0):
            return [1, Tree_t]
        if (error_t == 1):
            error_t = 0.99999999
        #This is the alpha
        vote_t = 0.5*math.log((1-error_t)/error_t)
        Trees_and_votes.append([vote_t,Tree_t])
        normalize = 0
        for s in S:
            pred_t = decisiontree_weigths.prediction(Tree_t, s[0])
            if pred_t != s[1]:
                s[2] = s[2]*math.exp(vote_t)
                normalize += s[2]
            else:
                s[2] = s[2]*math.exp(-vote_t)
                normalize += s[2]
        for s in S:
            s[2] = s[2]/normalize
    return Trees_and_votes
        

#Trees_and_votes = list of [vote, Tree]
#s example to test
#Label the two possible labels
def prediction_AdaBoost(Trees_and_votes,s, Label):
    sum_pred = 0
    for tv in Trees_and_votes:
        prediction_tree = decisiontree_weigths.prediction(tv[1], s)
        if (prediction_tree == Label[0]):
            pred_t = -1
        else: 
            pred_t = 1
        sum_pred += tv[0]*pred_t
    if(sum_pred <0):
        return Label[0]
    else:
        return Label[1]

