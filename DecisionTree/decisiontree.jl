

#S is the set of examples (list (set of examples) of list (example) of string (attributes values, order is important) or strings or ints (labels))
#Attributes is the set of attributes available (list, the order is important)
#Label are the lables (list)
#Gain is the type of gain we want to use to select the best attribute (string):
#   either information, me or gini
#MaxDepth is the maximum depth we want the tree to have (integer)

function ID3(S, Attributes, Label, Gain, MaxDepth)
    if #All examples have the same label or Attibutes is empty or MaxDepth = 0 
        #Return a leaf node with the most common label
    else 
        #Create a Root Node for the tree

        #We use the type of gain selected to choose the A from Attributes that best splits S
        BestA = best_attribute(S, Attributes, Gain, Label)

        #For each possible value v that A can take we: 
            #1. Add a new tree branch orresponding to A=v
            #2. Let Sv be the subset of examples with A=v:
                #If Sv is empty: we create a leaf node with the most common value of Label in S
                #Else below this branch, add the subtree ID3(Sv, Attibutes-{A}, Label, Gain, MaxDepth-1)
        
        #Return Root Node
    end

end

#This function selects and returns the best attribute to split the set of examples S
function best_attribute(S, Attibutes, Gain, Label)
    #Calculate the general gain, that is for all the examples we want to split
    if Gain == "gini"
        generalGain = giniGain(S)
    else if Gain == "me"
        generalGain = meGain(S)
    else 
        generalGain = informationGain(S)
    end
    #For each attribute, calculate each gain associated to the attribute and we want the
    #with max gain
    maxGain = 0
    maxA = Attributes[0]
    for A in Attributes
        #Get the examples with attribute == A
        S_A = #use a for to select the appropiate ones
        if Gain == "gini"
            attributeGain= giniGain(S_A)
        else if Gain == "me"
            attributeGain = meGain(S_A)
        else 
            attributeGain = informationGain(S_A)
        end
        if attributeGain > maxGain
            maxGain = attributeGain
            maxA = A
    end
    return maxA
end

#This function calculates and returns the information gain of the set of examples S
function informationGain(S)
    p = count(S[labels])

end

#This function calculates and returns the majority error gain of the set of examples S
function informationGain(S)
    
end

#This function calculates and returns the gini gain of the set of examples S
function informationGain(S)
    
end