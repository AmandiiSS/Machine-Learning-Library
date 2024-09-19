
import decisiontree

################# Data example

S =[
    [{"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Weak"},"No"],
    [{"Outlook":"Sunny","Temperature":"Hot","Humidity":"High","Wind":"Strong"},"No"],
    [{"Outlook":"Overcast","Temperature":"Hot","Humidity":"High","Wind":"Weak"},"Yes"],
    [{"Outlook":"Rainy","Temperature":"Mild","Humidity":"High","Wind":"Weak"},"Yes"],
    [{"Outlook":"Rainy","Temperature":"Cool","Humidity":"Normal","Wind":"Weak"},"Yes"],
    [{"Outlook":"Rainy","Temperature":"Cool","Humidity":"Normal","Wind":"Strong"},"No"],
    [{"Outlook":"Overcast","Temperature":"Cool","Humidity":"Normal","Wind":"Strong"},"Yes"],
    [{"Outlook":"Sunny","Temperature":"Mild","Humidity":"High","Wind":"Weak"},"No"],
    [{"Outlook":"Sunny","Temperature":"Cool","Humidity":"Normal","Wind":"Weak"},"Yes"],
    [{"Outlook":"Rainy","Temperature":"Mild","Humidity":"Normal","Wind":"Weak"},"Yes"],
    [{"Outlook":"Sunny","Temperature":"Mild","Humidity":"Normal","Wind":"Strong"},"Yes"],
    [{"Outlook":"Overcast","Temperature":"Mild","Humidity":"High","Wind":"Strong"},"Yes"],
    [{"Outlook":"Overcast","Temperature":"Hot","Humidity":"Normal","Wind":"Weak"},"Yes"],
    [{"Outlook":"Rainy","Temperature":"Mild","Humidity":"High","Wind":"Strong"},"No"]
]
Attributes = {
    "Outlook":["Sunny","Overcast","Rainy"],
    "Temperature":["Hot","Mild","Cool"],
    "Humidity":["High","Normal","Low"],
    "Wind":["Strong","Weak"],
}
Label = ["No","Yes"]
Gain = 'info'
MaxDepth = 1

########################### Tree construction

Tree = decisiontree.ID3(S, Attributes, Label, Gain, MaxDepth)


