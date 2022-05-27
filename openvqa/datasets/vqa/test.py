import json

load = json.load(open('answer_dict.json', 'r'))

for key in load[0].keys():
    if key=='e':
        print(load[0][key])
