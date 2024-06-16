# print all the model*dataset combinations present
import os
import json

if __name__=='__main__':
    ls = os.listdir('.')
    ls = [i for i in ls if i.startswith('out.') and not i.startswith('out.range')]
    types = [i.split('.')[1] for i in ls]
    dataset = [i.split('.')[2] for i in ls]
    model = ['.'.join(i.split('.')[3:]) for i in ls]
    res = {}
    for i in range(len(ls)):
        if types[i] not in res:
            res[types[i]] = {}
        if dataset[i] not in res[types[i]]:
            res[types[i]][dataset[i]] = []
        if('.'.join(ls[i].split('.')[:1] + ['range'] + ls[i].split('.')[1:]) in os.listdir('.')):
            model[i] = model[i] + ' (range)'
        res[types[i]][dataset[i]].append(model[i])
    print(json.dumps(res, indent=4))