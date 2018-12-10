import json
import os

# dict_keys(['info', 'images', 'licenses', 'annotations', 'categories'])


# minival2014 = json.load(open('instances_minival2014.json', 'r'))
train2014 = json.load(open('instances_train2014.json', 'r'))
val2014 = json.load(open('instances_val2014.json', 'r'))

train_list = open('train.txt', 'r').read().splitlines()
train_list = [int(os.path.splitext(ll)[0][-7:]) for ll in train_list]
val_list = open('minival2014.txt', 'r').read().splitlines()
val_list = [int(os.path.splitext(ll)[0][-7:]) for ll in val_list]

val = dict()
train = dict()
val['info'] = val2014['info']
train['info'] = train2014['info']
val['licenses'] = val2014['licenses']
train['licenses'] = val2014['licenses']
val['categories'] = val2014['categories']
train['categories'] = val2014['categories']
val['images'] = list()
train['images'] = train2014['images']
val['annotations'] = list()
train['annotations'] = train2014['annotations']
# split val2014
for img in val2014['images']:
    if img['id'] in train_list:
        train['images'].append(img)
    elif img['id'] in val_list:
        val['images'].append(img)
for ann in val2014['annotations']:
    if ann['image_id'] in train_list:
        train['annotations'].append(ann)
    elif ann['image_id'] in val_list:
        val['annotations'].append(ann)

print(len(train['images']))
print(len(val['images']))
print(len(train_list))
print(len(val_list))

json.dump(train, open('train.json', 'w'))
json.dump(val, open('val.json', 'w'))
