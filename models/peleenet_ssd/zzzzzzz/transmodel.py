import torch
import xlrd

mapf = xlrd.open_workbook('Net_paramter_size_1.xlsx').sheets()[0]
mapdic = dict()
for i in range(mapf.nrows):
    key = mapf.row_values(i)[2].strip()
    val = mapf.row_values(i)[0].strip()
    mapdic[key] = val

caffemodel = torch.load('CaffeNet.pth')['state_dict']
caffemodel.pop('data.data')
caffemodel.pop('data.label')
newmodel = dict()
for key, value in caffemodel.items():
    newmodel[mapdic[key]] = value

# update
chkpt = torch.load('../peleenet.pth')
chkpt.update(newmodel)
torch.save(chkpt, 'transedmodel.pth')
