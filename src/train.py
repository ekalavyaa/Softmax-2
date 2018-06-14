import numpy as np
import torch 
from data_loader import DatasetCreate
from torch.utils.data import DataLoader
from modelClass import Model
import torch.optim as opt
data = DatasetCreate()
train_data = DataLoader(data, batch_size=6, shuffle=True , sampler=None, num_workers = 4)

# data =np.loadtxt('data/train.csv', delimiter=",", skiprows=1, dtype=np.float)

# train_x = torch.tensor(data[:,0:-1], requires_grad=True, dtype = torch.float)
# train_y = torch.tensor(data[:,[-1]], requires_grad=True, dtype = torch.float)

# """ model """
model = Model()
    
# loss_fun = torch.nn.CrossEntropyLoss(size_average=True)
# optimizer = opt.Adam(model.parameters(), lr=.01)
# for lables, data in zip(train_x, train_y):
#     y_predict = model(data)
#     print(y_predict)



# """ optimizer """
optimizer = opt.SGD(model.parameters(), lr=.04)

# """ loss function """
loss_fun = torch.nn.NLLLoss(size_average=False)


for batch_no, batch_data in enumerate(train_data):
    print("batch_no = ", batch_no)
    input, labels = batch_data
    train_x = torch.tensor(input, requires_grad=False, dtype = torch.float)
    train_y = torch.tensor(labels, requires_grad=False, dtype = torch.long)
    optimizer.zero_grad()
    y_predict = model(train_x)
    print(y_predict)
    print(train_y.view(6,))
    loss = loss_fun(y_predict, train_y.view(6,))
    loss.backward()
    optimizer.step()

# class labels = ["poistive", "negative"]
# for train_x, train_y  in data:
#     # print(train_x)
#     train_x = torch.tensor(train_x, requires_grad=False, dtype = torch.float)
#     train_y = torch.tensor([1,2], requires_grad=False, dtype = torch.float)
#     print(train_y, )
#     # optimizer.zero_grad()
#     # y_predict = model(train_x)
#     # print(y_predict)
#     # loss = loss_fun(y_predict, train_y)
#     # print(loss.item())
#     # loss.backward()
#     # optimizer.step
