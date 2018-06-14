import torch 

train_y = torch.tensor([0,1], dtype=torch.long, requires_grad= False)
y = torch.tensor([[1,2],[3,3]], dtype=torch.float, requires_grad= False)
loss_fun = torch.nn.CrossEntropyLoss()
loss = loss_fun(y, train_y)
print(loss.item())
