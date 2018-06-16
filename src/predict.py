import torch

model = torch.load('model/model.pt')
model.eval()
input = torch.tensor([[80,81,81]], dtype= torch.float)
ypred = model(input)

values, pred = ypred.max(1)
print("values = ", values)
print("pred = ", pred)