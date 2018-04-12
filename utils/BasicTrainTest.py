#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import os.path as osp

def test(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0

    for i, (batch, label) in enumerate(test_loader):
        batch = batch.cuda()
        output = model(Variable(batch))
        pred_label = output.data.max(1)[1] 
        test_correct += pred_label.cpu().eq(label).sum() 
        test_total += label.size(0)

    model.train()
    return round( float(test_correct) / test_total , 3 )

def train_batch(model, optimizer, batch, label): 
    optimizer.zero_grad() # 
    input = Variable(batch)
    output = model(input)
    criterion = torch.nn.CrossEntropyLoss()
    criterion(output, Variable(label)).backward() 
    optimizer.step()
    return criterion(output, Variable(label)).data

def train_epoch(model, train_loader, optimizer=None):
    global num_batches
    for batch, label in train_loader:
        loss = train_batch(model, optimizer, batch.cuda(), label.cuda())
        if num_batches%10 == 0:
            print('%23s%-9s%-13s'%('the '+str(num_batches)+'th batch, ','loss is: ',str(round(loss[0],8))))
        num_batches +=1

def train_test(path, arch, model, train_loader, test_loader, optimizer=None, epoches=10):
    global num_batches
    num_batches = 0
    print("Start training.")
    if not osp.exists(path):
        os.mkdir(path)
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr = 0.001, momentum=0.9)

    for i in range(epoches):
        model.train()
        print("Epoch: ", i)
        train_epoch(model, train_loader, optimizer)
        acc = test(model, test_loader)
        print("Test Accuracy :"+str(round( acc , 3 )))
        # if acc>=0.8:
        filename = osp.join(path, arch + str(acc) + '.pth')
        state = model.state_dict()
        torch.save(state, filename)
    print("Finished training.")
