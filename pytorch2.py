import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
import gc
import  torch.nn.init as weight_init
import numpy as np
from pytorch1 import Music_data
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_auc_score
torch.manual_seed(512) #为CPU设置种子用于生成随机数，以使得结果是确定的
#========================================import===============

batch_size = 10000
sequence_lenth = 1
input_size = 21
hidden_size = 128
output_size = 1
num_layers = 5
learning_rate = 0.015
EPOCH = 15

# ==================================================size info=========
start = datetime.datetime.now()
train_loader = DataLoader(dataset=Music_data(train=True),
                                           batch_size=batch_size,
                                           num_workers=8,
                                           drop_last=True)
test_loader = DataLoader(dataset=Music_data(train=False),
                                           batch_size=batch_size,
                                           num_workers=8,
                                           drop_last=False)

print('data load time', datetime.datetime.now() - start)

class LSTMrnn(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, output_size):
        super(LSTMrnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.5,
                            batch_first=True)

        self.fc = nn.Linear(input_size, 128)
        self.fc_1= nn.Linear(128,256)
        self.fc_2 = nn.Linear(256,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.output_size)
        self.bn1 = nn.BatchNorm1d(input_size, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        self.fc.weight.data.uniform_(0., 0.01)
        self.fc2.weight.data.uniform_(-0.01, 0.01)
        self.fc3.weight.data.uniform_(0.0, 0.01)
        for name, weight in self.lstm.named_parameters():
            if len(weight.size()) == 1:
                weight_init.uniform(weight, 0, 0.01)
            else:
                weight_init.kaiming_uniform(weight)


    def forward(self, x):

        h0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0),
                                  self.hidden_size)).cuda()
        x = x.view(-1, input_size)
        x = self.bn1(x)
        batch_size = x.size(0)
        x = x.view(-1, input_size)

        out = self.fc(x)
        out=self.fc_2(self.fc_1(x**2))
        # out, _ = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])
        out = self.bn2(out)
        out = self.fc2(out)
        out = self.sig(self.fc3(out))
        return out
rnn = LSTMrnn(input_size, hidden_size, num_layers, output_size)
print(rnn)
rnn.cuda()
criterion = torch.nn.BCELoss()
criterion.cuda()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9)

def train(net, loader, criterion, optimizer):
    net.train()
    list_l = []
    for i, (x, y) in enumerate(loader):
        y = torch.FloatTensor(y.numpy()*1.0)
        x = torch.FloatTensor(x.numpy()*1.0)
        x = Variable(x.view(-1, sequence_lenth, input_size)).cuda()
        y = Variable(y).cuda()  #imgages=== torch.Size([100, 1, 19])
        outputs = net(x)
        outputs = outputs.view(-1)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==1:
            print(loss.data[0])
        list_l.append(loss.data[0])
    loss_epoch = np.average(list_l)
    return loss_epoch
def validate(net, loader, criterion):
    print('start validate.....')
    list_l = []
    list_y = []
    for i, (x, y) in enumerate(loader):
        list_y.extend(y.numpy())
        x = torch.FloatTensor(x.numpy()*1.0)
        x = Variable(x.view(-1, sequence_lenth, input_size)).cuda()
        outputs = net(x).cpu().data.numpy()
        list_l.extend(outputs)
    auc = roc_auc_score(list_y, list_l)
    return auc

for epoch in range(EPOCH):
        start_time = datetime.datetime.now()
        train_loss = train(rnn, train_loader, criterion, optimizer)
        roc = validate(rnn, test_loader, criterion)
        print(' train_loss: ',  train_loss)
        print(epoch, ' roc_score: ',  roc)
        # torch.save(rnn.state_dict(), str(epoch) + 'cxd1.pkl')
        print(datetime.datetime.now() - start_time)





