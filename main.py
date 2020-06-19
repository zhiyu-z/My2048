import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import pandas as pd


#Hyper parameter define
Epochs=10
Batch_Size=2048
LR=0.001


class MyDataread(data.Dataset):
    def __init__(self,root,transform=None):
        super().__init__()
        Data=pd.read_csv(root).values
        self.board=Data[:,0:16]
        self.label=Data[:,16]
        self.transform=transform
        self.index=0
        
    def __getitem__(self,index):
        board=self.board[index].reshape((4,4))
        board=board[:,:,np.newaxis]
        board=board/11.0
        label=self.label[index]
        if self.transform is not None:
            board=self.transform(board)
            board=board.type(torch.float)
        return board,label
        
    def __len__(self):
        return len(self.label)
    
    
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.LSTM(input_size=4,hidden_size=400,num_layers=4,batch_first=True)
        self.out=nn.Linear(400,4)
    
    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None)
        out=self.out(r_out[:,-1,:])
        return out
    
    
def main():
    TrainData=MyDataread(root="/mnt/My2048/traindata.csv",transform=transforms.ToTensor())
    TrainLoader=data.DataLoader(dataset=TrainData,batch_size=Batch_Size,shuffle=True,num_workers=0)
    
    rnn=RNN()
    optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()
        
    for epoch in range(Epochs):
        for step,(board,label) in enumerate(TrainLoader):
            board=Variable(board.view(-1,4,4))
            label=Variable(label)
                
            if torch.cuda.is_available():
                board=board.cuda()
                label=label.cuda()
                rnn.cuda()
                    
            out=rnn(board)
            loss=loss_func(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            if step % 50 == 0:
                out=rnn(board)
                predict=torch.max(out,1)[1]
                train_correct=(predict==label).sum().item()
                print('Epoch: ', epoch, '| test accuracy: %.4f' % (train_correct/(Batch_Size * 1.0)))
    torch.save(rnn,'RNN_model.pkl')
