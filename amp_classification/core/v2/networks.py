"""New MIC128 adaptations, trained from scratch; canonical length 6–50."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
from common import DATA

class SequenceModel(nn.Module):
    def __init__(self,name,config):
        super().__init__();self.name=name;self.config=config
        self.embedding=nn.Embedding.from_pretrained(torch.cat([torch.zeros(1,20),torch.eye(20)]),freeze=True,padding_idx=0)
        dropout=config.get('dropout',.3)
        if name=='TinyCNN':
            c=config.get('channels',48)
            self.convs=nn.ModuleList([nn.Conv1d(20,c,k,padding=k//2) for k in [3,5,7]])
            self.head=nn.Sequential(nn.Linear(6*c,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,1))
        elif name=='Das':
            h=config.get('hidden',100);self.rnn=nn.LSTM(20,h,batch_first=True,bidirectional=True)
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(2*h,1))
        elif name=='DeepAMP':
            self.conv1=nn.Conv1d(20,64,5);self.conv2=nn.Conv1d(64,64,5)
            self.dropout=nn.Dropout(dropout)
            self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(64*12,100),nn.ReLU(),nn.Linear(100,100),nn.ReLU(),nn.Linear(100,20),nn.ReLU(),nn.Linear(20,1))
        elif name=='APEX':
            embedding=np.load(DATA/'aaindex.npy');self.embedding=nn.Embedding.from_pretrained(torch.from_numpy(embedding),freeze=True,padding_idx=0)
            h=config.get('hidden',128);self.rnn=nn.GRU(embedding.shape[1],h,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
            self.norm=nn.LayerNorm(2*h);self.attn=nn.Linear(2*h,1)
            # Single attention/head adaptation. Do not inherit the original unassigned
            # permute operation or let padding contribute to normalization.
            self.head=nn.Sequential(nn.Linear(2*h,128),nn.ReLU(),nn.Linear(128,2048),nn.LayerNorm(2048),nn.ReLU(),nn.Dropout(.1),
                                    nn.Linear(2048,1024),nn.LayerNorm(1024),nn.ReLU(),nn.Dropout(.1),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,1))
        else:raise ValueError(name)

    def forward(self,tokens):
        mask=tokens.ne(0);length=mask.sum(1);x=self.embedding(tokens)
        if self.name=='TinyCNN':
            values=[]
            for conv in self.convs:
                h=F.relu(conv(x.transpose(1,2)))
                values += [h.masked_fill(~mask[:,None,:],-1e6).amax(2),(h*mask[:,None,:]).sum(2)/length[:,None]]
            z=torch.cat(values,1)
        elif self.name in ['Das','APEX']:
            pack=pack_padded_sequence(x,length.cpu(),batch_first=True,enforce_sorted=False)
            h,_=self.rnn(pack);h,_=pad_packed_sequence(h,batch_first=True,total_length=tokens.shape[1])
            if self.name=='Das':z=h.masked_fill(~mask[:,:,None],-1e6).amax(1)
            else:
                h=self.norm(h);attn=self.attn(h).squeeze(-1).masked_fill(~mask,-1e6).softmax(1);z=(h*attn[:,:,None]).sum(1)
        else:
            # Fixed 50-residue canvas, flank padding 5 as in source; explicit zero
            # padding and propagated centre masks at each pooling stage.
            h=F.pad(x.transpose(1,2),(5,5));m=F.pad(mask[:,None,:].float(),(5,5))
            h=F.relu(self.conv1(h));m=m[:,:,2:-2];h=h*m
            h=F.max_pool1d(h,2);m=F.max_pool1d(m,2);h=self.dropout(h)*m
            h=F.relu(self.conv2(h));m=m[:,:,2:-2];h=h*m
            h=F.max_pool1d(h,2);m=F.max_pool1d(m,2);z=(h*m).flatten(1)
        return self.head(z).squeeze(-1)
