### Copy of PY_ML_GAT_Model.ipynb to import inot pass_analysis_main.py

import pandas as pd
import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math

from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

#from pass_analysis.pass_data_processing.model_preparation import model_preparation_main, pass_graph, loader

#loader = DataLoader(pass_graph, batch_size= 4, shuffle= False) ## get ref # Loading in the data from the build graph
##----------------------------------------------------------

#----------------------------------
#             Config
#----------------------------------

in_channels = 6# OR IS THIS 7??  # check this +++ can we get this to 16, will it be faster??
hidden_channels  = 64   # hiddel and out_channels
Dropout= 0.2 # start at 0.2 then go up if needed
per_head = hidden_channels // 2
num_grade_classes = 5 # blunder, mistake, textbook, good, brilliant
compress_channel = 32
export_data_model_path_new = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/models/model_new.pt'

#------------------------------- 
#           Train
#-------------------------------

def export_data(model):
    torch.save(model.state_dict(), export_data_model_path_new) ### Refrance https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    #grades.to_csv('ML_Results.csv', index=False)

##  Train
def train(data, model, device):
    model.train()
    optimizer.zero_grad()
    grade = model(data.x, data.edge_index, data.batch, data.edge_attr) 
    # https://stackoverflow.com/questions/69965519/cross-entropy-loss-argument-target-position-2-must-be-tensor-not-numpy-n
    print(data.x.shape)
    ## = F.mse_loss(x_uncompressed, data.x) # input, target https://python.pages.doc.ic.ac.uk/lessons/pytorch/08-autoencoder/02-auto-encoder.html, from this found this https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
    grade_loss = F.cross_entropy(grade, data.y_grade.view(-1)) ## ref - https://discuss.pytorch.org/t/valueerror-expected-input-batch-size-1-to-match-target-batch-size-4-how-can-i-fix-this/202084
    loss = grade_loss
    loss.backward()
    optimizer.step()
    return float(loss.detach()) # need to change back to return total_loss.detech() or item()


#------------------------------- 
#            Test
#-------------------------------

grades_data = []
@torch.no_grad()  ## do we need this
def test(loader, model, device):
    device = torch.device('cpu')
    model.eval()
    accs = []   
    for data in loader:
        data = data.to(device)
        grade = model(data.x, data.edge_index, data.batch, data.edge_attr) #https://halil7hatun.medium.com/graph-neural-networks-gnns-1f463df4bb77
        grade_predictions = grade.argmax(dim=1)
        correct_predictions = int((grade_predictions == data.y_grade).sum()) ##
        total = data.y_grade.shape[0]
        accs.append(correct_predictions/total)
        
        # Convert From Numpy Autoencoder to Numpy for KMeans
        #compress = compress.cpu().detach().numpy()# https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
      #  kmeans = KMeans(n_clusters=5, init='k-means++', random_state=15, n_init=1) ## using 5 clusters as 5 different classication could use the elbow algo to decide my clusters as well check # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html, 5 cluster becuase we have 5 different catergorise
       # grades = kmeans.fit_predict(compress) + 1 ## check varible name
       # print(f'grades: {grade_predictions}')
        # Do I need to swithc back from Numpy
    return accs


#------------------------------- 
#          Training Loop
#-------------------------------
 
def training(loader, model, device, epoch=100):
    times = []
    for epoch in range(1, epoch + 1): ##change this back to 200 ## if adding para backing change to 1, args.epochs + 1 - args start at 0, deault = 200
        start = time.time()
        for data in loader:
            data = data.to(device)
            loss = train(data, model, device)
            print(f'loss: {loss}')
        accs = test(loader, model, device)
        print(f'grade : {accs}')
        times.append(time.time() - start)
    return times, accs, loss


def display_ouput(times, accs, epoch, loss, model):
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    print(f'Grade Accuracy: {accs} Epoch: {epoch}, Loss: {loss}')
    print(f'\n===================================================================\n \t\tExporting \n===================================================================\n') 
  #  print(f'\tPassing Grade Results:\n\n{grades}\n\n')
    print(f'\tModel:\n\n{model}\n \n===================================================================\n') ## does it need to be a pickle can't it just be a txt ??? 
    export_data(model) ### why does it need to be a pickle can't it just be a txt?

def main():
    model.to(device)  ## sending the model to th GPU or CPU
    times, accs, loss = training(loader, model, device, epoch=100) 
    display_ouput(times, accs, 100, loss, model)

#main()


#----------------------------------
#             Model
#----------------------------------

# Refrances:
# 1 https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py 
# supporting # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#exercises - check

class PassingGAT(nn.Module):
    def __init__(self):  ## define here self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        # follow this format - in_channels, hidden_channels, heads, dropout=0.6
        self.conv1 = GATv2Conv(in_channels, per_head , heads=2, edge_dim=3, concat=True, dropout=Dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)

        self.conv2 = GATv2Conv(hidden_channels, per_head, heads=2, edge_dim=3, concat=True, dropout=Dropout)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, heads=1, edge_dim=3, concat=False, dropout=Dropout)

        # Residual projection
        self.res_proj = nn.Linear(in_channels, hidden_channels)
       # self.encoder = nn.Linear(hidden_channels, compress_channel) # https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/ - in out or out in ?? - https://python.pages.doc.ic.ac.uk/lessons/pytorch/08-autoencoder/04-refactoring-ae.html - 64 64
       # self.decoder = nn.Linear(compress_channel , in_channels) # https://python.pages.doc.ic.ac.uk/lessons/pytorch/08-autoencoder/02-auto-encoder.html 64 11?? not really getting compressed does make sense maybe half??

        # Heads - no longer needed ????
        self.grade_head   = nn.Linear(hidden_channels, num_grade_classes)
 
    def forward(self, x, edge_index, batch, edge_attr):

        # Residual connection
        res = self.res_proj(x)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=Dropout, training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm2(x)
        x = F.dropout(x, p=Dropout, training=self.training)
        # Layer 3
        x = self.conv3(x, edge_index, edge_attr)

        # Add residual impor
        x = x + res ## need to change this as this could break if dimention of modle change
        #compress = F.relu((self.encoder(x)))
        #x_uncompressed = self.decoder(compress) ### check this 



        #Reconsruction of the data
        x = global_mean_pool(x, batch) # from 40, 64 to  1, 64 one grade per pass not graph
       # predict
        grade = self.grade_head(x)

        return grade # outcome  # could use cross entropy loss but use this and nll_
device = torch.device('cpu')   
model = PassingGAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

