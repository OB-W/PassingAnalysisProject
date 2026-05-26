### Copy of PY_ML_GAT_Model.ipynb to import pass_analysis_main.py - for comment please see ML_GAT_Model.ipynb

import pandas as pd
import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


#----------------------------------
#             Config
#----------------------------------

in_channels = 6
hidden_channels  = 64   # hiddel and out_channels
Dropout= 0.2 # start at 0.2 then go up if needed
per_head = hidden_channels // 2
num_grade_classes = 5 # blunder, mistake, textbook, good, brilliant
compress_channel = 32
export_data_model_path_new = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/models/model_new.pt'


def export_data(model):
    torch.save(model.state_dict(), export_data_model_path_new) ### Refrance https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    #grades.to_csv('ML_Results.csv', index=False)


#----------------------------------
#             Model
#----------------------------------

# Refrances Model - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py 

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

