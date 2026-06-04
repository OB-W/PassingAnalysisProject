# Template from ML_GAT_Model.ipynb - Take out y_grade as not needed
# For comment please see ML_GAT_Model.ipynb

import pandas as pd
import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.file_path_config.file_paths import data_csv, passes_csv

device = torch.device('cpu')



def csv_import(path):
    dataframe = pd.read_csv(path)
    #print(dataframe.head())
    return dataframe

#------------------------------- 
#          Load Data In
#-------------------------------

passes_dataframe = pd.read_csv(passes_csv) # read data in
data_dataframe = pd.read_csv(data_csv) 
y_grade = passes_dataframe[['pass_grade']]
pass_graph = []

#------------------------------- 
#          Build Graph 
#-------------------------------
# Graph Level Prediction - 1 graph per pass

## Helped build the graph - https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial?dc_referrer=https%3A%2F%2Fwww.google.com%2F

def graph_construction():  ## One graph per pass 
    for index, rows in passes_dataframe.iterrows(): # creating the graph by going through each frame
        start_frame = rows['start_frame']
        end_frame = rows['end_frame']
        row = []
        column = []
        
        frame = data_dataframe[(data_dataframe['frame'] >= start_frame) & (data_dataframe['frame'] <= end_frame)]  ## validating if start and end frame are not before or after each other
        if frame.empty:
            print('Warning - No frame data')
            continue


        node_features = frame[['player_x', 'player_y', 'team', 'has_ball', 'ball_x', 'ball_y']] 
        edge_features = frame[['distance_to_ball', 'has_ball', 'team']]
        team = frame['team'].values
        
        frame_length = len(frame)
        for i in range(frame_length):   ### this is gonna be spenny,  sliding window # get refrance
            for j in range(frame_length):
                if i != j: ## connects all player - for just teammates- if i != j and team[i] == team[j]:
                    row.append(i)
                    column.append(j) 


        #   Data Formatting for Graph Attention Network 
        node_features = node_features.values.astype('float32') ## need to convert float32 to pytorch
        edge_features = edge_features.values[column].astype('float32') 
        edge_index = torch.tensor([row, column], dtype=torch.long)
        x = torch.tensor(node_features)
        edge_attr = torch.tensor(edge_features)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr) #edge_attr=edge_attr
        pass_graph.append(data)
    return pass_graph


def model_preparation_main():
    pass_graph = graph_construction()
    return pass_graph
