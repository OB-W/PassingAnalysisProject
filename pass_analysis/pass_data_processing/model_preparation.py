# Refrance 1 Framework -  (GitHub, n.d.) - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py

#https://www.youtube.com/watch?v=Gv0RT5N_mhg

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

device = torch.device('cpu')

## add tO export columns

path_1 = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/data.csv' # Cheslea Vs Arsenal 10 Minutes
#path_2 = '' # Demo1 - 30 seconds
passes_csv_path = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv' 

def csv_import(path):
    dataframe = pd.read_csv(path)
    #print(dataframe.head())
    return dataframe

#------------------------------- 
#          Load Data In
#-------------------------------

passes_dataframe = pd.read_csv(passes_csv_path) # read data in
data_dataframe = pd.read_csv(path_1) 
y_grade = passes_dataframe[['pass_grade']]
pass_graph = []

#------------------------------- 
#          Build Graph 
#-------------------------------
# Graph Level Prediction - 1 graph per pass

## Helped build the graph - https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial?dc_referrer=https%3A%2F%2Fwww.google.com%2F

def graph_construction():
    for index, rows in passes_dataframe.iterrows(): # creating the graph by going through each frame
        start_frame = rows['start_frame']
        end_frame = rows['end_frame']
        row = []
        column = []
    # https://www.geeksforgeeks.org/python/how-to-fix-valueerror-the-truth-value-of-a-series-is-ambiguous-in-pandas/
        
        frame = data_dataframe[(data_dataframe['frame'] >= start_frame) & (data_dataframe['frame'] <= end_frame)]  ## validating if start and end frame are not before or after each other
        if frame.empty:
            print('Warning - No frame data')
            continue

   
        ##frame =
        node_features = frame[['player_x', 'player_y', 'team', 'has_ball', 'ball_x', 'ball_y']]  ## Node Deature for that create the node
    #edge_features = data_dataframe[['player_x', 'player_y']] ## add vy and vx as well 
        edge_features = frame[['distance_to_ball', 'has_ball', 'team']]
        team = frame['team'].values
        
        frame_length = len(frame)
        for i in range(frame_length):   ### this is gonna be spenny,  sliding window # get refrance
            for j in range(frame_length):
                if i != j: ## connects all player - for just teammates- if i != j and team[i] == team[j]:  ## this is only ball holder to temmates, need teammate, might need to change if I want cross team interaction can test
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

    
        # size in each edge features that you pass into the model, -varible you pass into edge_attr 
    #need to break this down edge_features is just an array of feature ve for each edge, and column represents the indices of the edges, then your edge_attr is correct
    #https://www.kaggle.com/code/rafsunsheikh/convert-any-tabular-data-to-graph-for-gnn
    
    ### Creating One Object Per Pass to Calculate One Pass per Graph - https://pytorch-geometric.readthedocs.io/en/2.5.2/get_started/introduction.html#data-transforms

