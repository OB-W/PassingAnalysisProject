import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from .pass_data_processing.data_processing import data_processing_main
from training.ML_GAT.PY_ML_GAT_Model import PassingGAT

from .pass_grade_assignment import data_to_video_main
#from FootballPassingAnalysisProject2.pass_analysis.pass_grade_assignment import data_to_graph_main

class PassAnalysis:

    data_path = 'output/data.csv'
    passes_path = 'output/passes.csv'
    model_path = 'models/model.pt'

    def csv_import(data_path):
        dataframe = pd.read_csv(data_path)
    # print(data.head())
        return dataframe

    def passing_analysis():
        device = torch.device('cpu')
        dataframe = csv_import(data_path)

        pass_graph, passes_dataframe = data_processing_main(dataframe)

        loader = DataLoader(pass_graph, batch_size= 4, shuffle= False)
        model = PassingGAT(nn.Module).to(device) ## check if needed
        model.load_state_dict(torch.load(PassAnalysis, weights_only=True)) ## check if needed
        model.eval()


        grades_data = []
        with torch.no_grad(): #device = torch.device('cpu')
            accs = []   
            for data in loader:
                data = data.to(device)
                grade = model(data.x, data.edge_index, data.batch, data.edge_attr) #https://halil7hatun.medium.com/graph-neural-networks-gnns-1f463df4bb77
                grade_predictions = grade.argmax(dim=1)
                correct_predictions = int((grade_predictions == data.y_grade).sum()) ##
                total = data.y_grade.shape[0]
                accs.append(correct_predictions/total)



        passes_dataframe['pass_grade'] = grade_predictions 
        passes_dataframe.to_csv(passes_path, index=False) ## 
        print(passes_dataframe['pass_grade'])

    def pass_results(): ### might just call these in index . py
        #data_to_graph_main()
        data_to_video_main(grades, video_file, output_video, passes_csv)

    



