import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

#from pass_data_processing.data_processing import data_processing_main
from pass_analysis.pass_data_processing.data_processing import data_processing_main
from .pass_grade_assignment.data_to_video import data_to_video_main
from training.ML_GAT.PY_ML_GAT_Model import PassingGAT

#from FootballPassingAnalysisProject2.pass_analysis.pass_grade_assignment import data_to_graph_main

class PassAnalysis:


    def csv_import():
        dataframe = pd.read_csv('output/data.csv')
    # print(data.head())
        return dataframe
    
    def grade_to_passes_csv(passes_dataframe, predictions):
        pred_length = len(predictions)
        for i in range(pred_length):
            passes_dataframe.iloc[i, passes_dataframe.columns.get_loc('pass_grade')] = predictions[i].item() # Refrance - https://stackoverflow.com/questions/53806570/why-does-one-use-of-iloc-give-a-settingwithcopywarning-but-the-other-doesnt
        passes_dataframe.to_csv('output/passes.csv') ## Refrance - .item() in line 25 - https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
        return passes_dataframe

    def passing_analysis():

        data_path = 'output/data.csv'
        passes_path = 'output/passes.csv'
        model_path = 'models/model.pt'

        device = torch.device('cpu')
        dataframe = PassAnalysis.csv_import()

        pass_graph, passes_dataframe = data_processing_main(dataframe)

        loader = DataLoader(pass_graph, batch_size= 4, shuffle= False)
        model = PassingGAT().to(device) ## check if needed
        model.load_state_dict(torch.load('models/model.pt', weights_only=True)) ## check if needed
        model.eval()

        grade_predictions = []
        grades_data = []
        with torch.no_grad(): #device = torch.device('cpu')
            accs = []   
            predictions = []
            grade_output = []
            for data in loader:
                data = data.to(device)
                grade = model(data.x, data.edge_index, data.batch, data.edge_attr) #https://halil7hatun.medium.com/graph-neural-networks-gnns-1f463df4bb77
                grade_predictions = grade.argmax(dim=1)
                #correct_predictions = int((grade_predictions == data.y_grade).sum()) ##
                #accs.append(correct_predictions/total)
                predictions.extend(grade_predictions.cpu().numpy())
            print(passes_dataframe)
            print(predictions)
            #print(accs)
            #print(correct_predictions)
            passes_dataframe =PassAnalysis.grade_to_passes_csv(passes_dataframe, predictions)
        return grade_predictions, passes_dataframe, grade_output



    def pass_results(): ### might just call these in index . py
        video_file  =  "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/output.mp4"
        output_video = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/graded_output.mp4"
        hold_frame  = 45  # how long each badge stays on screen # this is
        passes_csv = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv"
        pass_grade = pd.read_csv(passes_csv)
        #data_to_graph_main()
        data_to_video_main(passes_csv)

    



