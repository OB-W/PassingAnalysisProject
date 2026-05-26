from pass_analysis.pass_data_processing.data_creation import data_creation_main
from .model_preparation import model_preparation_main


def data_processing_main(dataframe): 
    passes_dataframe = data_creation_main(dataframe) # from exports to data creation and creat passes.csv, and other features
    pass_graph = model_preparation_main() ## graph consruction similar to ML__GAT_Model
    return pass_graph, passes_dataframe 




