
#from .data_creation import data_creation_main working

from pass_analysis.pass_data_processing.data_creation import data_creation_main
from .model_preparation import model_preparation_main

def data_processing_main(dataframe):
    passes_dataframe = data_creation_main(dataframe)
    pass_graph = model_preparation_main()
    return pass_graph, passes_dataframe 




