import pandas as pd
import os
import numpy
import math

data_path = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/data.csv' # csv
# path = 'output/data.csv'

#https://pandas.pydata.org/docs/index.html


passes_path = '/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv'


def csv_import():
        dataframe = pd.read_csv(data_path)
    # print(data.head())
        return dataframe


def detect_passes(dataframe):
    passes = []
    last_holder = None
   # pass_grade = None 
    #current_holder = None
  #  pass_grade = None
    start_frame = None ## will not work if we actually start from a certain frame
    i = 0
    # for frame in dataframe["frame"].unique(): ## looing through the frames and 
    for frame, rows in dataframe.groupby('frame'): ## looing through the frames and 
       # rows = dataframe[dataframe['frame'] == frame]
        holders = rows[rows["has_ball"] == 1] 
        current_holder = holders['player_id'].values[0] if len(holders) > 0 else None
        if len(holders) == 0:
            continue
        current_holders = holders['player_id'].values[0]
        
        if last_holder is None:
            last_holder = current_holder
            start_frame = frame
            continue

        if current_holder != last_holder:  ### will be equal if there is no 
            
            pass_grade = holders['pass_grade'].iloc[0] if 'pass_grade' in holders.columns else None
            passes.append({'passes_id': i, 'player_id' : last_holder, 'reciever_id': current_holder, 
            'start_frame':start_frame, 'end_frame':  frame, 'pass_grade': pass_grade}) ## will be na if there are no holder then frame will just always fill 
            i +=1
            last_holder = current_holder
            start_frame= frame
        else:
            if start_frame is None:
                start_frame = frame
            # this need to be cleaned - below or recfaotro and the main as well
    passes_dataframe = pd.DataFrame(passes) ## ?
   # passes_dataframe = import_ML_Data(data, passes_dataframe)
    save_passes(passes_dataframe)
    return passes_dataframe

def save_passes(passes_dataframe):
    passes_dataframe.to_csv(passes_path, index=False)


def distance_to_ball(dataframe):
    dataframe['distance_to_ball'] = numpy.sqrt((dataframe['player_x'] - dataframe['ball_x'])**2 + (dataframe['player_y'] - dataframe['ball_y'])**2)
    return dataframe

    
def data_creation_main(dataframe):
    passes = detect_passes(dataframe)
    dataframe = distance_to_ball(dataframe)
  # print(dataframe.shape)
    #output_cv(dataframe)
    dataframe.to_csv('/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/data.csv', index=False) ## all other data so just data.csv or dataMain.c

def main():
    dataframe = csv_import()
    passes = detect_passes(dataframe)


main()

