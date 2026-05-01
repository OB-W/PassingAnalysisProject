import pandas as pd
from utils import get_center_of_bbox

def export_data(tracks, file='output/data.csv'): # saving to csv file, track has player and ball tracks
    rows = []

    for frame_num, players in enumerate(tracks["players"]): # goes through each frame and players in tracks["players"]
        ball = tracks["ball"][frame_num] #

        if 1 in ball: #
            ball_x, ball_y = get_center_of_bbox(ball[1]["bbox"])
        else: # 
            ball_x, ball_y = False, False

        for player_id, player in players.items(): # goes through each player in the frame
            x, y = player["position"] #
            team = player.get("team", None) # won't crash if not their
            has_ball = player.get("has_ball", False) #

            rows.append({   ## addind row for each player in frame with the, player id, team, player position, ball position, has_ball
                "frame":frame_num,
                "player_id": player_id,
                "team": team,
                "player_x":x,
                "player_y": y,
                "ball_x": ball_x,
                "ball_y":ball_y,
                "has_ball":has_ball
            })

    dataframe = pd.DataFrame(rows) 
    dataframe = pass_validation(dataframe)
    dataframe.to_csv(file, index=False) ## saving data to csv
   # print("saved file") #check this only temp

def pass_validation(dataframe): ## maybe refractor could be gold plating ref
    frame_count = {}
    hold_count = {}
    for frame, rows in dataframe.groupby('frame'):
        holder = None
        for _, row in rows.iterrows():
            if row['has_ball'] == True:
                holder = row['player_id']
    
        if holder is None: 
            dataframe.loc[dataframe['frame'] == frame, 'has_ball'] = False
            continue
        
        hold_count = {holder: hold_count.get(holder, 0) + 1}
    
        if holder in hold_count:
            hold_count[holder] += 1
        else:
            hold_count[holder] = 1 
    
        if hold_count[holder] >= 25:
            dataframe.loc[(dataframe['frame'] == frame) & (dataframe['player_id'] == holder), 'has_ball'] = True
            dataframe.loc[(dataframe['frame'] == frame) & (dataframe['player_id'] != holder), 'has_ball'] = False
        else:
            dataframe.loc[dataframe['frame'] == frame, 'has_ball'] = False  
    dataframe['has_ball'] = dataframe['has_ball'].astype(bool)    
    return dataframe




                
