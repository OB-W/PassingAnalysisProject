

def distance_to_player(dataframe):
    distances = []

    for i in range(len(dataframe)):
        curr = dataframe.iloc[i]
        prev_dist = None
        next_dist = None
    
        if i > 0:
            prev = dataframe.iloc[i - 1]
            if prev['frame'] == curr['frame']:
                prev_dist = math.dist((curr['player_x'], curr['player_y']), (prev['player_x'], prev['player_y']))
        if i < len(dataframe) - 1:
            nextLine = dataframe.iloc[i + 1]
            if nextLine['frame'] == curr['frame']:
                next_dist = math.dist((curr['player_x'], curr['player_y']), (nextLine['player_x'], nextLine['player_y']))
        if prev_dist is None and next_dist is None:
            distances.append(9999)
        elif prev_dist is None:
            distances.append(next_dist)
        elif next_dist is None:
            distances.append(prev_dist)
        else:         
            min_distance = min(prev_dist, next_dist)
            distances.append(min_distance)   
    print(dataframe)
    dataframe['distance_to_player'] = distances
    return dataframe


    def player_velocity(dataframe): # https://www.geeksforgeeks.org/python/python-program-to-calculate-acceleration-final-velocity-initial-velocity-and-time/
    dataframe["player_vx"]=dataframe["player_x"].diff() * fps # calculate the difference * time calculate the difference
    dataframe["player_vy"]=dataframe["player_y"].diff() * fps
    #print(dataframe)
    #print("player_velocity finished")
    return dataframe

def ball_velocity(dataframe):
  #  dataframe = dataframe.sort_values("frame")
    dataframe["ball_vx"] = dataframe["ball_x"].diff() * fps
    dataframe["ball_vy"] = dataframe["ball_y"].diff() * fps
    dataframe["ball_speed"] = (dataframe["ball_vx"]**2 + dataframe["ball_vy"]**2)**0.5 # Refrance - https://www.geeksforgeeks.org/maths/euclidean-distance/
    #print('ball_velocity finished')
    #print(dataframe) ## showing the 
    return dataframe