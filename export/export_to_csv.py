# need to do a data dum and interpret all the data
import pandas as pd
from utils import get_center_of_bbox

def export_data(tracks, file='output/data2.csv'): # saving to csv file, track has player and ball tracks
    rows = []

    for frame_num, players in enumerate(tracks["players"]): # goes through each frame and tracks
        ball = tracks["ball"][frame_num]

        if 1 in ball:
            ball_x, ball_y = get_center_of_bbox(ball[1]["bbox"])
        else:
            ball_x, ball_y = False, False

        for player_id, player in players.items(): # goes through each player in the frame
            x, y = player["position"]
            team = player.get("team", None) # won't crash if not their
            has_ball = player.get("has_ball", False)

            rows.append({
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
    dataframe.to_csv(file, index=False)
    print("saved file") #check this only temo


def dump(tracks):
    for obj_type, frames in tracks.items():
        print(f"\n{obj_type}")
        for track_id, info in frames[0].items():
            print(f"  {track_id}: {info}")

                
