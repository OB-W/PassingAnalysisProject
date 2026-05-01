#Plan:
#Format ML Output into Passes.csv

#Use passes.csv
#Video read / find the right frame

#Add icon to each right frame

#put back into output
# Jupyter 
# Then Vscode then github
import cv2
import pandas as pd


video_file  =  "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/output.mp4"
output_video = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/graded_output.mp4"
hold_frame  = 45  # how long each badge stays on screen # this is
passes_csv = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/pass_data_processing/passes.csv"


grades = {
    0.1: ("Blunder ??", (0,   0, 210)),
    0.3: ("Mistake ?", (0, 100, 255)),
    0.5: ("Inaccuracy ?!", (0, 200, 255)),
    0.7: ("Good !", (50, 205,  50)),
    0.9: ("Best !!", (255, 215,  0)),
}

def csv_import():
    passes_df = pd.read_csv(passes_csv).set_index("frame_id")
    print(passes_df.head())
    return passes_df


def draw_grade(frame, pass_grade):  # https://www.geeksforgeeks.org/python/python-opencv-write-text-on-video/
    label, color = grades.get(pass_grade)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (frame_width, frame_height), _ = cv2.getTextSize(label, font, 0.55, 2)
    cv2.rectangle(frame, (20, frame_height-60), (20 + frame_width + 10, frame_height - 20), (0, 0, 0), 3)  ### https://www.geeksforgeeks.org/python/python-opencv-cv2-rectangle-method/
    cv2.putText(frame, label, (25, h - 30), font, 0.55, (255,255,255), 2, cv2.LINE_AA)
def main(grades, video_file, output_video, passes_csv):
	## ref https://www.geeksforgeeks.org/python/saving-a-video-using-opencv/
    #passes_df = csv_import()
    passes_df = pd.read_csv(passes_csv)
    ##set_index("frame_id")
    
	
    cap  = cv2.VideoCapture(video_file) # Loading video
    fps  = cap.get(cv2.CAP_PROP_FPS) # don't know if needed
	
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
      
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height)) # https://www.geeksforgeeks.org/python/saving-a-video-using-opencv/

    active = {}   # { key: (row, expiry_frame) }
    frame_num = 0 # Ref https://www.geeksforgeeks.org/python/python-play-a-video-using-opencv/

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for index, row in passes_df[(passes_df["start_frame"] <= frame_num) & (passes_df["end_frame"] >= frame_num)].iterrows():
            draw_grade(frame, row['pass_grade'])

        out.write(frame)
        frame_num += 1

    cap.release()
    out.release()
    print("Done", output_video)

main(grades, video_file, output_video, passes_csv)
	
	
