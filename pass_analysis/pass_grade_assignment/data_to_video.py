#Quick Notes:

#Format ML Output into Passes.csv

#Use passes.csv
#Video read / find the right frame

#Add icon to each right frame
#put back into output.csv
import cv2
import pandas as pd


passes_csv = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv"

# Pass BGR Colour Refrance - https://stackoverflow.com/questions/50495608/how-to-define-bgr-color-range-map-color-code-to-color-name 
grades = { 0: ("Blunder ??",(0, 0, 210)),
    1: ("Mistake ?",(0, 100,255)),
    2: ("Textbook", (0,200, 255)),
    3: ("Good !",(50, 205, 50)),
    4: ("Great !!",(255, 215, 0)),
} 

def csv_import():  ## reading the csv file and creating passes_df
    passes_df = pd.read_csv(passes_csv).set_index("frame_id")
    print(passes_df.head())
    return passes_df


def draw_grade(frame, pass_grade):  # https://www.geeksforgeeks.org/python/python-opencv-write-text-on-video/
    label, color = grades.get(pass_grade)  # get lable and colour from dictionary
    font = cv2.FONT_HERSHEY_SIMPLEX
    (frame_width, frame_height), _ = cv2.getTextSize(label, font, 0.55, 2)
    ## Refrance 1 - code was inspired by annotiation_utils - 77-82 and 93
    overlay = frame.copy()
    cv2.rectangle(overlay, pt1=(100, 850), pt2=(850, 970), color=(255, 255, 255), thickness=cv2.FILLED)
    alpha = 0.4
    cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1-alpha, gamma=0, dst=frame)
    
    cv2.putText(frame, text=f"Pass Grade: {label}", org=(150, 950), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)


def data_to_video_main(passes_csv):
	## ref https://www.geeksforgeeks.org/python/saving-a-video-using-opencv/
    #passes_df = csv_import()
    passes_df = pd.read_csv(passes_csv)
    ##set_index("frame_id")

    video_file  =  "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/output.mp4"
    output_video = "/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/graded_output.mp4"
    
	
    cap  = cv2.VideoCapture(video_file) # Loading video
    fps  = cap.get(cv2.CAP_PROP_FPS) # don't know if needed
	
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
      
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height)) 
    #^^ Refrance - VideoWriter - https://www.geeksforgeeks.org/python/saving-a-video-using-opencv/

    active = {} 
    frame_num = 0 
# Refrance for this whole late part of code - https://www.geeksforgeeks.org/python/python-play-a-video-using-opencv/
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

## Extra - if time at the end get colour for the text to work 


	
	
