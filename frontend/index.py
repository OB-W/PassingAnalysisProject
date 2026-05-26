import streamlit as st
import time
import os
import sys
sys.path.append(os.path.abspath("."))
from main import process_video

import pandas as pd
from numpy.random import default_rng as rng

st.set_page_config(page_title="Football Analysis")

# menu
st.sidebar.title("Middlegame Analytics") 

st.sidebar.title("Settings")

st.sidebar.header("Options")
players = st.sidebar.toggle("Highlight Players", value=True)
goalkeepers = st.sidebar.toggle("Highlight Goalkeepers", value=True)
referees = st.sidebar.toggle("Highlight Referees", value=True)
ball = st.sidebar.toggle("Highlight Ball", value=True)
stats = st.sidebar.toggle("Show Statistics", value=True)
keypoints = st.sidebar.toggle("Show Keypoints", value=True, disabled=True, help="Feature coming soon")
speed = st.sidebar.toggle("Show Players' Speed", value=True, disabled=True, help="Feature coming soon")

# select classes to track

# data.yaml class IDs
# ball: 0, goalkeeper: 1, player: 2, referee: 3
options = {0: ball, 1: goalkeepers, 2: players, 3: referees, 4: stats}
classes = [key for key, value in options.items() if value is True]

st.sidebar.markdown("***")

st.sidebar.header("Video Source")

st.sidebar.subheader("Demo")

st.sidebar.write("Choose from 2 demo videos.")

uploaded_video = None
demo_video = None
start_analysis = None
processed = False

demo = st.sidebar.toggle("Demo", value=False)

if demo:
    videos = [
    "demos/demo1.mp4",
    "demos/demo2.mp4"
    ]

    demo_video = st.sidebar.radio("Select Video", videos)

    #preview demo video
    st.sidebar.video(demo_video)
    
    if demo_video:
        with open(demo_video, "rb") as f:
            demo_video_bytes = f.read()

    start_analysis = st.sidebar.button("Start Analysis", key="demo")


    if start_analysis:
        with st.spinner("Processing ..."):
            process_video(demo_video_bytes, classes)
            processed = True
        placeholder = st.empty()
        with placeholder.container():
            st.success("Video processing complete.")
            time.sleep(3)
        placeholder.empty()

st.sidebar.write("\n")

st.sidebar.subheader("Video Upload")
st.sidebar.subheader('Please Upload ONLY Match footage')

uploaded_video = st.sidebar.file_uploader("Select a video file.", type=["mp4"])

if uploaded_video:
    st.sidebar.video(uploaded_video)
    st.sidebar.write("Uploaded video:", uploaded_video.name)
    start_analysis = st.sidebar.button("Start Analysis", key="upload")

    if start_analysis:
        with st.spinner("Processing ..."):
            process_video(uploaded_video.read(), classes) 
            processed = True
        placeholder = st.empty()
        with placeholder.container():
            st.success("Video processing complete.")
            time.sleep(1)
        placeholder.empty()

# main page
st.title("PassingVision - Automated Analysis")
st.subheader("Computer Vision, Deep Learning & Unsupervised Machine Learning")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Usage", "Results", "Logs", " Line Graph Analysis", "Area Graph Analysis"])

with tab1:
    st.write("To use the automated analysis, follow these steps:")
    st.markdown("""

    Your video should only contain match footage, without replays or close up of players. For example, please see the demos.            

    1. Select the desired output options
    2. Upload a video or select a demo video
    3. Click on **Start Analysis**
    4. Go to the tab **Results** to see the output video
                
    For the best results, follow these video requirements:
    1. No sudden camera movements
    2. Minimum 720p quality
    3. No Zoom in on players
    4. No match replay
    5. No overcrowding
    6. Consistent 1/3 of the pitch always in view
    """)

with tab2:
    if processed:
        #open_video = open("output/output.mp4v")
        #read_video = open_video.read()
        st.video("output/graded_output.mp4")
        #st.video(read_video)
    #if processed:
        #with open("ouput/output.mp4", 'rb') as f:
        #        st.video(f.read())  ## need to be change on a live system
## Change back to default if I just install ffmpeg - didn't have permission to

with tab3:
    log_files_list = ["logs/tracking.log", "logs/camera_movement.log", "logs/memory_access.log"]

    selected_log_file = st.selectbox("Select Log File", log_files_list)

    try:
        with open(selected_log_file, "r") as log_file:
            log_contents = log_file.read()
        st.text_area("Logs", log_contents, height=450)
    except FileNotFoundError:
        st.error(f"Log file '{selected_log_file}' not found.")

with tab4: ## Graph that show data metrics - Refrance  - https://docs.streamlit.io/develop/api-reference/charts/
    st.write('Line Grah Data: Start Frame of Passes & Grades of Passes')
    pdf = pd.read_csv('/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv')
    #df_line = pd.DataFrame(rng(0).standard_normal((14, 2)), columns=["start_frame", "pass_grade"]) # Refrance - https://docs.streamlit.io/develop/api-reference/charts/st.line_chart
    st.line_chart(pdf, x="start_frame", y="pass_grade")

with tab5: ## Graoh that show data metrics
    st.write('Number of Passes per Grade')
    pdf = pd.read_csv('/home/c3646202/Desktop/FootballPassingAnalysisProject2/output/passes.csv')
    df = pd.DataFrame(rng(0).standard_normal((20, 5)), columns=["0.1", "0.3", "0.5", "0.7", "0.9"])
    st.bar_chart(df.iloc[0]) # Refrance - iloc[0] - https://docs.streamlit.io/develop/api-reference/data/st.dataframe