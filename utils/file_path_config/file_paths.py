from pathlib import Path ## https://docs.python.org/3/library/pathlib.html

project_direcotory = Path(__file__).resolve().parent.parent.parent


output = project_direcotory / "output"
models = project_direcotory / "models"



passes_csv = output / "passes.csv"
data_csv = output / "data.csv"
output_video = output / "graded_output.mp4"
model_path = models / "model.pt"
video_file  =  output / "output.mp4"