import yaml
import multiprocessing
from Multi_threading_fight_detection import run_fight_detection

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

profile = config["profiles"][config["default_profile"]]
input_video_paths = profile.get("input_video_paths", [])
display = profile.get("display", True) 

# Start a separate process for each video source
processes = []
for i, cam_index in enumerate(input_video_paths):
    p = multiprocessing.Process(
        target=run_fight_detection,
        args=(cam_index, profile, f"Camera {i+1}",display)
    )
    p.start()
    processes.append(p)

# Join processes
for p in processes:
    p.join()
