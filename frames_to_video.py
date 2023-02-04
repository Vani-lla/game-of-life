import cv2 as cv
from glob import glob

# Read all frames from ./frames directory
frames = [cv.imread(path) for path in sorted(glob('frames/frame*.png'), key=lambda path: int(path[12:].split('.')[0]))]
*size, _ = frames[0].shape

# Creating video from frames
fps = int(input("How many FPS: "))
video_format = input("Select video format ('mp4'/'avi'): ")

if video_format == 'mp4':
   video = cv.VideoWriter('Game_of_life.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]))
else:
   video = cv.VideoWriter('Game_of_life.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, (size[1], size[0]))

for frame in frames: video.write(frame)
video.release()
