import glob
import os
from PIL import Image


def make_gif(frame_folder):
    frame_names = [image for image in glob.glob(f"{frame_folder}*.png")]
    frame_names.sort()  # The images are not sorted by frame name
    frames = [Image.open(image) for image in frame_names]
    frame_one = frames[0]
    frame_one.save("/Users/pmhaughe/Desktop/Simulate.gif", format="GIF", append_images=frames[1:],
                   save_all=True, duration=10, loop=0)


def delete_frames(frame_folder):
    folder_name = os.listdir(frame_folder)

    for pic in folder_name:
        os.remove(os.path.join(frame_folder, pic))
