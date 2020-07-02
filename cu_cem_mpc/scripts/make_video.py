import cv2
import os
import re
import imageio

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):
    return sorted(v_list, key=str2int)


image_folder = 'tmp'
video_name = 'movie.gif'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = sort_humanly(images)
imgs = []
for filename in images:
    imgs.append(imageio.imread('./'+image_folder+'/'+filename))
imageio.mimsave(video_name, imgs)