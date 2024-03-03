import cv2
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class PixelDetails:
    x: int = 0
    y: int = 0
    R: int = 0
    G: int = 0
    B: int = 0
    Grey: float = 0.0
    SGrey: float = 0.0
    H: float = 0.0
    S: float = 0.0
    V: float = 0.0
    Gm: float = 0.0
    Gd: float = 0.0
    Gx: float = 0.0
    Gy: float = 0.0
    Lap: float = 0.0
    fZ: float = 1.0
    fG: float = 0.0
    fD: np.ndarray = np.empty(9, dtype=float)
    Dp: tuple = (0, 0)
    Lpq: np.ndarray = np.empty(9, dtype=tuple)
    localCost: np.ndarray = np.empty(9, dtype=tuple)
    totalCost: float = float('inf')
    expanded: bool = False
    pointerTo: tuple = None


test_video_folder_path = './test_videos'
image_folder_path = './test_video_images'
# # image_folder_path = './test_images'
output_image_folder_path = './output_video_images'
output_video_folder_path = './output_videos'


def video_to_images(video_filename):
    vid_cap = cv2.VideoCapture(os.path.join(test_video_folder_path, video_filename))
    second = 0
    
    # capture a frame at every 0.1 seconds so that for every second the video is split into 10 frames
    # # frame_rate = 0.1
    
    # capture a frame at every 0.05 seconds so that for every second the video is split into 20 frames
    frame_rate = 0.05
    
    success = True
    count = 1

    while success:
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = vid_cap.read()
        
        if success:
            cv2.imwrite((os.path.join(image_folder_path, str(count) + '.jpg')), frame)
        
        second += frame_rate
        count += 1


def extract_image(image_filename):
    global im_matrix
    global im_smoothened_matrix
    global num_rows
    global num_columns
    global pixel_details_matrix
    global active_list
    global shortest_cost_path
    global segmented_image
    global mask
    
    image = cv2.imread(os.path.join(image_folder_path, image_filename), 1)
    im_matrix = image[:,:]
    smoothened_image =  cv2.GaussianBlur(image.copy(), (3, 3), 0)
    im_smoothened_matrix = smoothened_image[:,:]
    num_rows = len(im_matrix)
    num_columns = len(im_matrix[0])
    
    pixel_details_matrix = np.empty((num_rows,num_columns), dtype=PixelDetails)
    
    active_list = []
    shortest_cost_path = []
    segmented_image = cv2.imread(os.path.join(image_folder_path, image_filename), 1)
    
    mask = np.full((num_rows,num_columns), 255, np.uint8)
    

def images_to_video(output_video_filename):
    frames = []
    # # fps = 10
    fps = 20
    
    for file in os.listdir(output_image_folder_path):
        frames.append(file)
    
    # split the root and extension part of the path name of the frames and sort them in ascending order
    frames.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    frame = cv2.imread(os.path.join(output_image_folder_path, frames[0]))
    height, width, _ = frame.shape
    size = (width, height)
            
    vid_out = cv2.VideoWriter(os.path.join(output_video_folder_path, output_video_filename), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for n in frames:
        vid_out.write(cv2.imread(os.path.join(output_image_folder_path, n)))
    
    vid_out.release()