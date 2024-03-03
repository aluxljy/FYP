import cv2
import numpy as np
import os
import time
import settings
from preprocessing import *
from sobel_operator import *
from laplacian import *
from cost_computation import *
from intelligent_scissors import *
from condensation import Condensation

start_time = time.time()

def scaling(value):
    if value > 255:
        scaled_value = 255
    elif value < 0:
        scaled_value = 0
    else:
        # type casting into integer truncates the decimals
        scaled_value = int(value)
    
    return scaled_value

# # video_filename = 'Circle_Rolling.avi'
video_filename = 'Tennis_Ball_Rolling_2.avi'

settings.video_to_images(video_filename)

image_filename = '1.jpg'
             
# # settings.extract_image('Black_Ellipse.jpg')
# # settings.extract_image('Lenna.png')
settings.extract_image(image_filename)
rgb_to_grey(settings.im_matrix, settings.num_rows, settings.num_columns)

# # bgr_im = np.empty((settings.num_rows,settings.num_columns,3))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         bgr_im[i,j] = np.array([settings.pixel_details_matrix[i,j].B, settings.pixel_details_matrix[i,j].G, settings.pixel_details_matrix[i,j].R])

# # cv2.imshow('RGB', np.array(bgr_im, dtype=np.uint8))
# # cv2.waitKey(0)

# # grey_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         grey_im[i,j] = settings.pixel_details_matrix[i,j].Grey

# # cv2.imshow('Greyscale', np.array(grey_im, dtype=np.uint8))
# # cv2.waitKey(0)

compute_gradients(settings.num_rows, settings.num_columns, 'x')

# # gradient_x_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         gradient_x = settings.pixel_details_matrix[i,j].Gx
# #         scaled_gradient_x = scaling(gradient_x)  
# #         gradient_x_im[i,j] = scaled_gradient_x

# # cv2.imshow('GradientX', np.array(gradient_x_im, dtype=np.uint8))
# # cv2.waitKey(0)

compute_gradients(settings.num_rows, settings.num_columns, 'y')

# # gradient_y_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         gradient_y = settings.pixel_details_matrix[i,j].Gy
# #         scaled_gradient_y = scaling(gradient_y)  
# #         gradient_y_im[i,j] = scaled_gradient_y

# # cv2.imshow('GradientY', np.array(gradient_y_im, dtype=np.uint8))
# # cv2.waitKey(0)

compute_gradients(settings.num_rows, settings.num_columns, 'magnitude')

# # gradient_m_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         gradient_m = settings.pixel_details_matrix[i,j].Gm
# #         scaled_gradient_m = scaling(gradient_m)  
# #         gradient_m_im[i,j] = scaled_gradient_m

# # cv2.imshow('GradientM', np.array(gradient_m_im, dtype=np.uint8))
# # cv2.waitKey(0)

rgb_to_grey(settings.im_smoothened_matrix, settings.num_rows, settings.num_columns, smoothened=True)

# # smoothened_grey_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         smoothened_grey_im[i,j] = settings.pixel_details_matrix[i,j].SGrey

# # cv2.imshow('Smoothened Greyscale', np.array(smoothened_grey_im, dtype=np.uint8))
# # cv2.waitKey(0)

compute_laplacian(settings.num_rows, settings.num_columns, diagonal=True)

# # laplacian_im = np.empty((settings.num_rows,settings.num_columns))

# # for i in range (settings.num_rows):
# #     for j in range (settings.num_columns):
# #         laplacian = settings.pixel_details_matrix[i,j].Lap
# #         scaled_laplacian = scaling(laplacian)  
# #         laplacian_im[i,j] = scaled_laplacian

# # cv2.imshow('Laplacian', np.array(laplacian_im, dtype=np.uint8))
# # cv2.waitKey(0)

compute_laplacian_feature_cost(settings.num_rows, settings.num_columns)
compute_gradient_magnitude_feature_cost(settings.num_rows, settings.num_columns)
compute_perpendicular_unit_vector(settings.num_rows, settings.num_columns)
compute_neighbourhood_link(settings.num_rows, settings.num_columns)
compute_gradient_direction_feature_cost(settings.num_rows, settings.num_columns)
compute_local_cost(settings.num_rows, settings.num_columns, 0.21, 0.16, 0.63)

# Lenna.png
# # seed_list = [(202, 334), (264, 353), (336, 334), (388, 306), (371, 232), (292, 218), (202, 334)]
# # seed_list = [(202, 334), (216, 346), (236, 352), (264, 353), (295, 348), (336, 334), (382, 309), (388, 306), (387, 289), (385, 264), (382, 247), (371, 232), (354, 226), (314, 219), (292, 218), (202, 334)]
# # seed_list = [(61, 163), (55, 184), (45, 222), (146, 345), (111, 399), (209, 370), (213, 365), (223, 367), (257, 370), (345, 381), (390, 384), (406, 381), (425, 377), (457, 374), (510, 382), (510, 316), (510, 240), (510, 175), (510, 60), (450, 63), (407, 64), (380, 78), (354, 79), (318, 87), (294, 87), (293, 81), (276, 90), (237, 138), (104, 133), (90, 143), (61, 163)]

# Black_Ellipse.jpg
# # seed_list = [(193, 357), (368, 596), (542, 356), (368, 117), (193, 357)]
# # seed_list = [(193, 360), (227, 500), (367, 597), (489, 529), (542, 356), (506, 212), (368, 117), (232, 206), (193, 360)]

# Circle_Rolling.avi
# # seed_list = [(177, 39), (192, 66), (214, 74), (235, 66), (250, 39), (239, 16), (214, 7), (186, 19), (177, 39)]

# Tennis_Ball_Rolling_2.avi
seed_list = [(369, 592), (368, 597), (372, 603), (376, 606), (381, 607), (383, 607), (387, 608), (391, 605), (394, 602), (396, 593), (394, 587), (391, 584), (388, 583), (384, 581), (380, 581), (376, 582), (374, 584), (372, 586), (370, 590), (369, 592)]

while len(seed_list) != 1:
    graph_search(seed_list[0][0], seed_list[0][1], seed_list[1][0], seed_list[1][1])
    find_shortest_cost_path(seed_list[0][0], seed_list[0][1], seed_list[1][0], seed_list[1][1])
    print('%d seeds left to process' %(len(seed_list) - 1))
    seed_list.pop(0)

draw_shortest_cost_path()

end_time = time.time()
time_elapsed = (end_time - start_time) / 60
print('Time elapsed to execute the program is', time_elapsed, 'minutes')

cv2.imshow('Segmentation', settings.segmented_image)
cv2.waitKey(0)

samples_num = 30
image_path = './test_video_images'
output_path = './output_video_images'
c = Condensation(samples_num, image_path, output_path)

for c.img_index in range(len(c.imgs)):
    c.select()
    c.propagate()
    c.observe()
    c.estimate()

settings.images_to_video('Test.avi')