import numpy as np
import settings

def rgb_to_grey(im_matrix, rows, columns, smoothened = False):
    for i in range(rows):
        for j in range(columns):
            pixel_position = im_matrix[i,j]
            original_b = pixel_position[0]
            original_g = pixel_position[1]
            original_r = pixel_position[2]

            greyscale = 0.30 * original_r + 0.59 * original_g + 0.11 * original_b
            
            if smoothened == False:
                pixel = settings.PixelDetails(x=j, y=i, R=original_r, G=original_g, B=original_b, Grey=greyscale)
                settings.pixel_details_matrix[i,j] = pixel
            elif smoothened == True:
                settings.pixel_details_matrix[i,j].SGrey = greyscale