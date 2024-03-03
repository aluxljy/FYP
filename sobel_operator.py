import math
import settings

def compute_gradients(rows, columns, orientation):
    for i in range(rows):
        for j in range(columns):
            if i == 0 or j == 0 or i == rows - 1 or j == columns - 1:
                calculated_gradient = 0
            else:
                if orientation == 'x':
                    calculated_gradient = settings.pixel_details_matrix[i-1,j-1].Grey + 2 * settings.pixel_details_matrix[i,j-1].Grey + settings.pixel_details_matrix[i+1,j-1].Grey - settings.pixel_details_matrix[i-1,j+1].Grey - 2 * settings.pixel_details_matrix[i,j+1].Grey - settings.pixel_details_matrix[i+1,j+1].Grey
                    settings.pixel_details_matrix[i,j].Gx = calculated_gradient
                elif orientation == 'y':
                    calculated_gradient = settings.pixel_details_matrix[i-1,j-1].Grey + 2 * settings.pixel_details_matrix[i-1,j].Grey + settings.pixel_details_matrix[i-1,j+1].Grey - settings.pixel_details_matrix[i+1,j-1].Grey - 2 * settings.pixel_details_matrix[i+1,j].Grey - settings.pixel_details_matrix[i+1,j+1].Grey
                    settings.pixel_details_matrix[i,j].Gy = calculated_gradient
                elif orientation == 'magnitude':
                    calculated_gradient = math.sqrt(settings.pixel_details_matrix[i,j].Gx ** 2 + settings.pixel_details_matrix[i,j].Gy ** 2)
                    settings.pixel_details_matrix[i,j].Gm = calculated_gradient