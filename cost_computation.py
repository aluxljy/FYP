import math
import numpy as np
import settings

def compute_laplacian_feature_cost(rows, columns):
    print('Computing Laplacian feature cost...')
    for i in range(rows):
        for j in range(columns):
            if settings.pixel_details_matrix[i,j].Lap == 0:
                settings.pixel_details_matrix[i,j].fZ = 0


def compute_gradient_magnitude_feature_cost(rows, columns):
    print('Computing gradient magnitude feature cost...')
    Gmax = find_maximum_gradient_magnitude(rows, columns)
    
    for i in range(rows):
        for j in range(columns):
            G = settings.pixel_details_matrix[i,j].Gm
            settings.pixel_details_matrix[i,j].fG = 1 - (G / Gmax)


def find_maximum_gradient_magnitude(rows, columns):
    max_gradient_m = 0
    
    for i in range(rows):
        for j in range(columns):
            if settings.pixel_details_matrix[i,j].Gm > max_gradient_m:
                max_gradient_m = settings.pixel_details_matrix[i,j].Gm
                
    return max_gradient_m


def compute_gradient_direction_feature_cost(rows, columns):
    print('Computing gradient direction feature cost...')
    for i in range(rows):
        for j in range(columns):
            if not(i == 0 or j == 0 or i == rows - 1 or j == columns - 1):
                Dp = settings.pixel_details_matrix[i,j].Dp
                Dq_list = []
                link_array = settings.pixel_details_matrix[i,j].Lpq
                
                # compute gradient direction feature cost between pixel (i,j) to neighbouring pixel (y,x) 
                for y in range(i-1, i+2):
                    for x in range(j-1, j+2):
                        Dq = settings.pixel_details_matrix[y,x].Dp
                        Dq_list.append(Dq)
                
                fD_array = np.empty(9, dtype=float)
                for n in range(9):
                    if n == 4:
                        fD_array[n] = 1.0
                    else:
                        dp = Dp[0] * link_array[n][0] + Dp[1] * link_array[n][1]
                        dq = link_array[n][0] * Dq_list[n][0] + link_array[n][1] * Dq_list[n][1]
                        
                        dp_inv = dp ** -1
                        dq_inv = dq ** -1
                        
                        if dp_inv == float('inf') or dp_inv  == float('-inf') or dq_inv == float('inf') or dq_inv == float('-inf'):
                            fD_array[n] = 1.0
                        else:
                            fD_array[n] = (1 / math.pi) * (math.cos(dp_inv) + math.cos(dq_inv))
                           
                settings.pixel_details_matrix[i,j].fD = fD_array


def compute_perpendicular_unit_vector(rows, columns):
    print('Computing perpendicular unit vector...')
    for i in range(rows):
        for j in range(columns):
            y = settings.pixel_details_matrix[i,j].Gy
            x = settings.pixel_details_matrix[i,j].Gx
            
            if (y > 0 and x == 0):
                perpendicular_y = 0
                perpendicular_x = 1
            elif (y == 0 and x > 0):
                perpendicular_y = -1
                perpendicular_x = 0
            elif (y < 0 and x == 0):
                perpendicular_y = 0
                perpendicular_x = -1
            elif (y == 0 and x < 0):
                perpendicular_y = 1
                perpendicular_x = 0
            elif (y == 0 and x == 0):
                perpendicular_y = 0
                perpendicular_x = 0
            elif (y > 0 and x > 0) or ( y < 0 and x > 0):
                perpendicular_y = -1 * math.sqrt(1 / ((y ** 2 / x ** 2) + 1))
                perpendicular_x = -1 * (y / x) * perpendicular_y
            else:
                perpendicular_y = math.sqrt(1 / ((y ** 2 / x ** 2) + 1))
                perpendicular_x = -1 * (y / x) * perpendicular_y
            
            settings.pixel_details_matrix[i,j].Dp = tuple((perpendicular_y, perpendicular_x))
    

def compute_neighbourhood_link(rows, columns):
    print('Computing neighbourhood link...')
    for i in range(rows):
        for j in range(columns):
            if not(i == 0 or j == 0 or i == rows - 1 or j == columns - 1):
                p = tuple((settings.pixel_details_matrix[i,j].Gy, settings.pixel_details_matrix[i,j].Gx))
                Dp = settings.pixel_details_matrix[i,j].Dp
                neighbouring_pixel_list = []
                
                # compute edge vector between pixel (i,j) and neighbouring pixel (y,x)
                for y in range(i-1, i+2):
                    for x in range(j-1, j+2):
                        q = tuple((settings.pixel_details_matrix[y,x].Gy, settings.pixel_details_matrix[y,x].Gx))
                        neighbouring_pixel_list.append(q)
                        
                link_array = np.empty(9, dtype=tuple)
                
                for n in range(9):
                    if n == 4:
                        link_array[n] = tuple((0, 0))
                    else:
                        if (Dp[0] * (neighbouring_pixel_list[n][0] - p[0]) + Dp[1] * (neighbouring_pixel_list[n][1] - p[1])) >= 0:
                            Lpq = tuple((neighbouring_pixel_list[n][0] - p[0], neighbouring_pixel_list[n][1] - p[1]))
                            link_array[n] = Lpq
                        else:
                            Lpq = tuple((p[0] - neighbouring_pixel_list[n][0], p[1] - neighbouring_pixel_list[n][1]))
                            link_array[n] = Lpq
                
                settings.pixel_details_matrix[i,j].Lpq = link_array
            

def compute_local_cost(rows, columns, wZ, wD, wG):   
    print('Computing local cost...')
    for i in range(rows):
        for j in range(columns):
            if not(i == 0 or j == 0 or i == rows - 1 or j == columns - 1):
                fZ_list = []
                fD_array = settings.pixel_details_matrix[i,j].fD
                fG_list = []
                
                # compute local cost on the directed link from pixel (i,j) to neighbouring pixel (y,x) 
                for y in range(i-1, i+2):
                    for x in range(j-1, j+2):
                        fZ = settings.pixel_details_matrix[y,x].fZ
                        fZ_list.append(fZ)
                        fG = settings.pixel_details_matrix[y,x].fG
                        fG_list.append(fG)
                           
                local_cost_array = np.empty(9, dtype=float)
                
                for n in range(9):       
                    if (n % 2) == 0:
                        # if pixel (y,x) is a diagonal neighbour to pixel (i,j)
                        # then scale fG of pixel (y,x) by 1
                        local_cost_to_neighbouring_pixel = wZ * fZ_list[n] + wD * fD_array[n] + wG * fG_list[n]
                        local_cost_array[n] = local_cost_to_neighbouring_pixel
                    else:
                        # if pixel (y,x) is a horizontal or vertical neighbour to pixel (i,j)
                        # then scale fG of pixel (y,x) by 1/(2)^(1/2)
                        local_cost_to_neighbouring_pixel = wZ * fZ_list[n] + wD * fD_array[n] + wG * (fG_list[n] / math.sqrt(2))
                        local_cost_array[n] = local_cost_to_neighbouring_pixel
                                                    
                settings.pixel_details_matrix[i,j].localCost = local_cost_array