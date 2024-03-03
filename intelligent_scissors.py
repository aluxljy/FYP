import cv2
import math
import settings

def graph_search(start_y, start_x, end_y, end_x):
    seed = tuple((start_y, start_x))
    settings.pixel_details_matrix[start_y,start_x].totalCost = 0
    settings.active_list.append(seed)
    
    while len(settings.active_list) != 0 and not(settings.pixel_details_matrix[end_y,end_x].expanded):
        p = find_minimum_cost_pixel()
        settings.active_list.remove(p)
        p_y = p[0]
        p_x = p[1]
        settings.pixel_details_matrix[p_y,p_x].expanded = True
        p_cost = settings.pixel_details_matrix[p_y,p_x].totalCost
        pq_cost_array = settings.pixel_details_matrix[p_y,p_x].localCost
        pixel_counter = 0
                
        for y in range(p_y-1, p_y+2):
            for x in range(p_x-1, p_x+2):
                if not(settings.pixel_details_matrix[y,x].expanded) and pq_cost_array[pixel_counter] != None:
                    q = tuple((y, x))
                    q_cost = settings.pixel_details_matrix[y,x].totalCost
                    
                    if pixel_counter % 2 == 0:
                        new_q_cost = min(p_cost + (pq_cost_array[pixel_counter] * math.sqrt(2)), q_cost)
                    else:
                        new_q_cost = min(p_cost + pq_cost_array[pixel_counter], q_cost)
                    
                    if new_q_cost != q_cost:
                        settings.pixel_details_matrix[y,x].totalCost = new_q_cost
                        settings.pixel_details_matrix[y,x].pointerTo = p
                    
                    if q not in settings.active_list:
                        settings.active_list.append(q)
                        
                pixel_counter += 1
                        
    settings.active_list = []


def find_minimum_cost_pixel():
    min_cost = float('inf')
    l = settings.active_list
    
    for n in range(len(l)):
        if not(l[n][0] == 0 or l[n][1] == 0 or l[n][0] == settings.num_rows - 1 or l[n][1] == settings.num_columns - 1):
            if settings.pixel_details_matrix[l[n][0],l[n][1]].totalCost < min_cost:
                min_cost = settings.pixel_details_matrix[l[n][0],l[n][1]].totalCost
                min_cost_pixel = l[n]
                
    return min_cost_pixel


def find_shortest_cost_path(start_y, start_x, end_y, end_x):
    seed_point = tuple((start_y, start_x))
    end_point = tuple((end_y, end_x))
    temp = []
    temp.append(end_point)
    
    # cv2.circle(settings.segmented_image, (start_x, start_y), 2, (0, 0, 255), 5)
    
    while seed_point not in temp:
        current_point = temp[len(temp) - 1]
        target_point = settings.pixel_details_matrix[current_point[0],current_point[1]].pointerTo
        temp.append(target_point)
    
    temp.reverse()
    temp.pop(-1)
    settings.shortest_cost_path.extend(temp)
    reset_attributes()
        

def draw_shortest_cost_path():
    path_list = settings.shortest_cost_path.copy()
    
    while len(path_list) != 1:
        start_point = (path_list[0][1], path_list[0][0])
        end_point = (path_list[1][1], path_list[1][0])
        cv2.line(settings.segmented_image, start_point, end_point, (0, 0, 255), 1)
        path_list.pop(0)
    

def reset_attributes():
    for i in range (settings.num_rows):
        for j in range (settings.num_columns):
            settings.pixel_details_matrix[i,j].totalCost = float('inf')
            settings.pixel_details_matrix[i,j].expanded = False
            settings.pixel_details_matrix[i,j].pointerTo = None