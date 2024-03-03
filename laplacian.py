import settings

def compute_laplacian(rows, columns, diagonal = False):
    for i in range(rows):
        for j in range(columns):
            if i == 0 or j == 0 or i == rows - 1 or j == columns - 1:
                calculated_laplacian = 1.0
            else:
                if diagonal == False:
                    calculated_laplacian = (1 * settings.pixel_details_matrix[i-1,j].SGrey) + (1 * settings.pixel_details_matrix[i,j-1].SGrey) + (- 4 * settings.pixel_details_matrix[i,j].SGrey) + (1 * settings.pixel_details_matrix[i,j+1].SGrey) + (1 * settings.pixel_details_matrix[i+1,j].SGrey)
                    settings.pixel_details_matrix[i,j].Lap = calculated_laplacian
                elif diagonal == True:
                    calculated_laplacian = (1 * settings.pixel_details_matrix[i-1,j-1].SGrey) + (1 * settings.pixel_details_matrix[i-1,j].SGrey) + (1 * settings.pixel_details_matrix[i-1,j+1].SGrey) + (1 * settings.pixel_details_matrix[i,j-1].SGrey) + (- 8 * settings.pixel_details_matrix[i,j].SGrey) + (1 * settings.pixel_details_matrix[i,j+1].SGrey) + (1 * settings.pixel_details_matrix[i+1,j-1].SGrey) + (1 * settings.pixel_details_matrix[i+1,j].SGrey) + (1 * settings.pixel_details_matrix[i+1,j+1].SGrey)
                    settings.pixel_details_matrix[i,j].Lap = calculated_laplacian