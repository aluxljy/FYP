# The following code was referred from tang, lang. (2023, March 9). Particle Filter. GitHub. https://github.com/tanglang96/particle-filter

import cv2 as cv
import numpy as np
import os
import settings

'''
Determine if the bin number of the current pixel value matches the target bin number.
Where x = the difference between the current bin number and the target bin number.
'''
def kronecker_delta(x):
    if x == 0:
        return 1
    else:
        return 0


'''
Increase the reliability of the colour distribution by employing a weighting function.
Assign smaller weights to pixels further away from the region centre.
Where r = the distance from the region centre. 
'''
def k(r):
    if r < 1:
        return 1 - r ** 2
    else:
        return 0


'''
A similarity measure between two colour distributions p and q.
'''
def bhattacharyya_coefficient(p, q):
    return np.sum(np.sqrt(p * q))


'''
Favor samples whose colour distributions are similar to the target model.
Small Bhattacharyya distances correspond to large weights.
Where variance = the Gaussian variance.
'''
def get_weight(b):
    variance = 0.01
    
    return np.exp(-(1. - b) / (2 * variance))


'''
Calculate the normalised cumulative probabilities.
'''
def get_random_index(weights):
    accumulated_weights = [0]
    index = []
    
    # calculate the normalised cumulative probabilities
    for i in range(len(weights)):
        accumulated_weights.append(accumulated_weights[i] + weights[i])
        
    for e in range(len(weights)):
        r = np.random.rand()  # generate a random number
        
        for i in range(len(weights)):
            if r > accumulated_weights[i] and r < accumulated_weights[i+1]:
                index.append(i)
    
    return index


'''
Object class for defining the state of the tracked object.
Where:
x, y = the horizontal and vertical starting point of the shortest cost path;
x_dot, y_dot = the motion with constant velocity;
Hx, Hy = the length of the half axes;
a_dot = the scale change.
'''
class state():
    def __init__(self, x, y, x_dot, y_dot, Hx, Hy, a_dot):
        self.x = x
        self.y = y
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.Hx = Hx
        self.Hy = Hy
        self.a_dot = a_dot
    
    '''
    Find the new shortest cost path of the tracked object using displacement.
    '''
    def calculate_path(self, initial_x, initial_y):
        displacement_x = self.x - initial_x
        displacement_y = self.y - initial_y
        path_list = settings.shortest_cost_path.copy()
        
        for n in range(len(path_list)):
            new_x = path_list[n][1] + displacement_x
            new_y = path_list[n][0] + displacement_y
            path_list[n] = tuple((new_y, new_x))
                
        return path_list

    '''
    Draw out the new shorted cost path of the tracked object.
    '''
    def draw_path(self, img, path, current_path_list):
        path_list = current_path_list.copy()
        
        while len(path_list) != 1:
            start_point = (path_list[0][1], path_list[0][0])
            end_point = (path_list[1][1], path_list[1][0])
            cv.line(img, start_point, end_point, (0, 0, 255), 1)
            path_list.pop(0)
            
        self.img = img
        cv.imwrite(path, self.img)
        
    '''
    Create a mask for the region of the tracked object.
    '''
    def create_mask(self, path_of_interest):
        self.mask = settings.mask.copy()
        
        for i in range (settings.num_rows):
            for j in range (settings.num_columns):
                if (i, j) in path_of_interest:
                    self.mask[i,j] = 0
        
        cv.floodFill(self.mask, None, (0, 0), 0)
        
        return self.mask

        
'''
Object class for producing colour histograms.
Where:
bin_num = the number of bins (histogram bars);
max_range = for HSV, in opencv, the range for H is 0 to 180 and the range for both S and V is 0 to 255;
divide = the starting value of each bin of the histogram;
frequency = the number of pixels in each bin.
'''               
class histogram():
    def __init__(self, bin_num=8, max_range=255.):
        self.bin_num = bin_num
        self.max_range = max_range
        self.divide = [max_range / bin_num * i for i in range(bin_num)]
        self.frequency = np.array([0. for i in range(bin_num)])

    '''
    Retrieve the corresponding bin that the colour belongs to.
    '''
    def get_hist_bin_location(self, x):
        for i in range(self.bin_num - 1):
            if (x >= self.divide[i] and x < self.divide[i+1]):
                return i
            elif (x > self.divide[-1] and x <= self.max_range):
                return self.bin_num - 1


'''
Intialise the initial state of the tracked object and compare the target colour histogram with the candidate colour histograms
of the sample position to predict its state in the corresponding time step.
'''
class Condensation():
    def __init__(self, num_of_samples=50, img_path='./test_video_images', output_path='./output_images'):
        self.num_of_samples = num_of_samples
        self.output_path = output_path
        self.DELTA_T = 0.05
        self.VELOCITY_DISTURB = 4.
        self.SCALE_DISTURB = 0.0
        self.SCALE_CHANGE_D = 0.001
        self.img_index = 0
        
        # # self.imgs = [os.path.join(img_path, '%01d.jpg' % (i + 1)) for i in range(66)]
        self.imgs = [os.path.join(img_path, '%01d.jpg' % (i + 1)) for i in range(100)]
        
        # starts processing from the first image
        print(self.imgs[0])
        first_img = cv.imread(self.imgs[0])
        print('processing image: %01d.jpg' % (self.img_index + 1))
        
        # # initial_state = state(x=40, y=214, x_dot=0., y_dot=0., Hx=34, Hy=37, a_dot=0.)
        initial_state = state(x=594, y=382, x_dot=0., y_dot=0., Hx=13, Hy=13, a_dot=0.)
        
        # # path_of_interest = initial_state.calculate_path(40, 214)
        path_of_interest = initial_state.calculate_path(594, 382)
        
        mask = initial_state.create_mask(path_of_interest)
        initial_state.draw_path(first_img, self.output_path + '/1.jpg', path_of_interest)
        self.state = initial_state
        
        # create samples randomly
        self.samples = []
        random_nums = np.random.normal(0, 0.4, (num_of_samples, 7))  # each sample has 7 variables
        self.weights = [1. / num_of_samples] * num_of_samples  # initially each sample has the same weight
        
        for i in range(num_of_samples):
            x0 = int(initial_state.x + (random_nums[i][0] * initial_state.Hx))
            y0 = int(initial_state.y + (random_nums[i][1] * initial_state.Hy))
            x_dot0 = initial_state.x_dot + (random_nums[i][2] * self.VELOCITY_DISTURB)
            y_dot0 = initial_state.y_dot + (random_nums[i][3] * self.VELOCITY_DISTURB)
            Hx0 = int(initial_state.Hx + (random_nums[i][4] * self.SCALE_DISTURB))
            Hy0 = int(initial_state.Hy + (random_nums[i][5] * self.SCALE_DISTURB))
            a_dot0 = initial_state.a_dot + (random_nums[i][6] * self.SCALE_CHANGE_D)
            sample = state(x0, y0, x_dot0, y_dot0, Hx0, Hy0, a_dot0)
            self.samples.append(sample)
        
        # # self.q = [histogram(bin_num=8, max_range=180), histogram(bin_num=8, max_range=255), histogram(bin_num=4, max_range=255)]
        self.q = [histogram(bin_num=8, max_range=180), histogram(bin_num=8, max_range=255), histogram(bin_num=4, max_range=255)]
        
        first_img = cv.imread(self.imgs[0])
        first_img = cv.cvtColor(first_img, cv.COLOR_BGR2HSV)

        # calculate the colour distribution of the target histogram q by assigning colours to their corresponding bins
        # iterate through the respective HSV histograms
        for hist in self.q:
            # iterate through every bin of the histogram
            for u in range(hist.bin_num):
                a = np.sqrt(initial_state.Hx ** 2 + initial_state.Hy ** 2)  # adapt the size of the region
                f = 0  # normalisation factor
                pixel_weight = []  # weightage of pixels according to distance from region centre
                x_bin = []  # boolean values to determine if the pixel value belongs to the corresponding bin
                
                for i in range(initial_state.x - initial_state.Hx, initial_state.x + initial_state.Hx):
                    for j in range(initial_state.y - initial_state.Hy, initial_state.y + initial_state.Hy):
                        if mask[j,i] == 255:
                            x_value = first_img[j][i][self.q.index(hist)]  # value of the pixel location in the corresponding histogram channel HSV
                            temp = k(np.linalg.norm((j - initial_state.y, i - initial_state.x)) / a)  # calculate using vector norm
                            f += temp
                            pixel_weight.append(temp)
                            x_bin.append(kronecker_delta(hist.get_hist_bin_location(float(x_value)) - u))
                            
                hist.frequency[u] = np.sum(np.array(pixel_weight) * np.array(x_bin)) / f  # colour distribution function
    
    '''
    Select a number of samples from the sample set of the previous time step with a specific probability.
    '''
    def select(self):
        if self.img_index < len(self.imgs) - 1:
            self.img_index += 1
        
        # starts processing from the second image onwards
        self.img = cv.imread(self.imgs[self.img_index])
        print('processing image: %01d.jpg' %(self.img_index + 1))
        
        # select new samples at random
        index = get_random_index(self.weights)
        new_samples = []
        
        for i in index:
            # may have duplicated samples
            new_samples.append(state(self.samples[i].x, self.samples[i].y, self.samples[i].x_dot, self.samples[i].y_dot, self.samples[i].Hx, self.samples[i].Hy, self.samples[i].a_dot))
        
        self.samples = new_samples

    '''
    Propagate each sample from the sample set of the previous time step by a linear stochastic differential equation.
    '''
    def propagate(self):
        for sample in self.samples:
            random_nums = np.random.normal(0, 0.4, 7)
            sample.x = int(sample.x + (sample.x_dot * self.DELTA_T) + (random_nums[0] * sample.Hx) + 0.5)
            sample.y = int(sample.y + (sample.y_dot * self.DELTA_T) + (random_nums[1] * sample.Hy) + 0.5)
            sample.x_dot = sample.x_dot + (random_nums[2] * self.VELOCITY_DISTURB)
            sample.y_dot = sample.y_dot + (random_nums[3] * self.VELOCITY_DISTURB)
            sample.Hx = int((sample.Hx * (sample.a_dot + 1)) + (random_nums[4] * self.SCALE_DISTURB) + 0.5)
            sample.Hy = int((sample.Hy * (sample.a_dot + 1)) + (random_nums[5] * self.SCALE_DISTURB) + 0.5)
            sample.a_dot = sample.a_dot + (random_nums[6] * self.SCALE_CHANGE_D)

    '''
    Observe and calculate the colour distribution for each sample from the sample set of the current time step.
    '''
    def observe(self):
        img = cv.imread(self.imgs[self.img_index])
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        B = []
        
        for i in range(self.num_of_samples):
            # check for out of bounds or invalid samples
            if (self.samples[i].x < 0 or self.samples[i].x > self.img.shape[1] - 1 or self.samples[i].y < 0 or self.samples[i].y > self.img.shape[0] - 1):
                B.append(0)
                continue
            
            # # path_of_interest = self.samples[i].calculate_path(40, 214)
            path_of_interest = self.samples[i].calculate_path(594, 382)
            
            mask = self.samples[i].create_mask(path_of_interest)
            
            # # self.p = [histogram(bin_num=8, max_range=180), histogram(bin_num=8, max_range=255), histogram(bin_num=4, max_range=255)]
            self.p = [histogram(bin_num=8, max_range=180), histogram(bin_num=8, max_range=255), histogram(bin_num=4, max_range=255)]
            
            # calculate the colour distribution of the candidate histogram p by assigning colours to their corresponding bins
            for hist in self.p:
                for u in range(hist.bin_num):
                    a = np.sqrt(self.samples[i].Hx ** 2 + self.samples[i].Hy ** 2)
                    f = 0
                    pixel_weight = []
                    x_bin = []
                    
                    for m in range(self.samples[i].x - self.samples[i].Hx, self.samples[i].x + self.samples[i].Hx):
                        for n in range(self.samples[i].y - self.samples[i].Hy, self.samples[i].y + self.samples[i].Hy):
                            # reassign values for pixels out of bounds
                            if n >= self.img.shape[0]:
                                n = img.shape[0] - 1
                            elif n < 0:
                                n = 0
                                
                            if m >= self.img.shape[1]:
                                m = img.shape[1] - 1
                            elif m < 0:
                                m = 0
                            
                            if mask[n,m] == 255:
                                x_value = img[n][m][self.p.index(hist)]
                                temp = k(np.linalg.norm((m - self.samples[i].x, n - self.samples[i].y)) / a)
                                f += temp
                                x_bin.append(kronecker_delta(hist.get_hist_bin_location(x_value) - u))
                                pixel_weight.append(temp)
                                
                    hist.frequency[u] = np.sum(np.array(pixel_weight) * np.array(x_bin)) / f
                    
            # calculate the Bhattacharyya coefficient for each sample of the set
            B_temp = bhattacharyya_coefficient(np.concatenate((self.p[0].frequency, self.p[1].frequency, self.p[2].frequency)), np.concatenate((self.q[0].frequency, self.q[1].frequency, self.q[2].frequency)))
            B.append(B_temp)
            
        # weight each sample of the set
        for i in range(self.num_of_samples):
            self.weights[i] = get_weight(B[i])
            
        self.weights /= sum(self.weights)
        
        for i in range(self.num_of_samples):
            print('dot: (%d,%d)  weight: %s' %(self.samples[i].x, self.samples[i].y, self.weights[i]))

    '''
    Estimate the mean state of the sample set at the current time step.
    '''
    def estimate(self):
        # weight each element of the set and calculate their mean state
        self.state.x = np.sum(np.array([s.x for s in self.samples]) * np.array(self.weights)).astype(int)
        self.state.y = np.sum(np.array([s.y for s in self.samples]) * np.array(self.weights)).astype(int)
        self.state.Hx = np.sum(np.array([s.Hx for s in self.samples]) * np.array(self.weights)).astype(int)
        self.state.Hy = np.sum(np.array([s.Hy for s in self.samples]) * np.array(self.weights)).astype(int)
        self.state.x_dot = np.sum(np.array([s.x_dot for s in self.samples]) * np.array(self.weights))
        self.state.y_dot = np.sum(np.array([s.y_dot for s in self.samples]) * np.array(self.weights))
        self.state.a_dot = np.sum(np.array([s.a_dot for s in self.samples]) * np.array(self.weights))
        print('img: %s  x: %s  y: %s  Hx: %s  Hy: %s  x_dot: %s  y_dot: %s  a_dot: %s' %(self.img_index + 1, self.state.x, self.state.y, self.state.Hx, self.state.Hy, self.state.x_dot, self.state.y_dot, self.state.a_dot))
        
        # # path_of_interest = self.state.calculate_path(40, 214)
        path_of_interest = self.state.calculate_path(594, 382)
        
        self.state.draw_path(self.img, self.output_path + '/%01d.jpg' %(self.img_index + 1), path_of_interest)
