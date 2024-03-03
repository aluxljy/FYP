# Tracking using Intelligent Scissors and CONDENSATION
Please ensure that Python is installed to run this project

Steps to run the project:
1. Download the ZIP file and open the folder in your desired IDE, preferable VS Code.

2. Create a virtual environment using the command ```py -3 -m venv .venv``` for Windows.
(*Make sure you are in the correct working directory which is the folder created at Step 1)

3. Activate the virtual environment using the command ```.venv\scripts\activate```.
(*Make sure you are currently in the correct working directory which is the folder created at Step 1)

4. Run the command ```pip install -r requirements.txt``` to install all the required libraries to run the project.

5. Run the main.py file to execute the project.

Overview of the project:

The proposed approach takes in a RGB video as input, segments the desired object and tracks it in every frame of the video sequence. It then outputs a video with a red coloured boundary that closely follows the contour of the said object throughout the tracking process. The tracking application developed is tested on an artificial test video ```Circle_Rolling.avi``` and a real-life recorded test video ```Tennis_Ball_Rolling_2.avi```, where the object in both test videos travels in a 2-dimensional direction and remains a uniform shape and size throughout the videos.

Results obtained for ```Circle_Rolling.avi``` and ```Tennis_Ball_Rolling_2.avi``` with one frame displayed respectively:

![image](https://github.com/aluxljy/FYP/assets/83107416/70227de3-97c7-486a-9493-9400be1b778a)
![image](https://github.com/aluxljy/FYP/assets/83107416/94c3546c-8875-4a09-8cdd-dd56b961b7bf)

The use of Intelligent Scissors presents a better way to represent the boundary information of a given object as shown in both results above. It is evident that the red line which is generated from the Intelligent Scissors provides vital boundary information that the Condensation algorithm uses to ensure it continues to track the yellow-coloured circle and the tennis ball throughout the image sequence.
