# Biometric-recognition-using-gait-analysis
This project was created by my team and me.Here we are able to identify people based on their walking styles.
A large part of the code was built using Marian Margeta's Gait Recognition repository:https://github.com/marian-margeta/gait-recognition.
We built on the dummy_pose_estimation and using the coordinates of the joints plotted a graph for the movement of 8 joints(Right and left knees,wrists,hips,pelvis and head top).Fourier Transform was performed on the graphs to give characteristic signatures.

To run the program:
1>This is the main repository for Gait analysis
2>All our dependencies,required folders,models are present in this repository.
3>The main file to execute is called Gait Recognition main.Run this .py file only for all the processes that you want to accomplish.
4>Change all the file locations according to your needs and run the code.
5>The GUI that opens,provides all the options to run the Project.
6>To provide data of new people click the "INSERT TRAINING VIDEO" option. 
7>To check for an unknown video click the "INSERT TESTING VIDEO" and after the process(the GUI starts responding again) click on "COMPARE".
8>The inserted test video's features is checked against the training dataset and the person with whom the error of the test video is the least,the person in the test video is assumed to be him/her.
9>The result is obtained on the Python Shell or can be checked by clicking the result option.
*Please take care that tkinter doesn't support MultiThreading.Please try to run the software with at the most two options used at a time.

Software used:
1>numpy-1.16.4
2>scipy<1.2.0
3>matplotlib-3.1.0
4>Open CV2-4.1.0
5>tkinter-8.6
6>mttkinter-0.6.1
7>imutils-0.5.2
8>pickle-4.0
9>sklearn-0.21.2

My team:
Sai Eashwar:https://github.com/MrHE1senberg
Monisha Chandra:https://github.com/llCriTicaLll
Adithya Bhat:https://github.com/AdithyaBhatPR
Sreyans Bothra(me):https://github.com/sreyansb
