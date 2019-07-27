resultfinal="0"
def dummyposetrain():
    import numpy as np
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import os
    import cv2
    from scipy.misc import imresize, imread
    from human_pose_nn import HumanPoseIRNetwork
    from scipy.spatial import distance
    from scipy.fftpack import fft, ifft
    net_pose = HumanPoseIRNetwork()
    net_pose.restore('models/MPII+LSP.ckpt')


    """def plotting(p,t,s):
        x = [i for i in range(1, len(all_file))]
        p=fft(p)
        t=fft(t)
        p=[i.real for i in p]
        plt.axis("off")
        #t=[i.real for i in p]
        #print(p,t)
        plt.plot(p, "r",label="rknee")
        
        
        # plt.legend()
        plt.show()"""

    def skeleton(ipath):
        img = imread(ipath)
        # img2 = imread('white.png')
        img = imresize(img, [299, 299])
        # img2 = imresize(img2, [299, 299])
        img_batch = np.expand_dims(img, 0)

        y, x, a = net_pose.estimate_joints(img_batch)
        y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

        # checking unique features for a person

        """elbow = (x[11], y[11], 0)
        wrist = (x[10], y[10], 0)
        shoulder = (x[12], y[12], 0)
        hip = (x[2], y[2], 0)
        knee = (x[1], y[1], 0)"""
        rhip = y[2]
        lhip= y[3]
        rwrist=y[10]
        lwrist=y[15]
        rknee=y[1]
        lknee=y[4]
        pelvis=y[6]
        head=y[9]
        
        """elbow_wrist = distance.euclidean(elbow, wrist)
        shoulder_elbow = distance.euclidean(elbow, shoulder)
        hip_knee = distance.euclidean(hip, knee)
        ewlist.append(elbow_wrist)
        shellist.append(shoulder_elbow)
        hiknlist.append(hip_knee)"""
        rhipl.append(rhip)
        lhipl.append(lhip)
        
        lwristl.append(lwrist)
        rwristl.append(rwrist)
        
        lkneel.append(lknee)
        rkneel.append(rknee)
        
        pelvisl.append(pelvis)
        
        headl.append(head)
        
        # ylist.append(str(y[1])) # only for right knee , value is in string.For comparing,convert to int

        joint_names = [
            'right ankle ',
            'right knee ',
            'right hip',
            'left hip',
            'left knee',
            'left ankle',
            'pelvis',
            'thorax',
            'upper neck',
            'head top',
            'right wrist',
            'right elbow',
            'right shoulder',
            'left shoulder',
            'left elbow',
            'left wrist'
        ]

        # Print probabilities of each estimation
        """for i in range(16):
            print('%s: %.02f%%' % (joint_names[i], a[i] * 100))"""

        # Create image
        """colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
        for i in range(16):
            if i < 15 and i not in {5, 9}:
                plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colors[i], linewidth=3)
        plot_img(img2, ipath)"""

    # change the directory to images folder to store the frames obtained from the training videos.
    #if not present create a new directory in the same folder as the program.
    os.chdir(r"images")
    img_dir = os.getcwd()
    image_list, rhipl,lhipl,rkneel,lkneel,rwristl,lwristl,headl,pelvisl=[],[],[],[],[],[],[],[],[]

    jointll=[rhipl,lhipl,rkneel,lkneel,rwristl,lwristl,headl,pelvisl]
    strjoint=["rhipl","lhipl","rkneel","lkneel","rwristl","lwristl","headl","pelvisl"]

    all_file = os.listdir(img_dir)
    
    for file in all_file:
        if file.endswith(".jpg"):
            skeleton(file)
    #print(ranklelist,lanklelist)
    # Plotting the unique points/lengths
    # Making a folder 'traindata' which stores the data of new people who data is given to train.
    os.mkdir("TrainData/"+str(inputValue))
    print("Processing fft..")
    for i in range(len(jointll)):
        jointll[i]=[x.real for x in fft(jointll[i])]
        print("Creating "+strjoint[i]+".txt")
        txtfile=open("TrainData/"+str(inputValue)+"/"+strjoint[i]+".txt","w")
        txtfile.write(str(jointll[i]))
        txtfile.close()
    for i in all_file:
        os.remove(i)
    print("Finished...!")
def dummyposetest():
    import numpy as np
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt
    import os
    import cv2
    from scipy.misc import imresize, imread
    from human_pose_nn import HumanPoseIRNetwork
    from scipy.spatial import distance
    from scipy.fftpack import fft, ifft
    net_pose = HumanPoseIRNetwork()
    net_pose.restore('models/MPII+LSP.ckpt')


    """def plotting(p,t,s):
        x = [i for i in range(1, len(all_file))]
        p=fft(p)
        t=fft(t)
        p=[i.real for i in p]
        plt.axis("off")
        #t=[i.real for i in p]
        #print(p,t)
        plt.plot(p, "r",label="rknee")
        
        
        # plt.legend()
        plt.show()"""

    def skeleton(ipath):
        img = imread(ipath)
        # img2 = imread('white.png')
        img = imresize(img, [299, 299])
        # img2 = imresize(img2, [299, 299])
        img_batch = np.expand_dims(img, 0)

        y, x, a = net_pose.estimate_joints(img_batch)
        y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

        # checking unique features for a person

        """elbow = (x[11], y[11], 0)
        wrist = (x[10], y[10], 0)
        shoulder = (x[12], y[12], 0)
        hip = (x[2], y[2], 0)
        knee = (x[1], y[1], 0)"""
        rhip = y[2]
        lhip= y[3]
        rwrist=y[10]
        lwrist=y[15]
        rknee=y[1]
        lknee=y[4]
        pelvis=y[6]
        head=y[9]
        
        """elbow_wrist = distance.euclidean(elbow, wrist)
        shoulder_elbow = distance.euclidean(elbow, shoulder)
        hip_knee = distance.euclidean(hip, knee)
        ewlist.append(elbow_wrist)
        shellist.append(shoulder_elbow)
        hiknlist.append(hip_knee)"""
        rhipl.append(rhip)
        lhipl.append(lhip)
        
        lwristl.append(lwrist)
        rwristl.append(rwrist)
        
        lkneel.append(lknee)
        rkneel.append(rknee)
        
        pelvisl.append(pelvis)
        
        headl.append(head)
        
        # ylist.append(str(y[1])) # only for right knee , value is in string.For comparing,convert to int

        joint_names = [
            'right ankle ',
            'right knee ',
            'right hip',
            'left hip',
            'left knee',
            'left ankle',
            'pelvis',
            'thorax',
            'upper neck',
            'head top',
            'right wrist',
            'right elbow',
            'right shoulder',
            'left shoulder',
            'left elbow',
            'left wrist'
        ]
    # Changing the directory again to images folder so that frames from the testing videos can be saved.
    #if not present create a new directory in the same folder as the program.
    os.chdir(r"images")
    img_dir = os.getcwd()
    image_list, rhipl,lhipl,rkneel,lkneel,rwristl,lwristl,headl,pelvisl=[],[],[],[],[],[],[],[],[]

    jointll=[headl,lhipl,lkneel,lwristl,pelvisl,rhipl,rkneel,rwristl]
    strjoint=["headl","lhipl","lkneel","lwristl","pelvisl","rhipl","rkneel","rwristl"]
    
    all_file = os.listdir(img_dir)
    for file in all_file:
        if file.endswith(".jpg"):
            
            skeleton(file)
            print("WORKING")
    #print(ranklelist,lanklelist)
    # Plotting the unique points/lengths
    for i in range(len(jointll)):
        jointll[i]=[x.real for x in fft(jointll[i])]
    for i in all_file:
        os.remove(i)
    print("Finished...!")
    
    return jointll


def compare_points():
    trainlist, testlist = [], []
    countlistlist = []  # stores sum of number of points which meet the threshold
    joint_files, train_folders = [], []
    # train_folders holds the text files in TrainData
    train_folders = os.listdir("TrainData")
    testlist = dummyposetest()  # list of lists
    
    for i in train_folders:
        # for each file(of individual joints) in all individuals
        joint_files = os.listdir("TrainData/" + str(i))
        countlist,errorlist= [],[]
        num = 0  # just a counter for testlist
        for j in joint_files:
            count, countlist = 0, []
            file = open("TrainData/" + str(i) + "/" + str(j), "r")
            trainlist = file.read().strip("[]")
            trainlist = list(map(float, trainlist.split(",")))
            # the order in testlist should be same as that of value of j
            value=0
            for value in range(min(len(trainlist), len(testlist[num]))):
                errorlist.append(abs(abs(trainlist[value]) -abs(testlist[num][value])))
            count=sum(errorlist)/(value)
            num += 1
            countlist.append(count)
            file.close()
        countlistlist.append(sum(countlist)/8)
    print(countlistlist)
    global resultfinal
    resultfinal=train_folders[countlistlist.index(min(countlistlist))]
    print(train_folders[countlistlist.index(min(countlistlist))])

def v2f():
    # Importing all necessary libraries
    import cv2

    # Read the video from specified path
    video=os.listdir("videos")
    cam = cv2.VideoCapture("videos/"+str(video[0]))
    fps = round(cam.get(cv2.CAP_PROP_FPS))
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # frame
    currentframe = 1

    while currentframe < frame_count:
        print(currentframe)
        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = 'images/' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)
            # increasing counter so that it will
            # show how many frames are created
        currentframe += 1


    # Release all space and windows once done
    print("Removing...")
    cam.release()
    cv2.destroyAllWindows()
    # removing the video from folder videos so that finally there is no video in the system.
    os.remove("videos/"+str(video[0]))
from tkinter import *
from mttkinter import *
import cv2
import os
inputValue=0
def opentraining():
    # taking the user to the videos folder so that he/she can drop their videos there
    # if not present create a new directory in the same folder as the program.
    os.system("start videos")
    # As long as a video is not given,do not start extracting frames from videos
    while os.listdir("videos")==[]:
        continue
    v2f()

def retrieve_input(textBox):
    global inputValue
    inputValue = textBox.get("1.0", "end-1c")
    # print(inputValue)


def newwindowtrain():
    top = Toplevel()
    top.title("INSERT VIDEO")
    top.geometry('250x250+550+200')
    top.config(bg="gold")
    
    # to end the display of insert video window and also open the folder where the video will be saved
    label = Label(top, bg="gold", fg="black", text="Please save your testing video in the folder\n that will be opened.\nPRESS SUBMIT after giving your NAME")
    button = Checkbutton(top, text="Continue", fg="black", font=("ARIEL", 9, "bold"), bg="gold", command=lambda: [f() for f in [top.destroy, opentraining, dummyposetrain]])
    textBox = Text(top, height=1, width=15)
    textBox.place(x=110, y=100)
    buttonCommit = Button(top, height=1, width=8, text="SUBMIT", bg="red", fg="white",command=lambda: retrieve_input(textBox))
    buttonCommit.place(x=115, y=120)
    label1 = Label(top, text="Name:", fg="black", bg="gold")
    label1.place(x=70, y=100)
    button.pack(side=BOTTOM)
    label.pack()
    
def newwindowtest():
    top = Toplevel()
    top.title("INSERT VIDEO")
    top.geometry('250x250+550+200')
    top.config(bg="gold")
    
    # to end the display of insert video window and also open the folder where the video will be saved
# command=lambda: retrieve_input() >>> just means do this when i press the button
    
    label = Label(top, bg="gold", fg="black", text="Please save your testing video in the folder\n that will be opened")
    button = Checkbutton(top, text="Continue", fg="black", font=("ARIEL", 9, "bold"), bg="gold", command=lambda: [f() for f in [top.destroy, opentraining]])
    button.pack(side=BOTTOM)
    label.pack()
    # cv2.waitKey(5000)
    # top.destroy()
def newwindow():
    top = Toplevel()
    top.title("RESULT")
    
    top.geometry('125x125+550+200')
    top.config(bg="gold")
    framelabel = Label(top, text="RESULT", bg="gold", fg="white", font=("ARIEL", 18, "bold"))
    # to end the display of insert video window and also open the folder where the video will be saved
# command=lambda: retrieve_input() >>> just means do this when i press the button
    label = Label(top, bg="gold", fg="black", text=str(resultfinal))
    framelabel.pack()
    label.pack()

    
root = mtTkinter.Tk()
root.geometry('500x500+500+175')
root.configure(background="white")
root.title("GAIT ANALYSIS")
root.resizable(False, False)
frame = Frame(root, width=600, height=750)
frame.config(bg="blue")
frame.pack()
framelabel = Label(frame, text="GAIT \n BIOMETRICS", bg="blue", fg="white", font=("ARIEL", 18, "bold"))

# c.pack()
button1 = Button(frame, text="INSERT TRAINING VIDEO", fg="black", font=("ARIEL", 9, "bold"), bg="cyan", command=newwindowtrain)
button2 = Button(frame, text="QUIT", fg="black", font=("Ariel", 9, "bold"), bg="cyan", command=root.quit)
button3 = Button(frame, text="COMPARE", fg="black", font=("Ariel", 9, "bold"), bg="cyan", command=compare_points)
button4 = Button(frame, text="INSERT TESTING VIDEO", fg="black", font=("Ariel", 9, "bold"), bg="cyan", command=newwindowtest)
button5 = Button(frame, text="RESULT", fg="black", font=("Ariel", 9, "bold"), bg="cyan", command=newwindow)
#framelabel.place(relheight=0.20, relwidth=0.3, x=0.33, y=0)
framelabel.place(x=165, y=0, relwidth=0.33, relheight=0.2)
button1.place(relheight=0.1, relwidth=0.332, relx=0.5, rely=0.4, anchor=CENTER)
button2.place(relheight=0.05, relwidth=0.332, relx=0.995, rely=1, anchor=SE)
button3.place(relheight=0.1, relwidth=0.332, relx=0.5, rely=0.6, anchor=CENTER)
button5.place(relheight=0.1, relwidth=0.332, relx=0.5, rely=0.7, anchor=CENTER)
button4.place(relheight=0.1, relwidth=0.332, relx=0.5, rely=0.5, anchor=CENTER)

root.mainloop()
