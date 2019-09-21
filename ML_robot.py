#Read the readme first
"""Software Requirments: Python(3.6)
                         Arduino IDE
         Python Library: RegEx
                         PyCmdMessebger
   Hardware requirements: "see readme"
pycmdMessenger library is used to send values to Arduino over the serial port(USB).
RegEx library is used to read coordinates from the gcode file."""

# Imports for arduino
import re
import PyCmdMessenger
import math
import time

# Imports for the model
import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from matplotlib import animation
import seaborn
from collections import namedtuple

# load the model
model = os.path.join('Model/pretrained/models', 'model-29')
style = None
bias = 1
force = False
animation = True
noinfo = True
save = None

def sample(e, u1, u2, std1, std2, rho):
    conv = np.array([[std1 * std1, std1 * std2 * rho],
                     [std1 * std2 * rho, std2 * std2]])
    mean = np.array([u1, u2])
    x, y = np.random.multivariate_normal(mean, conv)
    end = np.random.binomial(1, e)
    return np.array([x, y, end])

def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)

def split_strokes(stroke_pts):
    stroke_pts = np.array(stroke_pts)
    strokes = []
    b = 0
    for i in range(len(stroke_pts)):
        if stroke_pts[i, 2] == 1:
            strokes.append(stroke_pts[b:i+1, :2])
            b = i + 1
    return strokes

def text_sampling(sess, args_text, translation, style = None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    fields_data = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]
    
    print_len, style_len = 0, 0
    
    # one hot seq
    sequence = np.eye(len(translation), dtype=np.float32)[text]
    # add last column
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)
    
    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(fields_data.zero_states)
    
    for s in range(60 * len(text) + 1):
        e, pi, u1, u2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([fields_data.e, fields_data.pi, fields_data.mu1, fields_data.mu2,
                                               fields_data.std1, fields_data.std2, fields_data.rho, fields_data.finish,
                                               fields_data.phi, fields_data.window, fields_data.kappa],
                                              feed_dict={
                                                  fields_data.coordinates: coord[None, None, ...],
                                                  fields_data.sequence: sequence,
                                                  fields_data.bias: bias
                                              })
        phi_data += [phi[0, :]]
        window_data += [window[0, :]]
        kappa_data += [kappa[0, :]]
        # ---
        g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
        coord = sample(e[0, 0], u1[0, g], u2[0, g],
                       std1[0, g], std2[0, g], rho[0, g])
        coords += [coord]
        stroke_data += [[u1[0, g], u2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

        if not force and finish[0, 0] > 0.8:
            break
    
    coords = np.array(coords)
    # end of sentence
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords

# load the translation file
with open(os.path.join('data_parsed', 'translation.pkl'), 'rb') as file:
    translation = pickle.load(file)

# dictionary that maps the numbers to the letters
rev_translation = {v: k for k, v in translation.items()}

# create a set of all the appeared characters
charset = [rev_translation[i] for i in range(len(rev_translation))]
charset[0] = ''

# GPU
config = tf.ConfigProto(
    device_count={'GPU': 0}
)

# tf session
with tf.Session(config=config) as sess:
    # import model
    saver = tf.train.import_meta_graph(model + '.meta')
    saver.restore(sess, model)

    args_text = input('What to write: ')

    # sample the text, get the results
    phi_data, window_data, kappa_data, stroke_data, coords = text_sampling(sess, args_text, translation, style)

    # get the stroke data
    strokes = np.array(stroke_data)
    epsilon = 1e-8

    # to get the sequence of coordinates, as we trained the model on differences between points
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
    minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
    miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

    x = []
    y = []
    z = []
    fig, ax = plt.subplots(1, 1)
    for stroke in split_strokes(cumsum(np.array(coords))):
        for i in range(len(stroke[:, 0])):
            # put them inside lists for later
            if i == 0:
                x.append(stroke[i, 0])
                y.append(stroke[i, 1])
                z.append(1)
            else:
                x.append(stroke[i, 0]) 
                y.append(stroke[i, 1])
                z.append(5)
            
                
        # negative Y otherwise the output comes flipped
        plt.plot(stroke[:, 0], -stroke[:, 1])
    ax.set_title('Handwriting')
    ax.set_aspect('equal')
    plt.show()
    
        
# ------------------------------------------------------ MODEL DONE -----------------------------------------------------------
# coordinates saved in list x and y.
print("As per the equation\nfinal_val = (initial_val + shift)*scale")
shift_x = input("Shift x by : ")
scale_x = input("Scale x by : ")
shift_y = input("Shift y by : ")
scale_y = input("Scale y by : ")
file_name = input("Enter finename of .gcode file (default is coords.gcode) : ")
if file_name == "":
	file_name = "coords.gcode"
else:
	try:
		if file_name.split(".")[1] == "gcode":
			pass
	except IndexError:
		file_name += ".gcode"
x_new, y_new = 0, 0
with open(file_name, 'w') as gcode:
    print("Saving details.....")
    for i in range(len(x)):
        x_new = (x[i] + float(shift_x))*float(scale_x)
        y_new = (y[i] + float(shift_y))*float(scale_y)
        z_new = z[i]
        gcode.write('X' + str(x_new) + ' Y' + str(y_new) + ' Z' + str(z_new) + 's\n')
gcode.close()
print("File saved.")

# -------------------------------------------------------drawing starts here----------------------------------------------
# Initialize an ArduinoBoard instance.  This is where you specify the baud rate and
# serial port.  If you are using a non ATmega328 board, you might also need
# to set the data sizes (bytes for integers, longs, floats, and doubles).  
arduino = PyCmdMessenger.ArduinoBoard("COM9",baud_rate=115200)#28800

# List of commands and their associated argument formats. These must be in the
# same order as in the sketch.
commands = [["motor1","f"],
            ["motor2","f"],
            ["servoo","f"],
	    ["motor1_value_is","f"],
            ["motor2_value_is","f"],
            ["servo_value_is","f"]]

# Initialize the messenger
c = PyCmdMessenger.CmdMessenger(arduino,commands)





l1=80#float(input("Enter the value of link 1 (mm): "))
l2=120#float(input("Enter the value of link 2 (mm): "))
l3=120#float(input("Enter the value of link 3 (mm): "))
l4=80#float(input("Enter the value of link 4 (mm): "))
l5=58#float(input("Enter the value of link 5 (mm): "))
firstloop=0
th_prev1=90
th_prev2=0
xerr=0#-20#DISTANCE from origin
yerr=80
inc=0.05
dist=0
start = time. time()
pen_up=80.0
pen_down=110.0
utha=0
pen_lift_threshold=3.5#mm
with open("coords.gcode") as gcode:
     for line in gcode:
        line = line.strip()
        #coord = re.findall(r'[XY].?\d+.\d+', line)
        coordx = re.findall(r'X\d+.\d+', line)
        coordy = re.findall(r'Y\d+.\d+', line)
        #coordz = re.findall(r'Z\d+.\d+', line)
        coordz = line.split("Z")[1]
      
        #print(coordz)
        #print(coordx)
        #print(coordy)
        
        if coordz:
           z = coordz[0]
           #ze = float(z[1:])
           #ze=float(z)
           ze=int(z)
           #print(z)
           if (ze==1):#safe clearance signal
               '''c.send("servoo",pen_up)
               msg = c.receive()
               #print(msg)
               time.sleep(0.3)'''
               utha=1
           if(ze==2 and utha==1):
                '''c.send("servoo",pen_down)
                msg=c.receive()
                #print(msg)
                time.sleep(0.3)'''
                utha=0
              
        if coordx and coordy:
            
            firstloop=firstloop+1
            #print("{}-{}".format(coord[0],coord[1]))
            x = coordx[0]
            xe = float(x[1:])
            xe=xe+xerr
            #print(xe)
            y = coordy[0]
            ye = float(y[1:])
            ye=ye+yerr
            #print(ye)
            if (firstloop==1):
                c.send("servoo",pen_up)
                msg = c.receive()
                xc=xe
                yc=ye
                xprev=xc  #inc added to avoid division by zero in next while loop
                yprev=yc
                A1=xc
                B1=yc
                C1=(l1*l1 - l2*l2 + xc*xc +yc*yc)/(2*l1)

                A2=xc-l5
                B2=yc
                C2=(l4*l4 +l5*l5 -l3*l3-2*xc*l5+xc*xc+yc*yc)/(2*l4)
    
                check1=(A1*A1 + B1*B1 -C1*C1)
                check2=(A2*A2 + B2*B2 -C2*C2)
                y1b=B1
                if(check1<0 or check2<0):
                    print("The point is beyond the workspace")
        
                else:
                    y1a=math.sqrt(A1*A1 + B1*B1 -C1*C1)
                #th11=2*math.atan((-y1b+y1a)/(-A1-C1))
                    th12=2*math.atan((-y1b-y1a)/(-A1-C1))
                    th12=(180/(22/7))*th12
        
	
                    y2a=math.sqrt(A2*A2 + B2*B2 -C2*C2)
                    y2b=B2
	
                    th22=2*math.atan((-y2b+y2a)/(-A2-C2))
            #th22=2*math.atan((-y2b-y2a)/(-A2-C2))
                    th22=(180/(22/7))*th22
        #print("th11={0}" .format(th11))
                    #print("th12={0}" .format(th12))
        #print("th21={0}" .format(th21))
                    #print("th22={0}" .format(th22))
                    if(th12<0):
                        th12=360+th12
        #if(th22<0):
         #   th22=360+th22



    # Send
                    th12_new=th12-th_prev1
                    th22_new=th22-th_prev2



                    steps1=th12_new*8.888888889
                    steps1=round(steps1)
                    if(steps1!=0):
                        c.send("motor1",steps1)
                        msg = c.receive()
                        #print(msg)
                    th12_new=steps1*0.1125
                    th_prev1=th_prev1+th12_new



                    steps2=th22_new*8.888888889
                    steps2=round(steps2)
                    if(steps2!=0):
                        c.send("motor2",steps2)
                        msg = c.receive()
                        #print(msg)
                    th22_new=steps2*0.1125
                    th_prev2=th_prev2+th22_new
                    c.send("servoo",pen_down)
                    msg = c.receive()
                    print("Drawing...")
                    




          
             
            dist=math.sqrt((xprev-xe)*(xprev-xe)+(yprev-ye)*(yprev-ye))
            if(utha==1 and abs(xprev - xe)>=pen_lift_threshold):
               c.send("servoo",pen_up)
               msg = c.receive()
               print(msg)
               time.sleep(0.1)
               print(dist)
               
            
                
            if(dist>inc):
            
                while(dist>=inc and firstloop>1):
                    #start = time. time()
                    
                    delx=xprev-xe   #finding the travel is more on which axis
                    if(delx<0):
                       delx=-delx
                    dely=yprev-ye
                    if(dely<0):
                       dely=-dely

                    if(delx>=dely): #if travel more in x then it is better to have the straight line eqn in x to avoid infinite slope condition
                   
                       if (xe>xprev):
                          xc=xc+inc
                       else:
                          xc=xc-inc
                       yc=((yprev-ye)/(xprev-xe))*(xc-xprev)+yprev   
                    else:
                   
                       if (ye>yprev):
                          yc=yprev++inc
                       else:
                          yc=yprev-inc
                       xc=((xprev-xe)/(yprev-ye))*(yc-yprev)+xprev
                    dist=math.sqrt((xc-xe)*(xc-xe)+(yc-ye)*(yc-ye))
                    xprev=xc
                    yprev=yc

                    A1=xc
                    B1=yc
                    C1=(l1*l1 - l2*l2 + xc*xc +yc*yc)/(2*l1)

                    A2=xc-l5
                    B2=yc
                    C2=(l4*l4 +l5*l5 -l3*l3-2*xc*l5+xc*xc+yc*yc)/(2*l4)
    
                    check1=(A1*A1 + B1*B1 -C1*C1)
                    check2=(A2*A2 + B2*B2 -C2*C2)
                    y1b=B1
                    if(check1<0 or check2<0):
                        print("The point is beyond the workspace")
        
                    else:
                        y1a=math.sqrt(A1*A1 + B1*B1 -C1*C1)
                #th11=2*math.atan((-y1b+y1a)/(-A1-C1))
                        th12=2*math.atan((-y1b-y1a)/(-A1-C1))
                        th12=(180/(22/7))*th12
        
	
                        y2a=math.sqrt(A2*A2 + B2*B2 -C2*C2)
                        y2b=B2
	
                        th22=2*math.atan((-y2b+y2a)/(-A2-C2))
            #th22=2*math.atan((-y2b-y2a)/(-A2-C2))
                        th22=(180/(22/7))*th22
        #print("th11={0}" .format(th11))
                        #print("th12={0}" .format(th12))
        #print("th21={0}" .format(th21))
                        #print("th22={0}" .format(th22))
                        if(th12<0):
                            th12=360+th12
        #if(th22<0):
         #   th22=360+th22



    # Send
                        th12_new=th12-th_prev1
                        th22_new=th22-th_prev2



                        steps1=th12_new*8.888888889
                        steps1=round(steps1)
                        if(steps1!=0):
                            c.send("motor1",steps1)
                            msg = c.receive()
                            #print(msg)
                        th12_new=steps1*0.1125
                        th_prev1=th_prev1+th12_new

                        #time.sleep(0.002)

                        steps2=th22_new*8.888888889
                        steps2=round(steps2)
                        if(steps2!=0):
                            c.send("motor2",steps2)
                            msg = c.receive()
                            #print(msg)
                        th22_new=steps2*0.1125
                        th_prev2=th_prev2+th22_new

        
            
                        #time.sleep(0.002)
                        #end = time. time()
                        #print(end - start) #around 0.003
                xprev=xe
                yprev=ye
            if(utha==1):
               c.send("servoo",pen_down)
               msg = c.receive()
               time.sleep(0)
               utha=0
print("Drawing Finished")
c.send("servoo",pen_up)
msg = c.receive()
end = time. time()
print("Total Time(min): ",((end - start)/60))
