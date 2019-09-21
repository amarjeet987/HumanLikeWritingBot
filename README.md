# HumanLikeWritingBot
A robot that writes human like handwriting with the help of a deep learning model.

## Approach
### On observing from top level, the project consists of 2 parts :

### 1) The Model
<dl>
  <dd>A deep learning model that could generate human like handwriting from input given as text by the user. It outputs the handwriting as a group of coordinates for each stroke which can be plotted serially to get the whole writing. The model is based on Alex Graves's paper available <a href="https://arxiv.org/abs/1308.0850">here</a>. 
<h3>Files : <h3>
  <p>Files related to the model can be found in the <i>"Model"</i> folder.<br><br>
    <b>1)<i> data_preprocessor.ipynb </i></b><br>
    This file preprocesses the dataset in the <i>"data"</i> folder. The data that is used here is <a href="http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database/download-the-iam-on-line-handwriting-database">data/original-xml-part.tar.gz</a>, download the data, extract it and paste it in the <i>"data"</i> folder. This file would parse the data in form in which it could be passed through the model and does the required preprocessing, and finally saves the datasets.<br><br>
    <b>2)<i> data_generator.ipynb </i></b><br>
    It basically helps in creating batches and passing them to the model during the training process from the preprocessed data stored in <i>"data_parsed"</i> folder.<br><br>
    <b>3)<i> model.ipynb </i></b><br>
    It is the implementation of the model to be trained on the data we preprocessed. A pretrained ready to use model is added in the <i>"pretrained"</i> folder.<br><br>
    <b>4)<i> write.ipynb </i></b><br>
    It uses a trained model to generate human like handwriting of the text we feed it as input. A sample output is shown below :<br><br>
    <p align="center">
      <img width="540" height="250" src="https://user-images.githubusercontent.com/38986305/65377344-59d1c380-dcc8-11e9-9e2d-a269ad34d34f.gif">
    </p>
  </dd></dl>
    
### 2) The bot
<dl><dd>
  The bot is given a bunch of coordinates formatted in a <i>gcode</i> file and it traces those coordinates. It is operated by Arduino.
  <h3>Files : </h3>
    <b>1)<i> ML_robot.py </i></b><br>
    This file takes the text from user as input, spits out the corresponding handwriting output as a bunch of coordinates and makes sure that the arduino retrieves the coordinates. Scaling and shifting can also be done here before "plotting" the text in plain paper. A demonstration is shown below :<br><br>
  <b> Input : </b> Rain rain go away<br>
  <b> What the robot draws : </b><br><br>
  <p align="center">
      <img width="540" height="250" src="https://user-images.githubusercontent.com/38986305/65377515-c77eef00-dcca-11e9-8f0f-d27d77c5cdc7.gif">
    </p>
</dd></dl>
