The dataset contains two sub-datasets: DX-I and DX-II.

1. Dataset Description:

1.1. Sensor Info:

The off-the-shelf IMU sensor Shimmer31 is used in this study. The Shimmer3 IMU contains a 3-axis accelerometer unit, a 3-axis gyroscope unit, and a 3-axis magnetometer unit. The signals from the accelerometer and gyroscope were used in this study, hence 6 Degrees-of-Freedom (6 DoF). Two IMUs were attached to the left and right wrists of the participant. The sampling frequency is 64 Hz.

1.2. DX-I dataset:

DX-I was collected in semi-controlled environments from 13 participants. The total duration of this dataset is 8.9 hours. The average duration of each participant is 35.6±28.6 min. In each drinking session, Water/tea/cola was provided to participants. A camera was used to record the entire drinking session as the ground truth annotation. They were asked to drink while sitting on chairs and a sofa, and also drink while standing. Both hands could be used to take the cups and drink, no matter whether the participant is a left-hander or a right-hander. The data were collected in the real work environment or home environment. They can work on a laptop, talk, walk, eat chips, watch TV or smoke. To collect more drinking gestures in this session, they were required to drink with a higher frequency. This dataset contains 410 drinking gestures (left hand : right hand : two hands = 101:266:43).

1.3. DX-II dataset:

DX-II was collected in free-living environments. The data were taken at the locations that are preferred by the participants, including work place and home. Seven participants took part in this experiment. Three out of the seven in DX-II also participated in DX-I. Each participant joined the experiment for 6.5±2.0 consecutive hours (from morning to afternoon or evening). The total duration of this dataset is 45.2 hours. The participants can drink water at their own pace. Each session contains daily life-related activities including, but not limited to, drinking water, eating snacks, eating lunch, working with laptops, watching smartphones and walking. A camera was placed on the desk in the office or at home to capture the drinking gesture. In total, there are 304 drinking gestures in DX-II dataset (left hand : right hand : two hands = 142:152:10).


1.4. Annotation:

-Drinking gesture (labelled as 1):

The movement from raising the left/right hand to the mouth with a container until putting away the container from the mouth is considered as a drinking gesture.

-Null (labelled as 0):

The Null class contains all the other daily activities during the experiment.


2. Dataset file description:

2.1. DX-I/DX-II 
        	-Raw data and Annotation
			-Left hand 
				subject_id - session_id . csv (Left hand IMU data, 64 Hz)
					Column name of .csv file : [t,ax,ay,az,gx,gy,gz,anno]
					
				
			-Right hand 
				subject_id - session_id . csv (Right hand IMU data, 64 Hz)
					Column name of .csv file : [t,ax,ay,az,gx,gy,gz,anno]
        	-Hand-Mirrored
			subject_id - session_id . csv (Right hand IMU data concatenate hand-mirrored Left hand data, 64 Hz)
					Column name of .csv file : [t,ax,ay,az,gx,gy,gz,anno]		


2.2. pkl data (ready to use data)
-DX_I_X.pkl   (A list, the element of list is numpy array, each of the numpy array represents the 6 channel IMU data of one session in DX-I.)
-DX_I_Y.pkl   (A list, the element of list is numpy array, each of the numpy array represents the label (0 or 1) data of one session in DX-I.)
-DX_II_X.pkl  (A list, the element of list is numpy array, each of the numpy array represents the 6 channel IMU data of one session in DX-II.)
-DX_II_Y.pkl  (A list, the element of list is numpy array, each of the numpy array represents the label (0 or 1) data of one session in DX-II.)
		
2.3. code [See https://github.com/Pituohai/drinking-gesture-dataset]
-DX-I data prepare.py (import data from seperate .csv file, load entire data to .pkl data)
-DX-I data visualization.py (import data from .pkl file, visualize 6 channel IMU signal and label)

*Note: 
1. Each of the .csv file has 8 columns. [t,ax,ay,az,gx,gy,gz,anno], [t] represents the timestamp, [ax,ay,az,gx,gy,gz] represents the 6 channels data, [anno] represents the label.
2. In pkl data, the timestamp is removed. 
3. Each of the .csv file in Hand-Mirrored folder represents the right IMU data + hand-mirrored left IMU data. Suppose for one session, right hand raw IMU data with size (10,6), left hand raw IMU data with size (12,6), the hand-mirrored .csv file has data with size (22,6), the first 10 represent right hand data, the last 12 represent hand-mirrored left hand data.
4. The code example can also be used to DX-II dataset.
5. 3 participants from DX-I also took part in DX-II. The overlap information is (ids in DX-I folds and DX-II folds):
     DX-I         DX-II
      1-3          4-1
      3-1          5-1
      7-1          6-1
      


