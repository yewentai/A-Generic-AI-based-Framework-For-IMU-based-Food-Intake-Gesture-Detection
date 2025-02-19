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
		X_L.pkl: left hand raw IMU data
		X_R.pkl: right hand raw IMU data
		Y_L.pkl: left hand label
		Y_R.pkl: right hand label

*Note: 
1. In pkl data, the timestamp is removed.
2. 3 participants from DX-I also took part in DX-II. The overlap information is (ids in DX-I folds and DX-II folds):
     DX-I         DX-II
      1-3          4-1
      3-1          5-1
      7-1          6-1
      


