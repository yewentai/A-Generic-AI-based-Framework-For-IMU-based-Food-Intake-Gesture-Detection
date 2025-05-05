The dataset contains three sub-datasets: FD-I, FD-II, and Meal-Only (MO) dataset.

1. Dataset Description:

1.1. Sensor Info:
The off-the-shelf IMU sensor Shimmer3 is used in this study. The Shimmer3 IMU contains a 3-axis accelerometer unit, a 3-axis gyroscope unit, and a 3-axis magnetometer unit. The signals from the accelerometer and gyroscope were used in this study, hence 6 Degrees-of-Freedom (6 DoF). Two IMUs were attached to the left and right wrists of the participant. The sampling frequency is 64 Hz.


1.2. FD-I dataset:
The FD-I dataset contains 34 days of IMU data from 34 participants. On the data collection day, our research assistants met the participant, instructed the participant to wear IMU wristbands. Participants were free to engage in their normal daily activities. The daily activity of each participant was recorded by a camera for annotation. Experiment locations contained participant's home (apartments, student residence), restaurants, library, university learning center, and campus rest areas. Our research assistants were responsible for the recording when participants changed their locations to ensure that all eating and drinking gestures were captured. There were no limitations on the participants' activity during data collection. At least one meal was collected from each participant, and the minimum data collection duration was 6 h. Both eating alone and social eating scenarios were included in the dataset. A total of 251.70 h two-handed IMU data were collected, which contains 74 eating episodes, with 4,568 eating and 1,100 drinking gestures. The dataset contains four eating styles including forks & knives, chopsticks, spoon, and hands.


1.3. FD-II dataset:
The FD-II dataset contains 27 days of IMU data from 27 participants. The experiment protocol was the same as the FD-I dataset. However, in this dataset, only meal sessions were recorded by cameras (Some videos were collected using participants' own smartphones). All other eating/drinking gestures outside of meal sessions were not recorded. Therefore, the ground truth information only contains the bite information during meals and the meal boundaries. The FD-II dataset is considered as a hold-out dataset, which contains 52 meals with 2,723 eating gestures (including four eating styles) over a total duration of 261.68 h.


1.4. Meal-Only (MO) dataset:
The MO dataset contains 46 meal sessions from 46 participants, with a total of 2,894 eating and 763 drinking activities. It should be noted that part of this dataset was collected together with our previous Eat-Radar project and there is no overlap in participants between the MO and FD datasets. This dataset is a training-only dataset in the study Eating Speed Measurement Using Wrist-Worn IMU Sensors in Free-Living Environments (https://arxiv.org/abs/2401.05376).


1.5. Annotation:
-Eating gesture   (labelled as 1):
The movement from raising the left/right hand to the mouth with cutleries (fork & knife, chopsticks, spoon, and hand) until putting away the cutlery from the mouth is considered as an eating gesture.

-Drinking gesture (labelled as 2):
The movement from raising the left/right hand to the mouth with a container until putting away the container from the mouth is considered as a drinking gesture.

-Others           (labelled as 0):
The Others class contains all the other daily activities during the data collection.



2. Dataset file description:
2.1. Files under the FD-I folder:
        	-X_L.pkl (A list, the element of list is numpy array, each of the numpy array represents the 6 channel IMU data [ax,ay,az,gx,gy,gz] from left hand of one participant)
                          This pkl data includes participant's left hand IMU data.
		          The file contains N arary, N represents the number of participants. The size of each array is N_samples x N_channels (6), where N_samples = duration (sec) x 64.
			  
                -X_R.pkl (A list, the element of list is numpy array, each of the numpy array represents the 6 channel IMU data [ax,ay,az,gx,gy,gz] from right hand of one participant)
                          This pkl data includes participant's right hand IMU data.
		          The file contains N arary, N represents the number of participants. The size of each array is N_samples x N_channels (6), where N_samples = duration (sec) x 64.

        	-Y_L.pkl (A list, the element of list is numpy array, each of the numpy array represents the bite-level label (0, 1, 2) data for the left hand of one participant)
                          This pkl data includes participant's left hand bite-level label data.
		          The file contains N arary, N represents the number of participants. The size of each array is N_samples x 1, where N_samples = duration (sec) x 64.

                -Y_R.pkl (A list, the element of list is numpy array, each of the numpy array represents the bite-level label (0, 1, 2) data for the right hand of one participant)
                          This pkl data includes participant's right hand bite-level label data.
		          The file contains N arary, N represents the number of participants. The size of each array is N_samples x 1, where N_samples = duration (sec) x 64.


2.2. Files under the FD-II folder:
        	-X_L.pkl 			  
                -X_R.pkl 
        	-Y_L.pkl (It should be noted that only eating gestrues during meal sessions were labelled)
                -Y_R.pkl (It should be noted that only eating gestrues during meal sessions were labelled)


2.3. Files under the MO folder:
        	-X_L.pkl 			  
                -X_R.pkl 
        	-Y_L.pkl 
                -Y_R.pkl

***Note***: 
1. The sampling frequence of the IMU data in .pkl file is 64 Hz.
2. The left hand IMU data from X_L.pkl file is the raw data before hand-mirroring processing.
3. The annotation is point-wise annotation, the annotation sequence and the data sequence with same index has been aligned already.
4. If you have further questions, please concact: chunzhuo.wang@kuleuven.be

      


