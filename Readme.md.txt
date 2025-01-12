This dataset contains audio recordings used for training and testing a machine learning model to detect emergency vehicle sounds, specifically ambulances, and distinguish them from traffic noise sounds. The dataset is organized into two main parts: training and testing. Each part contains subfolders for ambulance and traffic noise sounds.

->train/ambulance/: This folder contains sound recordings of ambulance sirens. These audio files are used for training the machine learning model to recognize ambulance sounds.

->train/others/: This folder contains recordings of general traffic noises (e.g., cars passing by, honking, etc.). These files are used for training the model to recognize sounds that are not related to emergency vehicles.

->test/ambulance/: This folder contains sound recordings of ambulance sirens that are used to test the model's performance on previously unseen data.

->test/others/: This folder contains recordings of traffic noises used to test the model's ability to distinguish emergency vehicle sounds from non-emergency sounds.