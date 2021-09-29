# kNN-Classifier
**Frame-level six-class vowel classification experiment**

**
Problem Statement:**

Record each of the following six vowels (V1, V2, V3, V4, V5, V6)
V1 as in Hid
V2 as in Head
V3 as in Had
V4 as in Hudd
V5 as in Hod
V6 as in Hood
in a sustained manner for at least 1 second at 16kHz. Record each vowel separately for ten times. Trim 1second of vowel from each recording. First eight recordings of all vowels form the training set and remaining two from all vowels form the testset. Carry out a frame-level six-class vowel classification experiment using kNN classifier. The accuracy of the classification is computed as the percentage of frames correctly classified in the test set. For this purpose, consider a frame length of 20msec and a frame shift of 10msec. As a representative feature vector in each frame, experiment with three features: 1) 2-dimensional comprising first and second formants, 2) 13-dimensional mel frequency cepstral coefficients (MFCCs) , 3) 12-dimensional MFCCs excluding the 0-th coefficient.

Report the classification accuracy using all three features on the test set for K=1, 5, 10, 50, 100, 500, 1000 in the kNN classifier.

**Results:**
![image](https://user-images.githubusercontent.com/79351706/135350071-87cf56b1-74e5-4fbc-a7ce-d0c81737d4ad.png)
