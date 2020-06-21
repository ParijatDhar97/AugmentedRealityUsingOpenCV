# Augmented Reality Cam-Using-OpenCV-Python

-------------------OpenAR v1.0-----------------------

In this project I've tried to use Augmented Reality Technology to re-create a small task.

What it basically does?
It uses the target image given by the end-user and superimposes a video on it. Here we are imposing video, but we can imposing any 2d object on it. We will be using ORB detector to find the keypoints and descriptors for the image. Then we will be finding matches using knn BruteForce Matcher(we are finding the distance between the camera and the target image, and based on our favourable distance we are continuing our operation). This method provides us the good matches to determine if the target was detected or not. Then we are creating the right mask and augment our image and then video.


Application of AR is huge:

Medical Facilities : From operating MRI equipment to performing complex surgeries.
Retail : World famous motorcycle brand Harley Davidson is one great instance of a brand making the most of this trend, by developing an an AR app that shoppers can use in-store.
Design and Modelling : From interior design to architecture and construction, AR is helping professionals visualize their final products during the creative process. and many more...

Video describing this project:
https://drive.google.com/file/d/1c25qxV1WlVGS_Ta4IQ16ODCNz4YaVXkf/view?usp=sharing

Problems faced:
While creating the detector and the key points for the images, we faced some problems like the key points were able to get detected by the ORB detector for the target image but for the webcam images, it was failing to detect major key points. So we had to make some adjustments in the code and in the length function, after that it was able to detect the key points and its efficiency greater than 80% which is quite a good percentage to proceed with the project.  

Inspired from:

Abhishek Singh's Super Mario Bro's Recreated as Life size Augmented Reality Game : https://www.youtube.com/watch?v=QN95nNDtxjo
Microsift's Hololens : https://www.youtube.com/watch?v=6lxGU66w0NM
Google's AR Core : https://www.youtube.com/watch?v=VOVhCTb-1io

Thanks to Murtaza Hassan for his unbelievable efforts.

Connect with me:

LinkedIn : https://github.com/ParijatDhar97
Portfolio : https://parijatdhar97.github.io/iamparijatdhar/

