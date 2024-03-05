Based-on Computer Vision Applications for Bus Stop Passenger Detection System




![](./img.001.png)

Yu-Cheng Chang

*Department of Computer Information and Network Engineering,*

*Lunghwa University of Science and Technology*

Taiwan <yucheng208@outlook.com>

Hua-Wen Tsai

*Department of Computer Information and Network Engineering,*

*Lunghwa University of Science and Technology*

Taiwan <hwtsai@mail.lhu.edu.tw>

Chao-Yi Huang

*Department of Bachelor's Program in Intelligent Robot,*

*National Yunlin University of Science and Technology*

Taiwan <joyatnet0@gmail.com>


Zong-Rong Wu

*Department of Technical and Vocational Program in Artificial Intelligence, National Yunlin University of Science and Technology*

Taiwan <jason.jiong.long@gmail.com>


***Abstract—*Passenger transport is one of the most common ways of commuting in Taiwan. It plays an important role in the transportation system due to its large number of stations, dense frequency, and cheap transportation. Due to the unfriendly transportation environment and a large number of passengers, a blind spot of passenger transportation exists, which leads to traffic accidents at the station. We research to make the "Bus Stop Passenger Detection System". Taking the object detection of "Wheelchairs" into consideration, it is more convenient to assist the disabled to find the passenger transportation system, which makes Taiwan's transportation system more convenient.**

***Keywords—Pose Detection, Object Detection, Transportation System, Wheelchairs***

1. Introduction

Buses are one of the main modes of transportation in Taiwan's public transportation system due to their low cost, frequent schedules, and dense network of bus stops. However, issues such as aging bus drivers, poor road conditions, and overcrowded buses have led to poor visibility for drivers and an inability to detect passengers waiting at bus stops. This results in passengers being unintentionally passed by as well as posing a potential risk of traffic accidents.

Therefore, many scholars have focused on researching advanced driver assistance systems (ADAS) to achieve a more sophisticated transportation infrastructure. According to Ref. [1], today's ADAS-equipped vehicles use image recognition and sensors to assist drivers with tasks such as lane departure warning, lane change detection, and collision detection. Vehicles integrate LIDAR sensors for more precise assistance. However, the application of these technologies in public transportation systems is still rare. Therefore, we design a more suitable assistance system for public transportation from this perspective.

When designing an assistance system suitable for public transportation, the driving environment and the driver's condition must be taken into consideration. Therefore, we found a solution that uses an artificial neural network to build a detection model for human behavior [2]. We extend the detection object to passengers in the bus station to assist in the transportation status of buses.

We have proposed a Based-on Computer Vision Application for Bus Stop Passenger Detection System. This system not only assists drivers in detecting the posture of passengers waiting at the bus stop but also detects whether passengers require mobility aids such as wheelchairs, and provides reminders to drivers through voice functions. In this article, we explain the object detection function of the system, the pose detection and definition, and describe the hardware control scheme in Section II. In Section III, we summarize the content of this study.

1. Computer Vision Application for Bus Stop Passenger Detection System

Figure 1 shows the proposed system with three major parts: object detection, pose detection, and hardware control. We use Raspberry Pi's Bluetooth detection and pairing function to confirm the distance between the bus and the bus stop. When Bluetooth pairing is successful and the distance is confirmed, the system calls the OpenCV function to capture the image in front of the bus. Then, the image is transmitted to the object detection module, and various objects are labeled. Subsequently, the image is transmitted to the pose detection module which uses joint labeling and coordinates for deep learning to determine whether the passenger's posture meets the defined standard. Finally, the system integrates the results of object detection and pose detection for judgment and output via hardware.

![](./img/002.jpeg)

Fig. 1. Hardware components of computer vision application bus stop passenger detection system.


![](./img/003.png)***979-8-3503-3386-2/23/$31.00 ©2023 IEEE*	152**

*2023 IEEE 3rd International Conference on Electronic Communications, Internet of Things and Big Data*


**


![](./img/004.png)

1. *Object Detection*

In the system architecture, we use YOLO as the object detector to detect passengers and wheelchairs [3]. Before using the object detection model, we obtained basic training, testing, and validation set data through field shooting and assistance from the Kaggle image dataset blog [4]. To improve the system and model, we also used web crawling techniques to collect and select images of people and wheelchairs from the Internet [5,6]. Furthermore, we classified highly similar vehicles such as bicycles and skateboards into the non-wheelchair category, and conducted multiple experiments to solve the overfitting problem. We found that setting the number of epochs to 100 during training was the most appropriate, and the convergence effect became optimal.

During the experiment, we conducted multiple tests simulating single and multiple scenarios for both regular and wheelchair passengers. From the output results of YOLO (Fig. 2), we found that both the hand-raising and phone-using actions of regular passengers could be accurately detected and masked.

![](./img/005.png)![](./img/006.png)

Fig. 2. YOLO for the scenario of regular single passenger standing and raising hand.

Figure 3 showed that wheelchair passengers could be recognized as humans by the detector, regardless of whether they raised their hands or not. In image recognition, simpler environments improve recognition accuracy. However, to verify whether multiple persons or objects nearby would lead to non-wheelchair detection, we conducted relevant experiments as presented in Fig. 4, which showed that both multiple regular passengers and a mix of regular and wheelchair passengers successfully performed object detection with good performance.

![](./img/007.jpeg)![](./img/008.jpeg)

Fig. 3. Scenario for wheelchaired passenger raising a hand or sitting.

![](./img/009.jpeg)![](./img/010.jpeg)

Fig. 4. Scenario of multiple passengers in a wheelchair and standing.

1. *Pose Detection*

For pose detection, we used OpenPose as the posture detection system [7−9]. After completing the object detection with YOLO, we proceeded with posture detection. During posture detection, we defined three postures: sitting, standing, and raising hand. By training the Artificial Neural Network (ANN) with the COCO 16 key points in the OpenPose model, we completed the training of the ANN model [10]. Next, we integrated the trained ANN model with OpenPose and used the neural network to fully automate the detection results, eliminating the need for developers to perform coordinate calculations and judgments.

In Fig. 5, we input the image detected by YOLO into OpenPose. We detected if the object was in a wheelchair, then determined if it was in a seated position. Next, we used the code in Fig. 6 to integrate the results and enable the system to differentiate between regular passengers waiting at the bus stop and wheelchair users who require assistance. Additionally, regular passengers waited in a standing position, so if a person was detected by YOLO, and OpenPose detected a raised hand, a notification was sent to the bus driver to stop at the bus stop as shown in Figs 7 and 8.

![](./img/011.png)![](./img/012.png)

Fig 5. YOLO and OpenPose for regular single passenger standing and raising a hand.

![](./img/013.jpeg)

Fig. 6. Code snippet: Integration and decision-making using YOLO and OpenPose.

![](./img/014.jpeg)![](./img/015.jpeg)

Fig 7. YOLO and OpenPose for a wheelchair user raising a hand or sitting.

![](./img/016.jpeg)![](./img/017.jpeg)

Fig 8. YOLO for multiple passengers in a wheelchair and non-wheelchair.

1. *Hardware Controlled*

Raspberry Pi was chosen as the central processor mainly due to its characteristics and features. Apart from the built-in Bluetooth and Wi-Fi capabilities, it has independent artificial

*2023 IEEE 3rd International Conference on Electronic Communications, Internet of Things and Big Data*


intelligence and logical computing abilities, as well as powerful and versatile I/O ports. Its potential for future expansion is limitless (Fig. 9).

![](./img/018.jpeg)

Fig. 9. Hardware architecture diagram of the proposed system.

After describing the development board, we installed YOLO and OpenPose on the Raspberry Pi for detection, and the results were processed by the hardware. We provided real-time video to the bus driver on a screen and audio in the sound system. The text-to-speech package was "pyttsx3" as shown in Fig. 10. This package directly calls the system sound effects for speech synthesis, so it is not affected by cross-system platform issues and can be used universally.

![](./img/019.jpeg)

Fig. 10. Code snippet: Text-to-Speech (TTS) synthesis.

1. Conclusions

Buses are important in Taiwan's public transportation system. Due to aging drivers, poor road conditions, and overloading, drivers' visibility can be affected, and passengers may accidentally miss their stops. Therefore, we propose a "Computer Vision Application for Bus Stop Passenger detection system" that detects objects and human postures and provides visual and audio prompts to bus drivers to reduce incidents of passengers missing their stops. Adding more audio-visual effects to remind passengers is planned for the future of the Computer Vision Application for Bus Stop Passenger Detection System, making the system more complete and encompassing the detection of passenger status inside the vehicle. This will contribute to the improvement of Taiwan's public transportation system.

References

1. S. Raviteja, R Shanmughasundaram, “Advanced Driver Assitance System (ADAS),” *2018 Second International Conference on Intelligent Computing and Control Systems (ICICCS)*, India, June 2018.
1. Yingzi Lin, P. Tang, Wenjun Zhang, “Artificial neural network modeling of driver handling behavior in a driver-vehicle- environment system,” *International Journal of Vehicle Design*, January 2005.
1. Norhan Bayomi, Mohanned El Kholy, John E. Fernandez, Senem Velipasalar, Tarek Rakha, “Building Envelope Object Detection Using YOLO Models,” *2022 Annual Modeling and Simulation Conference (ANNSIM)*, San Diego, CA, USA, July 2022.
1. D. Sculley, Jeff Moser, William Cukierski, Jerad Rose, Meghan O’Connell, Myles O’Neill, et al., “Kaggle”, [https://www.kaggle.com](http://www.kaggle.com/)/, 2010.
1. Ph.D Chang, Chih-Yung & Po-Wei Su, (2019) “Smart Web Generator based on Artificial Intelligence and Web Crawler Techniques”, *Master's thesis, Tamkang University*, Taiwan, 2019/06/14.
1. Wes McKinney, “Python for Data Analysis, 2nd Edition”, *O'Reilly Media, Inc*, October 2017.
1. Zao Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, Yaser Shheikh “OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol 43, issue 1, pp. 172-186, January 2021.
1. Z. Cao, T. Simon, S. Wei et al., "Realtime Multi-person 2D Pose Estimation Using Part Affinity Fields," *2017 IEEE Conference on Computer Vision and Pattern Recognition(CVPR)*, Honolulu, HI, pp. 1302-1310, doi:10.1109/CVPR.2017.143, 2017.
1. Lawrence Y. Deng, Xiang Yann Lim, See Sang Ho, “Developing A Parser Framework based on OpenPose Skeleton Detection,” *2021 IEEE 3rd Eurasia Conference on Biomedical Engineering, Healthcare and Sustainability (ECBIOS)*, Tainan, Taiwan, 2021/05/28.
