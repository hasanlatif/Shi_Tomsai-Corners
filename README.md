# Shi Tomsai Corners (Better than Harris Corners)
* System uses canny edge detection algorithm to get edges on test and ground-truth imaeg .After getting the edges, Just use the bitwise xor operation to get the defects in pcb.
* Canny Edge Detector firstly applies gaussian filter to remove any unwanted noise.Then it finds the intensity gradients to get edges.
* XOR Truth Table

![](https://github.com/hasanlatif/Snapchat-like-Filters-python/blob/master/Readme_pics/shi-tomsai-corners.PNG)

# Results
![](https://github.com/hasanlatif/Snapchat-like-Filters-python/blob/master/Readme_pics/Result.png)



# Limitations: 
* This  system is not ment to use for Industrial Puprose.
# Note:
  * Waiting for your suggestions.If you find any lag in documentation in any ways,shoot me an email at hasanlateef@outlook.com
  * This code requires scikit-image,numpy and matplotlib in order to give the output
  * Implementation of Machine Learning Algorithms from scratch is in pipline.
  * This research is based in part upon work supported by the Office  EZ Technologies. The views and conclusions contained herein are    those of the authors and should not be interpreted as necessarily representing the official policies or endorsements.




