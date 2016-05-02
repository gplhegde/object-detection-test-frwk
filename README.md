# Object detection test framework
This is a python based framework for testing performance and accuracy of generic object detection model on various standard and custom
datasets using OpenCV. It accepts models in the form of XML files which abide to OpenCV format. It also provides tools to convert the
floating point models into fixed point with certain precision and analyse the performance.

# Packages
1. *lib/dataset*: 
This provides infra to add new dataset for evaluation. Each newly added dataset must define a class with some mandatory
methods to read the object annotation files which are specific to that particular dataset. Once these mandatory methods are defined in the
dataset specific class, it sholud be registered in lib/dataset/datasets.py. This allows the object evaluator to recognize the new daraset.

2. *lib/obj_detector*: 
This is where the object detector is implemented. Generally a cascade classifier is implemented. As of now, LBP
cascade classifier is implemented. Any new detection algorithm can be added to this package and made available to the object evaluator.

3. *lib/obj_evaluator*: 
This evaluates the specific dataset registerd as stated above using the object detector and the ground thuth
annotations of a particular dataset. The default object detector used is OpenCV multiscale cascade detector. However, any new detector
defined in obj_detector package can be used in this evaluator. It has the evaluation criteria and some configurations defined in its own
configuration file. The main evaluator module in this package reports the performance and accuracy of detection using the specified accuracy
metrics.
