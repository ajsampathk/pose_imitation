## Getting Started

To get started with trt_pose, follow these steps.

### Step 1 - Install Dependencies

1. Install PyTorch and Torchvision.  To do this on NVIDIA Jetson, we recommend following [this guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-6-0-now-available)

2. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

    ```python
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo python3 setup.py install --plugins
    ```

3. Install other miscellaneous packages

    ```python
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib
    ```
### Step 2 - Install trt_pose

```python
git clone https://github.com/ajsampathk/pose_imitation.git
cd pose_imitation
sudo python3 setup.py install
```
### Step 3 - Setting up PyTorch models

```python
mkdir tasks/human_pose/models
cd tasks/human_pose/models
```
Download the resnet18 model:
| Model | Jetson Nano(FPS) | Jetson Xavier(FPS) | Weights |
|-------|-------------|---------------|---------|
| resnet18_baseline_att_224x224_A | 22 | 251 | [download (81MB)](https://drive.google.com/open?id=1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd) |

If you'd like to use densenet or another model and train it yourself, I suggest you take a look at the [official trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) repo from NVIDA

Let's optimize this model with TensorRT, run the ```optimize.py``` script by passing the above model weights

```python
python3 optimize.py resnet18_baseline_att_224x224_A_[EPOCH].pth
```
This should create an optimized model named ```trt_resnet18_baseline_att_224x224_A_[EPOCH].pth``` 

### Step 4 - Setting up ROS packages

To get the inference image and the inference data published on a ROS network, we need to set up a few things.

#### Step 4.1 - ROS Package Modules for python3
ROS does not have a great track record when it comes to python3 and especially CVBridge with Python3. So rather than creating a ROS environment with python3 as the default interpreter, we will add ROS support for python3 instead.
Install ```python3-rospkg-modules``` from apt:

```sudo apt-get install python3-rospkg-modules```

Now, drop into a python3 shell and ```import rospy```, if everything went okay, there will be no errors. If you get an import error add ```/opt/ros/[DISTRO]/lib/python2.7/dist-packages``` to your PYTHONPATH

#### Step 4.2 - TRT Bridge
Now that the ROS module problem has been solved we can build our ROS bridge for the inference image conversion from CV image to ROS image.
A ROS package is located at ```ros_trt/src/trt_bridge```, you could either build the package at ```ros_trt/``` by:

```python
cd ros_trt/
catkin_make
```
Or, you can copy the folder ```trt_bridge``` to your desired workspace and build it there.
Once it is built, we should have everything we needto run our inference engine.

**Note- ros-melodic-cv-bridge is a dependancy**

### Step 5 - Classification 
We have the joint positions from the trt inference engine we just set up, now the next task is t detect/classify different gestures. 

#### Step 5.1 - Raw Datset 

The dataset for this task is pretty straightforward, it consists of pictures with *ONLY ONE* person doing a gesture. 
Collect pictures of different gestures with a folder for each, the scripts will consider the folder names as labels. The structure should be similar to this:

    .
    ├── ...
    ├── dataset                         # root dataset folder [DATASET]
    │   ├── gesture_1                   # Label of the gesture
    |   |    ├── gesture_1_frame_0.jpg  # individual image files 
    |   |    ├── gesture_1_frame_1.jpg 
    |   |    └── ...
    │   ├── gesture_2        
    │   └── ...            
    └── ...
    
#### Step 5.2 - Generating the labelled data

To generate the required labelled data, we will pass all the images through the inference engine and get the required relevent informatio. There are two different approaches to this, one with the *(X,Y)* coordinates of each joint or, to pre-process tose coordinates to get the joint angles.

##### Step 5.2.1 - Coordinate dataset

The ``` generate_dataset.py``` will generate the coordinate labelled data, so we can run:

```python
python3 generate_dataset.py [DATASET FOLDER]
```

##### Step 5.2.2 - Processed dataset

The ```generate_dataset_processed.py``` will generate the processed angle data.

```python
python3 generate_dataset_processed.py [DATA FOLDER]
```

#### Step 5.3  - Training 

Once the dataset generation is done, it should produce two ```.json``` files, ```Labels_processed_[DATASET].json``` and ```LabelIndex[DATASET].json```.
We can now train the clasifer model with the generated files using the ```train.py``` script.

```python
python3 train.py Labels_processed_[DATASET].json LabelIndex[DATASET].json
```
You may want to change the number of epochs and batches based on the number of images you have in the dataset. You can take a look at the ```train.py``` script to modify these values.

#### Step 5.4 - Real-Time Classification

The training will generate a ```.pth``` file in the ```models/``` folder with the name ```classifier_net_[DATASET]_[M]x[N]_h1xh2_[M]x[N]_size_[SIZE].pth```.
Modify the name inside the ```classify.py``` script accordingly.

We can pass the LabelIndex to the classify script like this:

```pyhton3 classify.py LabelIndex_[DATASET].json```

### Step 6 - Executing all modules

Now, running everything step-by-step, make sure a camera is connected to the nano, the default camera input is video0. if you have multiple cameras connected, make sure the right one is selected in the ```inference.py``` script. **Note- Make sure you run inference and classify in the tasks/human_pose/ directory**

#### Step 6.1 - Inference Engine

```python
source ../../ros_trt/devel/setup.bash
python3 inference.py 
```
#### Step 6.2 - Classifier 
In a new terminal at ```[PATH]/tasks/human_pose/``` directory

```python
python3 classify.py LabelIndex_[DATASET].json
```
#### Step 6.3 - Bridge 
In another terminal.

```python
source [PATH-TO-REPOSITORY-ROOT]/ros_trt/devel/setup.bash
rosrun trt_bridge trt_bridge.py
```
You can then start ```rqt``` to visualize the ```/pose/image_raw``` topic to see the results.


### Results
#### Inference :
![](test_results.gif "ROS-TRTPOS:")

#### Classification:
![](gesture_results.gif "ROS-Gesture-Classification")
