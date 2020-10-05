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

If you'd like to use densenet or use your own model, I suggest you take a look at the [official trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose) repo from NVIDA

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

Now, drop into a python3 shell and ```import rospy```, if everything went okay, there will be no errors. If you get an import error add ```/opt/ros/[DISTRO]/lib/python2.7/dist-packages``` tothe PYTHONPATH

#### Step 4.2 - TRT Bridge
Now that the ROS module problem has been solved we can build our ROS bridge for the inference image conversion from CV image to ROS image.




### Resluts

![](test_results.gif "ROS-TRTPOSE:")

![](gesture_results.gif "ROS-Gesture-Classification:")
