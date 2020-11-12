FROM ajsampathkk/jetson-pytorch-ros:base

MAINTAINER Abhijith Sampathkrishna aj.sampathk@gmail.com


RUN apt-get -y update
RUN apt-get install -y lsb-release
RUN apt-get install -y software-properties-common

RUN apt-get install -y python3-pip

RUN pip3 install tqdm cython pycocotools
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-rospkg-modules

WORKDIR /home/

RUN git clone https://github.com/ajsampathk/pose_imitation.git
RUN apt-get update
RUN apt-get install -y ros-melodic-cv-bridge
ENV PYTHONPATH "${PYTHONPATH}:/opt/ros/melodic/lib/python2.7/dist-packages/"


COPY ros_entrypoint.sh /
ENTRYPOINT ["/bin/bash","ros_entrypoint.sh"]
CMD ["/bin/bash"]
