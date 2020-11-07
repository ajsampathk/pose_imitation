FROM ajsampathkk/jetson-pytorch-ros:base

MAINTAINER Abhijith Sampathkrishna aj.sampathk@gmail.com


RUN apt-get -y update
RUN apt-get install -y lsb-release
RUN apt-get install -y software-properties-common

RUN apt-get install -y python3-pip

RUN pip3 install tqdm cython pycocotools
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-rospkg-modules

COPY ros_entrypoint.sh /
ENTRYPOINT ["/bin/bash","ros_entrypoint.sh"]
