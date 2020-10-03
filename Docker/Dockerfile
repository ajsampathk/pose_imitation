FROM nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3

MAINTAINER Abhijith Sampathkrishna aj.sampathk@gmail.com


RUN apt-get -y update
RUN apt-get install -y lsb-release
RUN apt-get install -y software-properties-common
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update
RUN apt-get install -y ros-melodic-ros-base

