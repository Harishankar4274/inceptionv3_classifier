FROM ubuntu:latest
MAINTAINER harishankardubey2911@gamil.com
ENV TZ=Indian
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
CMD echo 3 > /proc/sys/vm/drop_caches
RUN apt-get update -y
RUN apt-get install -y apt-utils
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install tensorflow
RUN pip3 install numpy
RUN pip3 install keras
RUN pip3 install opencv-python
RUN pip3 install pillow
RUN mkdir /dataset/
VOLUME /dataset/
CMD cd /dataset/
CMD python3 main.py
