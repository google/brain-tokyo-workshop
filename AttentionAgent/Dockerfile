#Download base image ubuntu 16.04
FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y python3 python3-pip xvfb python3-opengl fontconfig python3-dev python-opencv
RUN apt-get install -y build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev unzip libboost-all-dev
RUN apt-get install -y wget g++ make cmake libsdl2-dev git zlib1g-dev libbz2-dev libjpeg-dev libfluidsynth-dev libgme-dev libopenal-dev libmpg123-dev libsndfile1-dev libwildmidi-dev libgtk-3-dev timidity nasm tar chrpath

WORKDIR /opt/app
ADD . /opt/app
RUN pip3 install -r requirements.txt
