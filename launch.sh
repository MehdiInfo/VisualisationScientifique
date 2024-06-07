#/bin/bash

xhost +local:$USER
docker run \
  --rm \
  -it \
  --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/:/dev/ \
  -v `pwd`:/visualisation-scientifique \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  romeogit.univ-reims.fr:5050/mnoizet/visualisationscientifique \
  bash
