# dl4ds_fixed
Try to fix issues when installing DL4DS

docker run --gpus all -it --name dl4ds -e JUPYTER_ENABLE_LAB=yes -p 8888:8888 -v ${PWD}:/root/ -w /root/ wk1984/dl4ds2024