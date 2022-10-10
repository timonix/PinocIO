pip install -r requirements.txt

pip install numpy==1.21.3

# OpenCV install guide: https://www.youtube.com/watch?v=QzVYnG-WaM4&ab_channel=SamWestbyTech
# https://raspberrypi-guide.github.io/programming/install-opencv
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y
sudo apt install libgl1-mesa-glx
#sudo apt install python3-opencv
sudo apt-get install python-opencv

# Installing PyTorch
# Guide: https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531
# Wheel file: https://drive.google.com/file/d/1D3A5YSWiY-EnRWzWbzSqvj4YdY90wuXq/view
# pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f    https://download.pytorch.org/whl/torch_stable.html

sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
pip3 install /home/pi/Documents/torch-1.0.0a0+8322165-cp37-cp37m-linux_armv7l.whl

