# R-pi installation steps
Steps to run the segmentation code on the R-pi model 3B. This is based on the Raspberry pi Lite OS. 

## Installation steps

Install the Raspberry Pi Lite OS. You can use the Raspberry pi imager tool for this.

### enable SSH
If you want to enable ssh access do the following:
1. add ssh access to the R-pi [link](https://phoenixnap.com/kb/enable-ssh-raspberry-pi). This involves just adding an empty file called "ssh" to the boot directory of the SD card
```
	cd /Volumes/boot/
	touch ssh
```
2. setup wifi on r-pi [link] (https://raspberrytips.com/raspberry-pi-wifi-setup/)

### Install  dependencies

1. Install python 3.7 [link](https://installvirtual.com/install-python-3-7-on-raspberry-pi/)
2. add & upgrade pip:
```
curl -O https://bootstrap.pypa.io/get-pip.py  
sudo python3.7 get-pip.py
pip3 install --upgrade pip
```

3. install pytorch
```
sudo apt-get update  
sudo apt-get upgrade  
sudo apt-get install libopenblas-dev libopenmpi-dev libomp-dev  
# above 58.3.0 you get version issues  
sudo -H pip3 install setuptools==58.3.0  
sudo -H pip3 install Cython  
sudo -H pip3 install 'https://wintics-opensource.s3.eu-west-3.amazonaws.com/torch-1.4.0a0%2B7963631-cp37-cp37m-linux_armv7l.whl'
```

4. Install other dependency libraries 
```
sudo -H pip3 install numpy
sudo -H pip3 install scipy
#install Raster IO
sudo apt-get install libatlas-base-dev
sudo -H pip3 install rasterio
```


