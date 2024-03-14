sudo apt-get -y update && sudo apt-get -y upgrade
sudo apt-get -y install ubuntu-drivers
sudo ubuntu-drivers install --gpgpu
sudo apt-get -y install build-essential
sudo apt-get -y install nvidia-utils-535-server
sudo apt -y autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get -y update
sudo apt-get -y install cuda-toolkit-12-4
