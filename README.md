# SafeCycling
SafeCycling is a device that assists cyclists by detecting dangerous approaches by vehicles in their surroundings.

The repo contains the code for interfacing with Labforge's monocular Bottlenose camera, including connection to the camera over Ethernet, uploading the neural network to be run on the camera, and pulling out the model's output (in the form of bounding boxes) for further processing. In future updates, work for other subsystems may also be added. 

## Requirements

### Hardware
- Labforge Bottlenose (monocular version)
- Ethernet cable
- Power supply for Bottlenose ([recommended option](https://www.mouser.ca/ProductDetail/XP-Power/VEL24US120-US-JA?qs=w%2Fv1CP2dgqrSc7r6Jvqsow%3D%3D))
- x86-based computer running Ubuntu 22.04

## Installation

**NOTE:** Instructions are for the computer. See [Labforge's docs](docs.labforge.com) for information about setting up the camera.

#TODO
- make sure to note how to set up ebus-sdk, Ubuntu version etc
## Usage


```bash
python stream.py ?ModelFile ?MAC ?CalibrationFile ?AIHasDepth
# ModelFile        - (optional) path to model file to upload to Bottlenose
# MAC              - (optional) mac address of Bottlenose to connect to or first one
# CalibrationFile  - (optional) path to calibration weights file to upload to camera
# AIHasDepth       - (optional) boolean True or False whether AI model outputs an estimate for depth or not
```
