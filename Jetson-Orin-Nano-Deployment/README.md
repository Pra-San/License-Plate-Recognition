
# Deployment on Jetson Orin Nano

The following steps work for any yolov7 model, with a few version and dependency tweaks to the setup file, run command and detect.py file. The model runs on the GPU successfully after following the steps in this readme.









## What is Jetson Orin Nano?


Jetson Orin Nano is an edge device launched by Nvidia. It is specifically designed with the intent of model inferencing. It has a powerful graphics card, which is integrated into its ARM architecture. The 8GB device was used for this project.

## Project Specifics
Since Jetson nanos have lots of different versions and each version has its own wheel for packages like torch. Different cuda versions also pose problems when running the code. 

The torch and torchvision setup process should work with any cuda version, so do not worry if your CUDA version is different.
### Versions used:

- Jetpack 5.1.1 
- PyTorch 1.11
- torchvision v0.12.0
- cuda 11.4 

Rest of the dependencies can be found in `requirements_modified.txt`.











# To run my script:

Clone this repository.
Then run this command from within this folder

```bash
  chmod +x setup.sh
  ./setup.sh
```

For the camera, change the source number according to your system. To know which one to use, run this command 
```bash
ls /dev/video*
```
Then change the source number in the following command accordingly. For me it was 4.
```bash
  python detect-plates.py --weights best.pt --conf 0.25 --img-size  640 --source 4 --device 0
```

There are many other customizations in the flags, please go through yolov7's github repository for all of them.
Next step is to download the detection model. The detection model is too big for github due to which I shared the google drive link to download it.

### To test if torch is using cuda and if installation of torchvision and cv2 was done correctly:

Terminal:
```bash
python3
```
python3: 
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import torchvision
print(torchvision.__version__)

import cv2
print(cv2.__version__)
```
## Link to Detection Model
[Google Drive Link](https://drive.google.com/file/d/10NCPSvGCaMqh_DXi5Bukf-6vvKICoMho/view?usp=sharing)



## Important Notes

#### Some commands in opencv do not work. You have to comment/remove them if you modify the script or get errors while running. 

The following do not work unless opencv is installed differently (built from source):
| Functions | Originally used in               |
| :-------- | :------------------------- |
| `cv2.imshow` | detect_plates_single_gate.py , detect-plates.py  |
| `cv2.waitkey()` | datasets.py |

Any module not found errors, just `pip3 install` the module.

The only advice that I can give you if you run into any errors after this is , do not trust chatgpt and instead just google the problem. If the problem becomes too complex then its time to flash (resinstall the OS) the storage device.


## References


[Downloading pytorch and torchvision](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)

