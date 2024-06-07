
# License Plate Recognition










## Tech Stack

- Python
- HTML/CSS/Javascript
- SQLite3 Database
- Flask
- YOLOv7
- OpenCV, EasyOCR, Pytorch, Numpy






# To run my script:

Clone the yolov7 repository.
Place the detect-plates.py script and best.pt weights file in the yolov7 directory. 
Then run this command (all the commands below involve usage of CUDA, Nvidia GPU, to use CPU refer the documentation in the yolov7 github)

```bash
  python detect-plates.py --weights best.pt --conf 0.25 --img-size  640 --source path/to/image --device 0
```
For internal camera
```bash
  python detect-plates.py --weights best.pt --conf 0.25 --img-size  640 --source 0 --device 0
```

For external camera
```bash
  python detect-plates.py --weights best.pt --conf 0.25 --img-size  640 --source 1 --device 0
```








There are many other customizations in the flags, please go through yolov7's github repository for all of them.
## Link to Detection Model
[Google Drive Link](https://drive.google.com/file/d/10NCPSvGCaMqh_DXi5Bukf-6vvKICoMho/view?usp=sharing)

# Images and Videos of Testing
![License Plate Detection Img1](https://github.com/Pra-San/License-Plate-Recognition/blob/main/images/L1.jpg?raw=true)
![UI](https://github.com/Pra-San/License-Plate-Recognition/blob/main/images/UI%20with%20entry.png?raw=true)
![UI2](https://github.com/Pra-San/License-Plate-Recognition/assets/97389894/60f85182-2178-41ec-8c37-f5c3bcb73b94)


https://github.com/Pra-San/License-Plate-Recognition/assets/97389894/4da29fd1-4bde-49a4-96c8-42f9e776ca64

https://github.com/Pra-San/License-Plate-Recognition/assets/97389894/89b2bb39-b92e-4a81-9741-07769d8840e0


https://github.com/Pra-San/License-Plate-Recognition/assets/97389894/936ae59c-6c9a-490f-90ad-415b1f76a211


https://github.com/Pra-San/License-Plate-Recognition/assets/97389894/948855a1-fec8-46a7-ab8e-337769a90a53


## YOLOv7
YOLOv7 was made by [@WongKinYiu](https://github.com/WongKinYiu).

YOLOv7's Repository:
- [https://github.com/WongKinYiu/yolov7.git](https://github.com/WongKinYiu/yolov7.git)




