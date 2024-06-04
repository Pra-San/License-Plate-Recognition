
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



## YOLOv7
YOLOv7 was made by [@WongKinYiu](https://github.com/WongKinYiu).

YOLOv7's Repository:
- [https://github.com/WongKinYiu/yolov7.git](https://github.com/WongKinYiu/yolov7.git)




