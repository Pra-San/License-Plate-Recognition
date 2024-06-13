import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import easyocr
import os
import requests

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort.sort import *

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]  # default to random color if not provided
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
def inFormat(text):
    if len(text) < 6:
        return False
    if text[0].isalpha() and text[1].isalpha() and text[2].isdigit() and text[3].isdigit() and text[4].isalpha() and \
            (text[5].isalpha() or text[5].isdigit()) and (len(text) > 6 and text[6:].isdigit()):
        return True
    else:
        return False
    
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Initialize SORT tracker
    sort_tracker = Sort()

    # Dictionary to store the last three directions for each object
    direction_history = {}

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Prepare detections for SORT
                sort_detections = []
                for *xyxy, conf, cls in det:
                    xyxy_cpu = [x.cpu().numpy() for x in xyxy]
                    sort_detections.append([xyxy_cpu[0], xyxy_cpu[1], xyxy_cpu[2], xyxy_cpu[3], conf.cpu().numpy()])

                # Update SORT tracker
                tracked_objects = sort_tracker.update(np.array(sort_detections))

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results and apply OCR
                for *xyxy, obj_id in tracked_objects:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (obj_id, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'ID {int(obj_id)}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # Extract the bounding box region and apply OCR
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Ensure the coordinates are within the image bounds
                        if x1 < 0: x1 = 0
                        if y1 < 0: y1 = 0
                        if x2 > im0.shape[1]: x2 = im0.shape[1]
                        if y2 > im0.shape[0]: y2 = im0.shape[0]

                        # Check if the ROI is valid
                        if x2 > x1 and y2 > y1:
                            roi = im0[y1:y2, x1:x2]
                            if roi.size > 0:
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
                                ocr_result = reader.readtext(thresh)
                                j=0
                                for i in thresh:
                                    cv2.imwrite(".\\runs\\"+str(j)+".jpg",thresh)
                                    j+=1
                                for (bbox, text, prob) in ocr_result:
                                    text = text.replace(" ", "")
                                    text = text.upper()
                                    
                                    print(f"OCR Text: {text} (Confidence: {prob:.2f}) for ID {int(obj_id)}")
                                    if len(text)<9 or len(text)>10:
                                        # os._exit(os.EX_OK)
                                        continue
                                    dict_char_to_int = {'O': '0','I': '1','J': '3','A': '4','G': '6','S': '5'}
                                    dict_int_to_char = {'0': 'O','1': 'I','3': 'J','4': 'A','6': 'G','5': 'S'}
                                    license_plate_ = ''
                                    if len(text) == 10:
                                        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_int_to_char, 5: dict_int_to_char, 
                                                6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int, 9: dict_char_to_int}
                                        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                                            if text[j] in mapping[j].keys():
                                                license_plate_ += mapping[j][text[j]]
                                            else:
                                                license_plate_ += text[j]
                                    else:
                                        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_int_to_char, 
                                                5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int, 8: dict_char_to_int}
                                        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                                            if text[j] in mapping[j].keys():
                                                license_plate_ += mapping[j][text[j]]
                                            else:
                                                license_plate_ += text[j]

                                    state_list = ['AP', 'AR', 'AS', 'BR', 'CG', 'GA', 'GJ', 'HR', 'HP', 'JH', 'KA', 'KL', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'WB', 'AN', 'CH', 'DN', 'DL', 'JK', 'LA', 'LD', 'PY']

                                    # Determine direction
                                    if obj_id not in direction_history:
                                        direction_history[obj_id] = []

                                    # Get the current bounding box center
                                    nc_x, nc_y = (x1 + x2) / 2, (y1 + y2) / 2

                                    # Check if there are previous entries in direction_history
                                    if len(direction_history[obj_id]) >= 1:
                                        pc_x, pc_y = direction_history[obj_id][-1][:2]
                                        if nc_y > pc_y:
                                            direction = "towards"
                                        elif nc_y < pc_y:
                                            direction = "away"
                                        else:
                                            direction = "static"

                                        # Append the direction to the history
                                        direction_history[obj_id].append((nc_x, nc_y, direction))

                                        # Only keep the last three directions
                                        if len(direction_history[obj_id]) > 3:
                                            direction_history[obj_id].pop(0)

                                        # Print direction for each detection
                                        # print(f"ID {obj_id} direction: {direction}")

                                        # Check if the last three directions are the same
                                        if license_plate_[0:2] in state_list and prob > 0.7 and inFormat(license_plate_):
                                            text = license_plate_
                                            if len(direction_history[obj_id]) >= 3 and all(len(d) == 3 and d[2] == direction for d in direction_history[obj_id][-3:]):
                                                if direction == "away":
                                                    url = 'http://localhost:5000/vehicle_exit'
                                                    response = requests.post(url, params={'license_plate': text})

                                                    # Check the response
                                                    if response.status_code == 201:
                                                        print("Vehicle entry recorded successfully")
                                                    elif response.status_code == 200:
                                                        print("Entry already exists for license plate:", text)
                                                    else:
                                                        print("Error:", response.json()['error'])
                                                elif direction == "towards":
                                                    url = 'http://localhost:5000/vehicle_entry'
                                                    response = requests.post(url, params={'license_plate': text})
                                                    # Check the response
                                                    if response.status_code == 201:
                                                        print("Vehicle entry recorded successfully")
                                                    elif response.status_code == 200:
                                                        print("Entry already exists for license plate:", text)
                                                    else:
                                                        print("Error:", response.json()['error'])
                                                else:
                                                    pass                                           

                                            
                                # Send a POST request to the Flask application
                                            
                                            print(f"Object {obj_id} is moving {direction}")

                                    else:
                                        # If no previous entry, append the current center
                                        direction_history[obj_id].append((nc_x, nc_y))

            # Stream results
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print(f"Results saved to {save_dir}")
        # if platform == 'darwin' and not opt.update:  # MacOS
        #     os.system('open ' + save_dir)

    print(f'Done. ({time.time() - t0:.3f}s)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam, 1 for external USB camera
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

