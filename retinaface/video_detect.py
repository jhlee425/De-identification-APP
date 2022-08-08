#-*- coding: UTF-8 -*-
import argparse
from re import sub
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
from PIL import Image
import cv2
import torchvision
import eval_widerface
import torchvision_model
import model
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from deid_utils import mosaic, elastic, gaussian_blur, shuffle

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def get_args():
    parser = argparse.ArgumentParser(description="Detect program for retinaface.")
    parser.add_argument('--video_path', type=str, default='London_Trim2.mp4', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, help='Path for model')
    parser.add_argument('--save_path', type=str, default='./out/result.mp4', help='Path for result image')
    parser.add_argument('--score_threshold', type=float, default='0.15', help='Detections with a score under this threshold will not be considered. This currently only works in display mode')
    parser.add_argument('--label_dis', type=str, default='undisplay', help='class label')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=152)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    parser.add_argument('--methods', type=str, default='', help='de-id methods', )
    parser.add_argument('--deid_level', type=float, help='deid level')
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()

    # Read video
    cap = cv2.VideoCapture(args.video_path)

    codec = cv2.VideoWriter_fourcc(*'mp4v')

    width = int(cap.get(3))
    height = int(cap.get(4))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = 10

    result_dir = args.save_path + '/' + os.path.basename(args.video_path)
    out = cv2.VideoWriter(result_dir, codec, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    while(True):
        ret, img = cap.read()

        if not ret:
            print('Video open error.')
            break

        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        if not args.scale == 1.0:
            size1 = int(img.shape[1]/args.scale)
            size2 = int(img.shape[2]/args.scale)
            img = resize(img.float(),(size1,size2))

        input_img = img.unsqueeze(0).float().cuda()
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, RetinaFace, args.score_threshold, iou_threshold=0.5)

        # np_img = resized_img.cpu().permute(1,2,0).numpy()
        np_img = img.cpu().permute(1,2,0).numpy()
        np_img.astype(int)
        img = np_img.astype(np.uint8)


        for j, boxes in enumerate(picked_boxes):
            if boxes is not None:
                for box,landmark,score in zip(boxes,picked_landmarks[j],picked_scores[j]):
                    c1, c2 = (int(box[0]),int(box[1])), (int(box[2]),int(box[3])) # c1 = (x, y), c2 = (x+w, y+h)
                    sub_img = img[c1[1]:c2[1], c1[0]:c2[0]]

                    if args.methods == 'blur':
                        if args.deid_level == 1:
                            sub_img = gaussian_blur(sub_img, (3, 3))
                        elif args.deid_level == 2:
                            sub_img = gaussian_blur(sub_img, (9, 9))
                        elif args.deid_level == 3:
                            sub_img = gaussian_blur(sub_img, (27, 27))
                    elif args.methods == 'mosaic':
                        if args.deid_level == 1:
                            sub_img = mosaic(sub_img, ratio=0.4)
                        elif args.deid_level == 2:
                            sub_img = mosaic(sub_img, ratio=0.2)
                        elif args.deid_level == 3:
                            sub_img = mosaic(sub_img, ratio=0.1)
                    elif args.methods == 'shuffle':
                        sub_img = shuffle(sub_img)
                    elif args.methods == 'distortion':
                        if args.deid_level == 1:
                            sub_img = elastic(sub_img, alpha=5000, sigma=8, random_state=None)   
                        elif args.deid_level == 2:
                            sub_img = elastic(sub_img, alpha=5000, sigma=6, random_state=None)
                        elif args.deid_level == 3:
                            sub_img = elastic(sub_img, alpha=5000, sigma=4, random_state=None)

                    img[c1[1]:c1[1]+sub_img.shape[0], c1[0]:c1[0]+sub_img.shape[1]] = sub_img
                    # cv2.circle(img,(int(landmark[0]),int(landmark[1])),radius=1,color=(0,0,255),thickness=2)
                    # cv2.circle(img,(int(landmark[2]),int(landmark[3])),radius=1,color=(0,255,0),thickness=2)
                    # cv2.circle(img,(int(landmark[4]),int(landmark[5])),radius=1,color=(255,0,0),thickness=2)
                    # cv2.circle(img,(int(landmark[6]),int(landmark[7])),radius=1,color=(0,255,255),thickness=2)
                    # cv2.circle(img,(int(landmark[8]),int(landmark[9])),radius=1,color=(255,255,0),thickness=2)
                    if args.label_dis == 'display':
                        cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),thickness=2)
                        cv2.putText(img, text=str(score.item())[:5], org=(int(box[0]),int(box[1])), fontFace=font, fontScale=0.5,
                                    thickness=1, lineType=cv2.LINE_AA, color=(255, 255, 255))

        out.write(img)
        cv2.imshow('RetinaFace-Pytorch',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print('Now quit.')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
