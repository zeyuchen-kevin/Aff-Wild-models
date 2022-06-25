# -*- coding: utf-8 -*-

# This file is created by Zeyu Chen, BIAI, Inc.
# Title           :video2image.py
# Version         :1.0
# Email           :k83110835@126.com
# Copyright       :BIAI, Inc.
# ==============================================================================


import os
import cv2
import shutil
from PIL import Image, ImageFilter
import numpy as np


def is_video(file_name):
    """
    This function will detect whether a file is a video.
    """
    video_ext = ['mp4', 'mov', 'mpg', 'avi']
    ext = file_name[file_name.rfind('.') + 1:].lower()
    return ext in video_ext


def save_image(image, addr, num):
    """
    Define the images to be saved.
    Args:
        image: the name of the saving image
        addr:  the picture directory address and the first part of the picture name
        num:   int dtype, the id number in the image filename
    """
    address = os.path.join(addr, str(num + 1) + '.jpg')
    cv2.imwrite(address, image)



def extract_frame(output_dir, video_path):

    # single video file
    if is_video(video_path):
        video_name = video_path.split('/')[-1].split('.')

        output_image_path = os.path.join(output_dir, video_name[0])

        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)

        os.makedirs(output_image_path)

        videoCapture = cv2.VideoCapture(video_path)

        success, frame = videoCapture.read()
        i = 0
        while success:
            # if frame.shape[0] < frame.shape[1]:
            #     frame = cv2.transpose(frame)
            #     frame = cv2.flip(frame, -1)
            save_image(frame, output_image_path, i)
            i += 1
            success, frame = videoCapture.read()

    # directory consisted of videos
    elif os.path.isdir(video_path):
        ls = os.listdir(video_path)
        for i, file_name in enumerate(sorted(ls)):
            if is_video(file_name):
                try:
                    print('Loading video {}'.format(file_name))
                    sub_video_name = file_name.split('.')

                    output_image_path = os.path.join(output_dir, sub_video_name[0])

                    if os.path.exists(output_image_path):
                        shutil.rmtree(output_image_path)

                    os.makedirs(output_image_path)

                    file_path = os.path.join(video_path, file_name)
                    videoCapture = cv2.VideoCapture(file_path)


                    success, frame = videoCapture.read()
                    i = 0
                    while success:
                        # if frame.shape[0] < frame.shape[1]:
                        #     frame = cv2.transpose(frame)
                        #     frame = cv2.flip(frame, -1)
                        save_image(frame, output_image_path, i)
                        i += 1
                        success, frame = videoCapture.read()
                except:
                    print('Processing video {} failed'.format(file_name))



if __name__ == '__main__':

    outdir = './images'
    videodir = './videos/105.avi'
    extract_frame(outdir, videodir)



