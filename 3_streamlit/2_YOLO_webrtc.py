import streamlit as st

from streamlit_webrtc import webrtc_streamer
from yolo_predictions import YOLO_Pred
import av

#Load the yolo model
yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    #any operation to be performed
    #flipped = img[::-1,:,:]
    pred_img = yolo.predictions(img)

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video":True,
                                                                                                    "audio":False})