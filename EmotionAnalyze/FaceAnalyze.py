# -*- codeing = utf-8 -*-
# @Time: 2022/4/14 14:33
# @Author: 刘
# @File: FaceAnalyze.py
# @Software: PyCharm
import time
import threading
import cv2
from deepface import DeepFace

rtmp_str = 'rtmp://media3.scctv.net/live/scctv_800'

rtmp_url = 'rtmp://118.195.200.217/live/livestream'

# 图片分析的emotion
# emotion1={'angry': 0.8526123637055555, 'disgust':0 , 'fear': 0,
#            'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 0,
#            'neutral': 0.0013034104862755555}


emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
           'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
           'neutral': 95.19895882655555}


class FaceAnalyze(threading.Thread):
    """docstring for Producer"""
    global emotion

    def __init__(self, rtmp_url='rtmp://media3.scctv.net/live/scctv_800'):
        super(FaceAnalyze, self).__init__()
        self.__count = 0  # 计数
        self.__emotion = emotion  # 表情值
        self.__rtmp_str = rtmp_url  # 拉流地址
        self.__stopFlag = False  # 停止标志
        # 通过cv2中的类获取视频流操作对象cap
        self.__cap = cv2.VideoCapture(self.__rtmp_str)
        # 调用cv2方法获取cap的视频帧（帧：每秒多少张图片）
        # fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.__fps = self.__cap.get(cv2.CAP_PROP_FPS)
        print(self.__fps)
        # 获取cap视频流的每帧大小
        # self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.size = (self.width, self.height)
        # print(self.size)

    def run(self):
        print('in producer')
        ret, image = self.__cap.read()
        while ret and (not self.__stopFlag):
            self.__count += 1
            if self.__count == 15:
                self.__count = 0
                try:
                    self.__emotion = DeepFace.analyze(image, actions=['emotion'])
                    print('分析后的结构', self.__emotion)
                    # print('emotion_photo:', self.temp)
                    self.__emotion = self.__emotion['emotion']
                except Exception as e:
                    print(e)
                # self.__emotion = self.__temp['emotion']
            ret, image = self.__cap.read()
            # cv2.imwrite('./images/{}.png'.format(self.count),image)

    def getEmotion(self):

        return self.__emotion

    def stop(self):
        self.__stopFlag = True


if __name__ == '__main__':
    count = 0
    fa = FaceAnalyze(rtmp_str)
    fa.start()
    while count < 15:
        print('count',count)
        time.sleep(1)
        count += 1
        print('hhh:',fa.getEmotion())
    fa.stop()
