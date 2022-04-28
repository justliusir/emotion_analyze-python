# -*- codeing = utf-8 -*-
# @Time: 2022/4/20 15:12
# @Author: 刘
# @File: startup.py
# @Software: PyCharm

import time
import threading
import socket
import requests

from EmotionAnalyze import FaceAnalyze
from EmotionAnalyze import AudioAnalyze

rtmp_url = 'rtmp://media3.scctv.net/live/scctv_800'

# 初始表情
emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
           'happy': 2.3132647399755555e-06, 'sad': 22.5444322850955555, 'surprise': 1.3817945387555555,
           'neutral': 95.19895882655555}


# audio='sad'+20

# face_emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
#                 'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
#                 'neutral': 95.19895882655555}
# audio_emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
#                  'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
#                  'neutral': 95.19895882655555}
# text_emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
#                 'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
#                 'neutral': 95.19895882655555}


# 情况一
# face={
#     'hay':70,
#     'hhh':20,
#     'zzz':10
# }
# audio={
#     'hay':100,
#     'hhh':0
# }
#
# eee={
#     'hay':85,
#     'hhh':10,
#     'zzz':5,
# }


# 情况二
# face={
#     'hay':70,
#     'hhh':20,
#     'zzz':10
# }
# audio={
#     'hay':0,
#     'hhh':100
# }
#
# eee={
#     'hay':35,
#     'hhh':60,
#     'zzz':0,
# }


def mark(face_emotion: dict, audio_emotion: dict):
    global emotion
    emotion['angry'] = (face_emotion['angry'] + audio_emotion['angry']) / 2  # 先计算两个的text_emotion['angry']
    # 中立的
    emotion['disgust'] = face_emotion['disgust']
    # 害怕的
    emotion['fear'] = face_emotion['fear']
    # 开心的
    emotion['happy'] = (face_emotion['happy'] + audio_emotion['happy']) / 2
    # 伤心的
    emotion['sad'] = (face_emotion['sad'] + audio_emotion['sad']) / 2
    # 惊讶的
    emotion['surprise'] = face_emotion['surprise']
    # 焦虑的
    emotion['neutral'] = (face_emotion['neutral'] + audio_emotion['neutral']) / 2
    return emotion


class SetEmotion(threading.Thread):
    global emotion  # , audio_emotion, text_emotion

    def __init__(self, rtmp_url='rtmp://media3.scctv.net/live/scctv_800',
                 post_url='http://192.168.0.102:5000/set_emotion'):
        super(SetEmotion, self).__init__()
        self.__stopFlag = False
        self.__emotion = emotion
        self.__post_url = post_url
        self.__faceanalyze = FaceAnalyze.FaceAnalyze(rtmp_url)
        self.__audioanalyze = AudioAnalyze.AudioAnalyze(rtmp_url)
        # self.__url = url

    def run(self):
        self.__faceanalyze.start()
        self.__audioanalyze.start()
        while not self.__stopFlag:
            time.sleep(1)
            face = self.__faceanalyze.getEmotion()
            audio = self.__audioanalyze.getEmotion()
            print('face:', face)
            print('audio:', audio)
            self.__emotion = mark(face_emotion=face, audio_emotion=audio)
            print(self.__emotion)
            try:
                req = requests.post(url=self.__post_url, json=self.__emotion)
                print(req.status_code)
                print(req.json())
            except Exception as e:
                print(e)

    def stop(self):
        self.__stopFlag = True
        self.__faceanalyze.stop()  # 脸部分析停止
        self.__audioanalyze.stop()  # 脸部分析停止


def use(rtmp_url: str, post_url: str, status: bool):
    ip = socket.gethostbyname(socket.gethostname())  # 获取本机ip地址
    print('rtmp_url', rtmp_url)
    print('status', status)
    if status == True:
        global pe
        pe = SetEmotion(rtmp_url=rtmp_url, post_url=post_url)
        # pe.setDaemon(True)
        pe.start()
    elif status == False:
        pe.stop()


if __name__ == '__main__':
    rtmp = 'rtmp://118.195.200.217/live/5cf6c7979d12fd99'
    use(rtmp_url=rtmp, post_url='http://192.168.145.78:5000/set_emotion', status=True)
    time.sleep(10)
    count = 0
    while count < 15:
        time.sleep(1)
        print('count', count)
        count += 1
    use(rtmp_url=rtmp, post_url='http://192.168.145.78:5000/set_emotion', status=False)
    # use('hhh', False)
