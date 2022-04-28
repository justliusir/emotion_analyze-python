import threading
# import requests
import time
import wave
import pyaudio  # 跟音频，开始主要录音
import ffmpeg  # 拉流的
import torch
import torch.nn as nn
import librosa
import numpy as np
import queue  # 队列
from python_speech_features import sigproc, fbank, logfbank

# 语调分析的emotion
emotion = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}


# emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
#            'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
#            'neutral': 95.19895882655555}


# Parallel network model structure
class Parallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = Cnn_Transformer(4)
        self.Transformerr = Transformer_Encoder(4)
        self.Transformerr.load_state_dict(torch.load(path1, map_location='cpu'))
        # state_dict = torch.load(self.model_path, map_location='cpu')
        self.fc1_linear = nn.Linear(1576, 4)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x, x_next):
        x1, x2, x3 = self.cnn(x)
        y1, y2, y3 = self.Transformerr(x_next)
        complete_embedding = torch.cat([y3, x3], dim=1)  # trans
        output_logits = self.fc1_linear(complete_embedding)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax


# Structure of Convolutional Neural Network in Parallel Network Model
class Cnn_Transformer(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
        )
        self.fc1_linear = nn.Linear(1536, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):  # 输入的形状为32*1*40*63
        conv2d_embedding1 = self.conv2Dblock1(x)  # conv2d_embedding1为 32*130*2*3
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)  # conv2d_embedding1为 32*780
        output_logits = self.fc1_linear(conv2d_embedding1)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax, conv2d_embedding1


# The structure of Transformer-encoder in parallel network model
class Transformer_Encoder(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=512,
            dropout=0.5,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.fc1_linear = nn.Linear(40, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):  # 输入的形状为32*1*40*63
        x_maxpool = self.transformer_maxpool(x)  # x_maxpool为32*1*40*15
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)  # x_maxpool_reduced为32*40*15
        x = x_maxpool_reduced.permute(2, 0, 1)  # x 为 15*32*40
        transformer_output = self.transformer_encoder(x)  # transformer_output 15*32*40
        transformer_embedding = torch.mean(transformer_output, dim=0)  # transformer_embedding 32*40
        output_logits = self.fc1_linear(transformer_embedding)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax, transformer_embedding


FEATURE1 = "logfbank"
FEATURE2 = 'mfcc'


# 特征提取
class FeatureExtractor(object):
    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, X):
        X_features = None
        accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('logfbank'):
            X_features = self.get_logfbank(X)
        if features_to_use in ('mfcc', 26):
            X_features = self.get_mfcc(X)
        if features_to_use in ('fbank'):
            X_features = self.get_fbank(X)
        if features_to_use in ('melspectrogram'):
            X_features = self.get_melspectrogram(X)
        if features_to_use in ('spectrogram'):
            X_features = self.get_spectrogram(X)
        if features_to_use in ('pase'):
            X_features = self.get_Pase(X)
        return X_features

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=20)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, n_mfcc=40):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            return mfcc_data

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.rate)
            mel = np.log10(mel + 1e-10)
            return mel

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features

    def get_Pase(self, X):
        return X


CHUNK = 1000  # 每次处理1000帧这样的数据
# FORMAT = pyaudio.paInt16   # 以16进制的方式打开音频文件
FORMAT = pyaudio.paInt16  # 以16进制的方式打开音频文件
CHANNELS = 1  # 声道数
RATE = 16000  # 每秒提取16000帧的数据   采样率
path = './models/augment-0_logfbank.pth'
path1 = './models/augment_1.pth'

q = queue.Queue(-1)
p = pyaudio.PyAudio()
out_print = "Please wait "


def mark(emotion_type: str, emotion: dict):
    print(type(emotion_type))
    print(emotion_type)
    emotion_list = ['neutral', 'sad', 'angry', 'happy']
    for item in emotion_list:
        if item == emotion_type:
            emotion[item] = 100
        else:
            emotion[item] = 0
    return


class PullRtmp(threading.Thread):
    def __init__(self, rtmp_url='rtmp://media3.scctv.net/live/scctv_800'):
        threading.Thread.__init__(self)
        self.__stopFlag = False  # 就是一个标志位而已
        self.__pullprocess = ffmpeg.input(rtmp_url).output('-', format='s16le', acodec='pcm_s16le', ac=1,
                                                           ar='16k').overwrite_output().run_async(pipe_stdout=True)

    def run(self):
        while not self.__stopFlag:
            in_bytes = self.__pullprocess.stdout.read(1000)
            if not in_bytes:
                break
            q.put(in_bytes)

    def stop(self):
        self.__stopFlag = True


class Analyze(threading.Thread):  # 实现解码
    global emotion

    def __init__(self, WAVE_OUTPUT_FILENAME="audio_source.wav"):
        threading.Thread.__init__(self)
        self.__stopFlag = False  # 就是一个标志位而已
        self.__WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME
        self.__frames = []
        self.__emotion = emotion

    def run(self):
        # 添加的
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 这一步是开始进行解码的操作
        print('decoder beginning')
        frame = []  # 用于存放中间变量的数据
        while not self.__stopFlag:
            frame.append(q.get())
            if (len(frame) < int(RATE / (CHUNK) * 4)):  # 如果读取的声音片段太小
                continue  # 继续执行上面的代码
            """修改了"""
            self.__frames = frame  # 每段要处理的数据存储
            frame = []  # 初始化列表
            if (len(self.__frames) > 20 * int(RATE / CHUNK)):  # 如果列表的长度过长
                del self.__frames[:int(RATE / CHUNK) * 4]  # 删除列表前面的一部分数据
            wf = wave.open(self.__WAVE_OUTPUT_FILENAME, 'wb')  # 打开那个声音文件
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.__frames))  # 将声音文件写入wav文件
            wf.close()
            model = Parallel()  # 调用模型
            model.load_state_dict(torch.load(path, map_location='cpu'))  # 加载模型
            # model = model.cuda()  # u、cuda加速
            model = model.to(device)  # u、cuda加速
            wav_data, _ = librosa.load(self.__WAVE_OUTPUT_FILENAME, 16000)  # 加载wav文件,其中的16000为采样频率吧
            print('wav_data的大小', len(wav_data))
            index = 0  # 索引从0开始
            X1 = []  # 设置列表，用来存储已经处理过的数据

            if len(wav_data) > 31999:
                X1.append(wav_data)
                Temp_X1 = np.array(X1)
                featureExtractor = FeatureExtractor(16000)  # 将每次取样的16000个点进行提取特征值
                X = featureExtractor.get_features(FEATURE2, Temp_X1)  # 利用mfcc提取特征，数据是处理过的x1
                Temp = featureExtractor.get_features(FEATURE1, Temp_X1)
                X = torch.tensor(X).unsqueeze(1).to(device)  # 在第二个位置上增加一个一维的数据
                Temp = torch.tensor(Temp).unsqueeze(1).to(device)
                with torch.no_grad():
                    _, out = model(Temp.float(), X.float())
                # 获取最大值以及最大值的索引
                max_value, max_idx = torch.max(out, dim=1)
                if max_idx[-1] == 0:
                    temp_print = 'neutral'
                    mark(temp_print, self.__emotion)

                elif max_idx[-1] == 1:
                    temp_print = 'sad'
                    mark(temp_print, self.__emotion)
                elif max_idx[-1] == 2:
                    temp_print = 'angry'
                    mark(temp_print, self.__emotion)
                else:
                    temp_print = 'happy'
                    mark(temp_print, self.__emotion)
                out_print = temp_print
                print(out_print)
                print(self.__emotion)
                audio_data = {
                    'type': 'audio',
                    'emotion': self.__emotion
                }
                index += int(0.4 * 16000)
        p.terminate()

    def getEmotion(self):

        return self.__emotion

    def stop(self):
        self.__stopFlag = True


class AudioAnalyze(threading.Thread):
    def __init__(self, rtmp_url='rtmp://media3.scctv.net/live/scctv_800'):
        super(AudioAnalyze, self).__init__()
        # self.__stopFlag = False
        self.__pullrtmp = PullRtmp(rtmp_url)
        self.__analyze = Analyze()

    def run(self):
        self.__pullrtmp.start()
        self.__analyze.start()

    def getEmotion(self):
        return self.__analyze.getEmotion()

    def stop(self):
        self.__analyze.stop()
        self.__pullrtmp.stop()


# 使用
def use():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(out_print)

    pr = PullRtmp()
    aa = Analyze('source.wav')

    pr.start()
    aa.start()
    count = 0
    while count < 15:
        time.sleep(1)
        print(count)
        count += 1
        print(aa.getEmotion())
    pr.stop(True)
    aa.stop(True)
    # decoder.join()
    p.terminate()


if __name__ == '__main__':
    aa = AudioAnalyze(rtmp_url='rtmp://media3.scctv.net/live/scctv_800')
    aa.start()
    count = 0
    while count < 15:
        time.sleep(1)
        print(count)
        count += 1
        print(aa.getEmotion())
    aa.stop()
    # use()
