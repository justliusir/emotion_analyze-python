# -*- codeing = utf-8 -*-
# @Time: 2022/4/8 13:24
# @Author: 刘
# @File: emotion_flask.py
# @Software: PyCharm
from flask import Flask, jsonify, request, make_response, render_template

from deepface import DeepFace
import argparse
import base64

import EmotionAnalyze.startup as api

# from EmotionAnalyze import AudioAnalyze,FaceAnalyze
# ---------------------------------------
app = Flask(__name__)

'''#变量
    #推流地址
    rtmp_str = 'rtmp://media3.scctv.net/live/scctv_800'(cctv地址)
    rtmp_url = 'rtmp://118.195.200.217/live/livestream'(本公司流媒体服务器地址)
    #初始表情
    emotion
    # 返回值
    response = {}(久版deepface的应用程序和小程序用得到，在/analyze中使用)
    用于存储推流前端发来的数据
    rtmp_data={
    'status':False, #是否推流
    'rtmp_url':'',  #推流地址
    'rtmp_key':''   #推流秘钥
}
    
'''
# rtmp_data = {
#         'isChange': True, # 是否改变
#         'rtmp_url': 'rtmp://118.195.200.217/live/livestream',  # 推流地址
#         'status': True  # 是否推流
# }
# # 推流地址
# rtmp_str = 'rtmp://media3.scctv.net/live/scctv_800'
# rtmp_url = 'rtmp://118.195.200.217/live/livestream'


# 初始表情
emotion = {'angry': 0.8526123637055555, 'disgust': 0.0013034104862755555, 'fear': 0.020891733374055555,
           'happy': 2.3132647399755555e-06, 'sad': 2.5444322850955555, 'surprise': 1.3817945387555555,
           'neutral': 95.19895882655555}

# 返回值
response = {}

# ---------------------------------------
'''# 网页
    1.首页:index.html
    2.运维岗位:yunwei.html
    3.Python岗位:python.html
    4.算法岗位:suanfa.html
    5.数据分析岗位:fenxi.html
'''


# 首页
@app.route('/')
def index():
    return render_template('index.html')


# 其他页面
@app.route('/<page>')
def page(page):
    # 重定向运维yunwei.html
    if page == 'yunwei.html':
        return render_template('yunwei.html')
    # 重定向Python工程师python.html
    elif page == 'python.html':
        return render_template('python.html')
    # 重定向算法suanfa.html
    elif page == 'suanfa.html':
        return render_template('suanfa.html')
    # 重定向数据分析fenxi.html
    elif page == 'fenxi.html':
        return render_template('fenxi.html')
    # 重定向首页index.html
    else:
        return render_template('index.html')


# ---------------------------------------
'''
    #算法
    # 1./set_emotion,情绪识别api(emotion_api)获取推流状态、地址以及秘钥,分析完情绪后，向这里发post请求，更新emotion值
    # 2./rtmpanalyze,前端应用通过该接口,传递推流地址，以及获取更新了emotion值
    # 3./analyze,旧时的deepface后端接口，供旧时的deepface前端应用程序和小程序使用
'''


# 1./set_emotion,情绪识别api(emotion_api)获取推流状态、地址以及秘钥,分析完情绪后，向这里发post请求，更新emotion值
@app.route('/set_emotion', methods=['POST'])
def set_emotion():
    global emotion
    data = request.get_json()
    if data:
        print(type(data))
        print(data)
        emotion = data
        print('succeed')
        return {'set_emotion': 'succeed'}
    return {'set_emotion': 'fail'}


'''# 到这弄两个变量，这样方便看
    1.如果没有传就默认
        rtmp_url = 'rtmp://media3.scctv.net/live/scctv_800'
        post_url = 'http://192.168.254.78:5000/set_emotion'
'''
rtmp_url = 'rtmp://media3.scctv.net/live/scctv_800'
post_url = 'http://192.168.0.102:5000/set_emotion'


# 2./rtmpanalyze,前端应用通过该接口,传递推流地址，以及获取更新了emotion值
@app.route('/rtmpanalyze', methods=['POST'])
def rtmp_analyze():
    global emotion,rtmp_url, post_url
    data = request.get_json()
    '''
        # 开启
        data['isChange'] = True
        data['rtmp_url'] = 'rtmp_url'
        data['status'] = True
        # 推流中
        data['isChange'] = False
        data['rtmp_url'] = 'rtmp_url'
        data['status'] = True
        # 关闭
        data['isChange'] = True
        data['rtmp_url'] = 'rtmp_url'
        data['status'] = False
    '''
    print(data)

    if data.get('isChange') == True:
        # 传了rtmp_url
        if data.get('rtmp_url'):
            # 改成传过来的rtmp_url
            rtmp_url = data.get('rtmp_url')
        # 传了post_url
        elif data.get('post_url'):
            # 改成传过来的post_url
            post_url = data.get('post_url')
        api.use(rtmp_url=rtmp_url, post_url=post_url, status=data.get('status'))
    print('返回的：', emotion)
    return emotion


# 3./analyze,旧时的deepface后端接口，供旧时的deepface前端应用程序和小程序使用
@app.route('/analyze', methods=['POST'])
def analyze():
    global emotion
    # print('***********')
    action = ['emotion']
    req = request.get_json()
    # 如果是小程序,那么执行这个分支。小程序是{'face_img':[.....]}
    if 'face_img' in req.keys():
        # print('ok')
        data = req['face_img']
        try:
            # result = DeepFace.analyze(data, actions=['emotion'])
            result = DeepFace.analyze(data, actions=action)
            response['data'] = result
            response['statusCode'] = 200
            print(result)
        except Exception as e:
            response['data'] = 'null'
            response['statusCode'] = 500
        return response  # 回复一个{statusCode:200/500;data:null/有数}
    # 如果是原本的flutter deepface
    elif "img" in list(req.keys()):
        data = req["img"]  # list
        try:
            result = DeepFace.analyze(data, actions=action)
            response['data'] = result
            response['statusCode'] = 200
            print(result)
        except Exception as e:
            response['data'] = 'null'
            response['statusCode'] = 500
        return response  # 回复一个{statusCode:200/500;data:null/有数}
    response['data'] = 'null'
    response['statusCode'] = 500
    return response


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-p', '--port',
    #     type=int,
    #     default=5000,
    #     help='Port of serving api')
    # args = parser.parse_args()
    # app.run(host='0.0.0.0', port=args.port, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)
