# -*- codeing = utf-8 -*-
# @Time: 2022/4/21 13:43
# @Author: 刘
# @File: test1.py
# @Software: PyCharm
import requests
import time

# get
# post
rtmp_data_on = {
    'isChange': True,  # 是否改变
    # 'rtmp_url': 'rtmp://172.24.17.3:1935/live/5cf6c7979d12fd99',  # 推流地址
    # 'rtmp_url': 'rtmp://118.195.200.217/live/livestream',  # 推流地址
    # 'rtmp_url': 'rtmp://media3.scctv.net/live/scctv_800',  # 推流地址
    # 'post_url': 'http://192.168.0.102:5000/set_emotion',
    'status': True  # 是否推流
}

rtmp_data_ing = {
    'isChange': False,  # 是否改变
    # 'rtmp_url': 'rtmp://172.24.17.3:1935/live/5cf6c7979d12fd99',  # 推流地址
    # 'rtmp_url': 'rtmp://118.195.200.217/live/livestream',  # 推流地址
    # 'rtmp_url': 'rtmp://media3.scctv.net/live/scctv_800',  # 推流地址

    'status': True  # 是否推流
}
rtmp_data_off = {
    'isChange': True,  # 是否改变
    # 'rtmp_url': 'rtmp://172.24.17.3:1935/live/5cf6c7979d12fd99',  # 推流地址
    # 'rtmp_url': 'rtmp://118.195.200.217/live/livestream',  # 推流地址
    # 'rtmp_url': 'rtmp://media3.scctv.net/live/scctv_800',  # 推流地址
    'status': False  # 是否推流
}
if __name__ == '__main__':
    url = 'http://118.195.200.217:5000/rtmpanalyze'
    # 开始推流
    req = requests.post(url=url, json=rtmp_data_on)
    print(req.status_code)
    print(req.json())
    count = 0

    # 推流中
    while count < 20:
        time.sleep(1)
        print('count', count)
        count += 1
        req = requests.post(url=url, json=rtmp_data_ing)
        print(req.status_code)
        print(req.json())

    # 结束推流
    req = requests.post(url=url, json=rtmp_data_off)
    print(req.status_code)
    print(req.json())
