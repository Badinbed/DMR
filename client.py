# -*- coding: utf-8 -*-import time


import socket
import time
import sys
#RPi's IP
SERVER_IP = "192.168.1.6"
SERVER_PORT = 8888
def send(digit):
    print("Starting socket: TCP...")
    server_addr = (SERVER_IP, SERVER_PORT)
    socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while True:
        try:
            print("Connecting to server @ %s:%d..." %(SERVER_IP, SERVER_PORT))
            socket_tcp.connect(server_addr)
            break
        except Exception:
            print("Can't connect to server,try it latter!")
            time.sleep(1)
            continue
    #digit=1000
    str='%d' %digit
    str = str.encode()
    socket_tcp.send(str)  #发送信息
    print(socket_tcp.recv(1024).decode()) #打印接收消息，并且译码
    socket_tcp.close() #关闭连接
