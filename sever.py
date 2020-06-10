import socket
import time
import sys

#define host ip: Rpi's IP
HOST_IP = "192.168.1.6"
HOST_PORT = 8888
print("Starting socket: TCP...")
#1.create socket object:socket=socket.socket(family,type)
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("TCP server listen @ %s:%d!" %(HOST_IP, HOST_PORT) )
host_addr = (HOST_IP, HOST_PORT)
#2.bind socket to addr:socket.bind(address)
socket_tcp.bind(host_addr)
#3.listen connection request:socket.listen(backlog)
socket_tcp.listen(1)
#4.waite for client:connection,address=socket.accept()



#5.handle

print("Receiving package...")
while True:
    socket_con, (client_ip, client_port) = socket_tcp.accept()
    try:
        data=socket_con.recv(512)
        if len(data)>0:
            print("Received:%s"%data)
            str = "Receive successfully!"
            str = str.encode()
            socket_con.send(str)
            #socket_con.send(data)
            #time.sleep(1)
            continue
    except Exception:
            socket_tcp.close()
            sys.exit(1)
