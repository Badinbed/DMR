# -*- coding: utf-8 -*-
__author__ = '凌霄一笑'
import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import predict
import cv2
from PIL import Image, ImageTk
import threading
import time
import client




class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None
    send_sign=0
    roi='11.jpg'
    r=0
    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right1 = ttk.Frame(self)
        frame_right2 = ttk.Frame(self)
        win.title("数字表识别")
        w, h = win.maxsize()
        win.geometry("{}x{}".format(w, h)) 
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="10", pady="10")
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        ttk.Label(frame_right1, text='表盘位置：').grid(column=0, row=0, sticky=tk.W)

        from_pic_ctl = ttk.Button(frame_right2, text="打开相册", width=20, command=self.from_pic)
        from_vedio_ctl = ttk.Button(frame_right2, text="打开摄像头", width=20, command=self.from_vedio)
        
        start_ctl = ttk.Button(frame_right2, text="拍照", width=20, command=self.start_vedio)   
        send_ctl = ttk.Button(frame_right2, text="上传", width=20, command=self.send) 
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.grid(column=0, row=1, sticky=tk.W)
        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=2, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=3, sticky=tk.W)
        from_pic_ctl.pack(anchor="se", pady="10")
        from_vedio_ctl.pack(anchor="se", pady="10")

        start_ctl.pack(anchor="se", pady="10")

        send_ctl.pack(anchor="se", pady="10")

        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            self.r_ctl.configure(text=str(r))
            self.update_time = time.time()
         
        elif self.update_time + 8 < time.time():
            self.roi_ctl.configure(state='disabled')
            self.r_ctl.configure(text="")
     



    def from_vedio(self):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('警告', '摄像头打开失败！')
                self.camera = None
                return
      
    
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True

    def from_pic(self):
        self.thread_run = False
        self.send_signn=0
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg"), ("png图片", "*.png")])
        if self.pic_path:
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            r, roi = self.predictor.predict(img_bgr)
            total_port_num = 0
            for singleline in r:
                if (singleline.isdigit()):
                    slot_port_num = int(singleline)
                    total_port_num = total_port_num*10 + slot_port_num

                          
            self.show_roi(total_port_num, roi)

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        #predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)

        
    def start_vedio(self):
        
        _, img_bgr = self.camera.read()
        self.img=img_bgr
        self.thread_run = False
        self.send_sign=1
        self.imgtk = self.get_imgtk(img_bgr)
        self.image_ctl.configure(image=self.imgtk)
        r, roi = self.predictor.predict(img_bgr)
        total_port_num = 0
        for singleline in r:
            if (singleline.isdigit()):
                slot_port_num = int(singleline)
                total_port_num = total_port_num*10 + slot_port_num
        self.r=total_port_num            
        self.show_roi(total_port_num, roi)
        
    def send(self):
        if (self.send_sign):
            time.time()

 
            time.localtime(time.time())

 
            t1=time.strftime('%Y-%m-%d',time.localtime(time.time()))
 
            t2=time.strftime('%H:%M:%S',time.localtime(time.time()))            
            cv2.imwrite('%s_%s_%d.jpg'%(t1,t2,self.r),self.img)
            client.send(self.r)
            self.send_sign=0
        else:
            return
            
        

def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()

    surface = Surface(win)
    # close,退出输出destroy
    win.protocol('WM_DELETE_WINDOW', close_window)
    # 进入消息循环
    win.mainloop()
