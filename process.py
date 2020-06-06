import sys
sys.path.append('/home/tg/shwd_ver3/')

import os
os.chdir('/home/tg/shwd_ver3/')

import struct
import shutil

from recognize import getNames, attendance,security,prepareFacebank
import cv2
import numpy
import json
import datetime


#print(type(sys.stdout))
#print(dir(sys.stdout))
output = sys.stdout
log = open('log.txt','a')
sys.stdout = log

def packPDU(data, buf = bytearray(''.encode('utf-8'))):
    return struct.pack('<i', len(data)) + data.encode('utf-8') + struct.pack('<i', len(bytearray(buf))) + bytearray(buf)
#-------我们主要参考handleSecurity
def handleSecurity(buf):#---------buf过来的是图像数据
    print('handleSecurity .....')
    image = numpy.asarray(bytearray(buf), dtype="uint8")  
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    #cv2.imwrite("/home/tg/shwd_ver2/imgs/aaa.jpg", img)
    s = datetime.datetime.now()
    boxes, labels = security(img)  #-------------给算法图片，我们可以在后面再加一个参数，表示围栏坐标
    '''
    下面是算法返回的结果，供志强参考
    attendance boxes: [[8.1211688e+02 1.7016057e+02 8.3442053e+02 1.9467551e+02 4.0892112e-01]
                       [2.1058069e+02 3.2764636e+02 4.6475287e+02 6.4114136e+02 7.7464485e-01]
                       [8.3309241e+02 1.9821501e+02 8.6148999e+02 2.3330859e+02 2.3384960e-01]
                       [8.1284644e+02 1.7071570e+02 8.3369104e+02 1.9412038e+02 2.3040701e-01]]
    attendance labes: [0. 1. 0. 1.]
    '''
    e = datetime.datetime.now()

    t = (e - s).microseconds

    data = 0

    if(len(boxes) > 0):
        d = {'type' : 'security', 'num' : len(boxes), 'box' : boxes, 'label' : labels, 'time' : t}
        data = json.dumps(d)
    else:
        data = '{"type" : "security" ,"num" : 0, "time" : ' + str(t) + '}'

    #buf = cv2.imencode('.jpg', img)[1]
    PDU = packPDU(data, bytearray(buf))
    bytesWriten = 0
    #把算法结果和图片返回给QT程序
    while bytesWriten < len(PDU):
        bytesWriten = bytesWriten + os.write(1, PDU[bytesWriten:])

def handleAttendance(buf):
    print('handleAttendance .....image len=',len(buf))
    image = numpy.asarray(bytearray(buf), dtype="uint8")  
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #print("handleAttendance img:",img)
    #cv2.imwrite("/home/tg/shwd_ver2/imgs/aaa.jpg", img)
    s = datetime.datetime.now()
    boxes, labels, names = attendance(img)
    print('handleAttendance boxes:',boxes)
    print('handleAttendance labels:',labels)
    print('handleAttendance names:',names)
    e = datetime.datetime.now()

    t = (e - s).microseconds

    data = 0
    print('num=',len(boxes))
    if(len(boxes) > 0):
        d = {'type' : 'attendance', 'num' : len(boxes), 'box' : boxes, 'name' : names, 'label' : labels, 'time' : t}
        data = json.dumps(d)
    else:
        data = '{"type" : "attendance" ,"num" : 0, "time" : ' + str(t) + '}'

    #buf = cv2.imencode('.jpg', img)[1]
    print('data:',data)
    PDU = packPDU(data, bytearray(buf))
    bytesWriten = 0

    while bytesWriten < len(PDU):
        bytesWriten = bytesWriten + os.write(1, PDU[bytesWriten:])


def handleRegist(buf):
    print('handleRegist datalen=',len(buf))
    image = numpy.asarray(bytearray(buf), dtype="uint8")  
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    #cv2.imwrite("/home/tg/shwd_ver2/imgs/aaa.jpg", img)
    s = datetime.datetime.now()
    print()
    boxes, labels, names = attendance(img)
    print('handleRegist boxes:',boxes)
    print('handleRegist labels:',labels)
    print('handleRegist names:',names)
    e = datetime.datetime.now()
    print("handleRegist now:",e.strftime('%H:%M:%S.%f'))
    t = (e - s).microseconds

    data = 0

    if(len(boxes) > 0):
        d = {'type' : 'regist', 'num' : len(boxes), 'box' : boxes, 'name' : names, 'label' : labels, 'time' : t}
        data = json.dumps(d)
        print('d:',d)
    else:
        data = '{"type" : "regist" ,"num" : 0, "time" : ' + str(t) + '}'

    #buf = cv2.imencode('.jpg', img)[1]
    PDU = packPDU(data, bytearray(buf))
    bytesWriten = 0

    while bytesWriten < len(PDU):
        bytesWriten = bytesWriten + os.write(1, PDU[bytesWriten:])
def handleRegistGetFace(buf):
    print('handleRegistGetFace .....')
    image = numpy.asarray(bytearray(buf), dtype="uint8")  
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    s = datetime.datetime.now()
    boxes, labels, names = attendance(img)
    print('boxes:',boxes)
    e = datetime.datetime.now()

    t = (e - s).microseconds

    data = 0

    if(len(boxes) > 0):
        d = {'type' : 'getface', 'num' : len(boxes), 'box' : boxes, 'name' : names, 'label' : labels, 'time' : t}
        data = json.dumps(d)
    else:
        data = '{"type" : "getface" ,"num" : 0, "time" : ' + str(t) + '}'

    #buf = cv2.imencode('.jpg', img)[1]
    PDU = packPDU(data, bytearray(buf))
    bytesWriten = 0

    while bytesWriten < len(PDU):
        bytesWriten = bytesWriten + os.write(1, PDU[bytesWriten:])
def face_cut(pic_path,pic_name):
    face = cv2.CascadeClassifier('/home/tg/shwd_ver3/haarcascade_frontalface_alt2.xml') 
    img = cv2.imread(pic_path+pic_name)
    #print(img)
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(100):
        faces = face.detectMultiScale(img,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        print('faces:',faces)#396 274 143 143]
        if len(faces):
           X,Y,W,H=0,0,0,0
           x,y,w,h=faces[0]
           if x+w+100>img.shape[1]:
              W = img.shape[1]-x
           else:
              W = w+100
           if y+h+100>img.shape[0]:
              H = img.shape[0]-y
           else:
              H = h+100
           if x-100 < 0:
              X = 0
           else:
              X = x-100
           if y-100 < 0:
              Y = 0
           else:
              Y=y-100
           cv2.imwrite(pic_path+'/0.jpg',img[y-100:y+h+100,x-100:x+w+100])
           return True
    return False
def handleLoadModule(buf):
#    buf = bytearray(buf)
    print('handleLoadModule .....')
    Len = struct.unpack('<I', buf[:4])[0]
    key = buf[4:Len+4].decode('utf-8')
    array = buf[Len+4:]
    d = {}
    try:
        os.mkdir('data/facebank/' + key)
    except FileExistsError:
        d = {'type' : 'module', 'result' : 'rename'}
    else:
        print('key2222222222222222=',key)
        f = open('data/facebank/' + key + '/0.jpg', 'wb')
        print('f =',f)
        f.write(array)
        f.close()
        print('key33333333333333333=',key)
        #prepareFacebank()
        #names = getNames()
        #print('name=',names)
        #d = {'type' : 'module', 'names' : key, 'result' : 'success'}
        #prepareFacebank()
        
        try:
            prepareFacebank()
        except:
            d = {'type' : 'module', 'result' : 'bankfail'}
            #shutil.rmtree('data/facebank/' + key + '/')
        else:
            #names = getNames()
            d = {'type' : 'module', 'names' : key, 'result' : 'success'}   
        
        d = {'type' : 'module', 'names' : key, 'result' : 'success'} 
        print("d=",d) 
        data = json.dumps(d)
        PDU = packPDU(data)
        bytesWriten = 0
        while bytesWriten < len(PDU):
              bytesWriten = bytesWriten + os.write(1, PDU[bytesWriten:])
#读取从QT发过来的数据包---是一个结构体，包含图像数据，包长度，包类型
def readPDU():
    print('readPDU .....')
    pduLen, msgType, msgLen = struct.unpack('<III', os.read(0, 12))   

    bytesRead = msgLen   #取出图片大小
    print('readPDU pdulen=%d,msgType=%d,msglen=%d'%(pduLen,msgType,msgLen))
    data = ''

    if msgLen > 0:  #读完图片为止
        try:
            data = os.read(0, bytesRead)
            if len(data) == 0:
                sys.exit(0)

            bytesRead = bytesRead - len(data)

            while(len(data) < msgLen):
                recv = os.read(0, bytesRead)
                if len(data) == 0:
                    sys.exit(0)
                bytesRead = bytesRead - len(recv)
                data = data + recv
        except:
            sys.exit(0)
    #根据不同包类型，调用算法的不同接口
    if msgType == 0:    # security
        
        handleSecurity(data)
    elif msgType == 1:  # attendance
        handleAttendance(data)
    elif msgType == 2:  # loadmodule
        print('handleLoadModule111111111111111111')
        handleLoadModule(data)
    elif msgType == 3:  # regist
        handleRegist(data)
    elif msgType == 4:  # getface---regist
        handleRegistGetFace(data)
        #pass
    else:
        pass

PDU = packPDU('{"type" : "init"}')
print(PDU)
os.write(1, PDU)

while True:
    readPDU()

'''
f = open("/home/tg/shwd_ver2/imgs/aaa.jpg", "rb")
handleFaceRecognize(f.read())
f.close()
'''
