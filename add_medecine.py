import time
import random
import serial
import asyncio
import websockets
from aip import AipOcr
import datetime
import cv2
from email.mime.text import MIMEText
import smtplib
import numpy as np
import os
from PIL import Image
import RPi.GPIO as GPIO
#全局变量mess用以接收服务器端消息
mess=''
noticeLists=[]
medicineAllList=[]
global false, null, true
false = null = true = ''
# Path for face image database
path = 'dataset'
########1.服务器收发信息
#从服务器接收信息
async def skt_recv():
    uri = 'ws://118.25.110.129:3001?client_id=test'
    async with websockets.connect(uri) as websocket:
        #while True:
        global mess
        mess=await websocket.recv()
        await websocket.close()
#向服务器发送信息
async def skt_send(stri):
    uri = 'ws://118.25.110.129:3001?client_id=test'
    async with websockets.connect(uri) as websocket:
        await websocket.send(stri)
        print ('发送成功')
        await websocket.close()

########2.文字识别
# 新建一个AipOcr对象
config = {
    'appId': '18564907',
    'apiKey': 'ZFyniMBrc1PItHEXFmZC1bOl',
    'secretKey': 'yZgbiIL9GRbCzPrzW5u8kaLUVeEjeQzn'
}
client = AipOcr(**config)

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# 读取图片
def get_file_content(file_path):
    with open(file_path, 'rb') as fp:
        return fp.read()

# 识别图片里的文字
def img_to_str(image_path):
    image = get_file_content(image_path)
    # 调用通用文字识别, 图片参数为本地图片
    result = client.basicAccurate(image)
    # 结果拼接返回
    if 'words_result' in result:
        return '\n'.join([w['words'] for w in result['words_result']])


########3.语音模块
ser=serial.Serial("/dev/ttyAMA0",baudrate=9600,timeout=3.0)
#speak：我要添加药品
#response:好的，请从手机端输入药品信息


########4.recognitionFace
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

read_status=20
#########5.main
while True:
    read_status=read_status-1
    #等待接收服务器信息
    asyncio.run(skt_recv())
    print("连接成功"+mess)
#     if(mess!='1'):
#         print(mess)
    flag=mess[0:6]
    #接收到服务器端输入药品信息成功的消息
    if (flag == "addone"):
        print("addone")
        user_id=mess[6:]
        #response：小程序输入完毕，请扫描药品信息
        scan="@TextToSpeech#请扫描药品信息:$"
        ser.write(scan.encode("GBK"))
        #扫描药品信息
        #计数扫描文字，避免卡顿
        count = 20
        medicineList=[]
        allMedicine=['银杏叶片','银翘解毒丸']
        addedMedicine={'confirm':user_id}
        #response：已经添加xxx
        while True:
            count = count - 1
            ret, img = cap.read()
            img = cv2.flip(img, -1)
            cv2.imshow('video',img)
            if count == 0:
                cv2.imwrite('test.jpg',img)
                medicine_name=img_to_str('test.jpg')
                medicine_name=medicine_name.split('\n', 1 )[0]# 以换行为分隔符，分隔成两个
                #查询并添加药品
                for singleMedicine in allMedicine:
                    if medicine_name == singleMedicine:
                        dic={"medicineName":medicine_name,"boxNum":0}
                        medicineAllList.append(dic)
                        medicineList.append(medicine_name)
                        medicine_name_speak="@TextToSpeech#已经添加"+medicine_name+"$"
                        ser.write(medicine_name_speak.encode("GBK"))
                        time.sleep(1)
                #count恢复初始化
                count = 20
                #读取串口信息，如果输入结束
                item = ser.read()
                index = int.from_bytes(item, 'big')
                #判断输入是否结束，如果结束，向服务器发送信息，并退出
                if(index==4):
                    medicine_push="@TextToSpeech#请将药品放入药箱$"
                    ser.write(medicine_push.encode("GBK"))
                    #产生随机数，随机开N号箱子
                    rd = random.randint(1,4)
                    #发送信息
                    addedMedicine.update({'medicine':medicineList})
                    asyncio.run(skt_send(str(addedMedicine)))
                    #更新boxNum
                    for sigleMedicineList in medicineList:
                        for sigleMedicineAllList in medicineAllList:
                            if(sigleMedicineAllList["medicineName"]==sigleMedicineList):
                                sigleMedicineAllList["boxNum"]=rd
                                break
                    break
            k = cv2.waitKey(1) & 0xff
            if k == 27: # press 'ESC' to quit
                break
        #cap.release()
        cv2.destroyAllWindows()
        #加入控制机械部件的代码
        print("have added and sended")

    #如果接受到的信息的时间和药品等
    elif(flag=="notice"):
        print("notice")
        mess=mess[6:]
        print(mess)
        dic=eval(mess)
        noticePerson=dic["noticePerson"]
        print(noticePerson)
        noticeTimes=dic["time"]
        print(noticeTimes)
        noticeID=dic["_id"]
        noticeEmailAddr=dic["emailAddr"]
        print(noticeEmailAddr)
        noticeAcrtTime=dic["acrtTime"]
        if(noticeAcrtTime[0]=="0"):
            noticeHour=noticeAcrtTime[1]
        else:
            noticeHour=noticeAcrtTime[0]+noticeAcrtTime[1]
        if(noticeAcrtTime[3]=="0"):
            noticeMinute=noticeAcrtTime[4]
        else:
            noticeMinute=noticeAcrtTime[3]+noticeAcrtTime[4]
        print(noticeAcrtTime+","+noticeHour+","+noticeMinute)
        for block in dic["medicines"]:
            noticeEatTime=block["beforeEat"]
            print(noticeEatTime)
            noticeMedicineName=block["name"]
            print(noticeMedicineName)
            noticeMedicineNum=block["package"]
            print(noticeMedicineNum)
            noticeTable={"noticeID":noticeID,"noticeEmailAddr":noticeEmailAddr,
                            "noticeHour":noticeHour,"noticeMinute":noticeMinute,
                            "noticePerson":noticePerson,"noticeTimes":noticeTimes,
                            "noticeEatTime":noticeEatTime,
                            "noticeMedicineName":noticeMedicineName,
                            "noticeMedicineNum":noticeMedicineNum}
            noticeLists.append(noticeTable)
            noticeSpeak="@TextToSpeech#添加提醒成功$"
            ser.write(noticeSpeak.encode("GBK"))
    if(flag=="delete"):
        print("delete")
        mess1=mess[6:10]
        mess2=mess[10:]
        dic=eval(mess2)
        if(mess1=="medc"):
            deleteMedicine=dic["name"]
            medicineListIndex=0
            deleteMedicineIndex=0
            GoOn=0
            for singleMedicineAllList in medicineAllList:
                if(singleMedicineAllList["medicineName"]==deleteMedicine):
                    GoOn=1
                    noticeSpeak="@TextToSpeech#删除药品:"+deleteMedicine+"成功，请从药箱中取出药品$"
                    ser.write(noticeSpeak.encode("GBK"))
                    #记录当前索引位置
                    deleteMedicineIndex=medicineListIndex
                    #打开固定药箱
                    if(singleMedicineAllList["boxNum"]==1):
                        GPIO.setmode(GPIO.BCM)
                        GPIO.setup(5,GPIO.OUT)
                        startTime=time.time();
                        while(True):
                            endTime=time.time();
                            if(endTime-startTime>2):
                                break;
                        GPIO.output(5,GPIO.HIGH)
                        GPIO.cleanup()
                    if(singleMedicineAllList["boxNum"]==2):
                        GPIO.setmode(GPIO.BCM)
                        GPIO.setup(6,GPIO.OUT)
                        startTime=time.time();
                        while(True):
                            endTime=time.time();
                            if(endTime-startTime>2):
                                break;
                        GPIO.output(6,GPIO.HIGH)
                        GPIO.cleanup()
                    if(singleMedicineAllList["boxNum"]==3):
                        GPIO.setmode(GPIO.BCM)
                        GPIO.setup(13,GPIO.OUT)
                        startTime=time.time();
                        while(True):
                            endTime=time.time();
                            if(endTime-startTime>2):
                                break;
                        GPIO.output(13,GPIO.HIGH)
                        GPIO.cleanup()
                    if(singleMedicineAllList["boxNum"]==4):
                        GPIO.setmode(GPIO.BCM)
                        GPIO.setup(19,GPIO.OUT)
                        startTime=time.time();
                        while(True):
                            endTime=time.time();
                            if(endTime-startTime>2):
                                break;
                        GPIO.output(19,GPIO.HIGH)
                        GPIO.cleanup()
                    break    
                medicineListIndex=medicineListIndex+1
            if(GoOn==0)
                noticeSpeak="@TextToSpeech#删除失败，药品:"+deleteMedicine+"不在药箱中$"
                ser.write(noticeSpeak.encode("GBK"))
            if(GoOn==1)
                medicineAllList.pop(medicineListIndex)
            time.sleep(4)
        if(mess1=="notc"):
            deleteNoticeID=dic["_id"]
            num=0
            for noticeList in noticeLists:
                if(noticeList["noticeID"]==deleteNoticeID):
                    break
                num=num+1
            print(noticeLists[num]["noticePerson"])
            noticeLists.pop(num)
            noticeSpeak="@TextToSpeech#删除提醒成功$"
            ser.write(noticeSpeak.encode("GBK"))
    
    else:
#         print("else")
        #正常情况下，判断提醒时间
        nowtime=datetime.datetime.now()
        nowtimeDate=nowtime.date()
        for noticeList in noticeLists:
            setTime=datetime.time(int(noticeList["noticeHour"]),int(noticeList["noticeMinute"]),0,0)
            noticeTime1=datetime.datetime.combine(nowtimeDate,setTime)
            noticeTime2=noticeTime1+datetime.timedelta(seconds=10)
            emailTime1=noticeTime1+datetime.timedelta(minutes=20)
            emailTime2=emailTime1+datetime.timedelta(seconds=1)
            boxOpenTime=1
            if (nowtime > noticeTime1 and nowtime < noticeTime2):
                noticeSpeakFirst="@TextToSpeech#吃药时间到啦!$"
                ser.write(noticeSpeakFirst.encode("GBK"))
                time.sleep(2)
                noticeSpeakSecond="@TextToSpeech#"+noticeList["noticePerson"]+"应该"+noticeList["noticeEatTime"]+"吃"+noticeList["noticeMedicineNum"]+noticeList["noticeMedicineName"]+"$"
                ser.write(noticeSpeakSecond.encode("GBK"))
                if(boxOpenTime==1):
                    for sigleMedicineAllList in medicineAllList:
                        if(noticeList["noticeMedicineName"]==sigleMedicineAllList["medecineName"]):
                            #打开固定药箱
                            if(singleMedicineAllList["boxNum"]==1):
                                GPIO.setmode(GPIO.BCM)
                                GPIO.setup(5,GPIO.OUT)
                                startTime=time.time();
                                while(True):
                                    endTime=time.time();
                                    if(endTime-startTime>2):
                                        break;
                                GPIO.output(5,GPIO.HIGH)
                                GPIO.cleanup()
                            if(singleMedicineAllList["boxNum"]==2):
                                GPIO.setmode(GPIO.BCM)
                                GPIO.setup(6,GPIO.OUT)
                                startTime=time.time();
                                while(True):
                                    endTime=time.time();
                                    if(endTime-startTime>2):
                                        break;
                                GPIO.output(6,GPIO.HIGH)
                                GPIO.cleanup()
                            if(singleMedicineAllList["boxNum"]==3):
                                GPIO.setmode(GPIO.BCM)
                                GPIO.setup(13,GPIO.OUT)
                                startTime=time.time();
                                while(True):
                                    endTime=time.time();
                                    if(endTime-startTime>2):
                                        break;
                                GPIO.output(13,GPIO.HIGH)
                                GPIO.cleanup()
                            if(singleMedicineAllList["boxNum"]==4):
                                GPIO.setmode(GPIO.BCM)
                                GPIO.setup(19,GPIO.OUT)
                                startTime=time.time();
                                while(True):
                                    endTime=time.time();
                                    if(endTime-startTime>2):
                                        break;
                                GPIO.output(19,GPIO.HIGH)
                                GPIO.cleanup()
                            break
                boxOpenTime=0
                time.sleep(6)
            if(nowtime > emailTime1 and nowtime < emailTime2):#and xinhao==1):
                mail_content = "您好，系统判断您的家人没有按时吃药，请您及时提醒他们吃药"
                content = MIMEText(mail_content, 'plain', 'utf-8') # 第一个参数：邮件的内容；第二个参数：邮件内容的格式，普通的文本，可以使用:plain,如果想使内容美观，可以使用:html；第三个参数：设置内容的编码，这里设置为:utf-8
                reveivers = noticeList["noticeEmailAddr"]#1329298050@qq.com,
                content['To'] = reveivers # 设置邮件的接收者，多个接收者之间用逗号隔开
                content['From'] = str("智能药箱") # 邮件的发送者,最好写成str("这里填发送者")，不然可能会出现乱码
                content['Subject'] = "服药提醒" # 邮件的主题
                smtp_server = smtplib.SMTP_SSL("smtp.qq.com", 465) # 第一个参数：smtp服务地址（你发送邮件所使用的邮箱的smtp地址，在网上可以查到，比如qq邮箱为smtp.qq.com） 第二个参数：对应smtp服务地址的端口号
                smtp_server.login("981239736@qq.com", "tynudmffkiulbbfh") # 第一个参数：发送者的邮箱账号 第二个参数：对应邮箱账号的密码
                smtp_server.sendmail("981239736@qq.com", [reveivers], content.as_string()) # 第一个参数：发送者的邮箱账号；第二个参数是个列表类型，每个元素为一个接收者；第三个参数：邮件内容
                smtp_server.quit() # 发送完成后加上这个函数调用，类似于open文件后要跟一个close文件一样
        #正常情况下，进行人脸识别开锁
        #加一句:添加管理员:好的，现在确认主管理员身份    打开药箱:好的，请扫描管理员面部信息

#         item = ser.read()
#         index = int.from_bytes(item, 'big')
        index=1
        if(read_status==0):
            read_status=20
            item = ser.read()
            index = int.from_bytes(item, 'big')
        #如果想打开药箱
#         if(index != 7):
#             index=5
        if(index==6):
            confidenceTimes=0
            unconfidenceTimes=0
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer/trainer.yml')
            cascadePath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath);
            font = cv2.FONT_HERSHEY_SIMPLEX
            #iniciate id counter
            id = 0
            GoOn = 0
            names = ['None','Jason', 'Tom', 'Ilza', 'Z', 'W']
            minW = 0.1*cap.get(3)
            minH = 0.1*cap.get(4)
            while True:
                ret, img =cap.read()
                img = cv2.flip(img, -1) # Flip vertically
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 5,
                    minSize = (int(minW), int(minH)),
                )
                for(x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])                    # Check if confidence is less them 100 ==> "0" is perfect match 
                    if (confidence < 100):
                        id = names[id]
                        confidenceTimes=confidenceTimes+1
                        #confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        id = "unknown"
                        unconfidenceTimes=unconfidenceTimes+1
                        #confidence = "  {0}%".format(round(100 - confidence))
                    cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.imshow('camera',img)
                if(confidenceTimes>20):
                    confirmSpeak1="@TextToSpeech#身份确认，请打开药箱$"
                    ser.write(confirmSpeak1.encode("GBK"))
                    GoOn=1
                    break
                if(unconfidenceTimes>20):
                    confirmSpeak2="@TextToSpeech#身份未知$"
                    ser.write(confirmSpeak2.encode("GBK"))
                    GoOn=0
                    break
                k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
            #cap.release()
            cv2.destroyAllWindows()
            if(GoOn==1):
                index2=0
                while(True):
                    item2 = ser.read()
                    index2 = int.from_bytes(item2, 'big')
                    if(index2==7||index2==8||index2==9||index2==10)
                        break
                if(index2==7):#控制药箱打开
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(5,GPIO.OUT)
                    startTime=time.time();
                    while(True):
                        endTime=time.time();
                        if(endTime-startTime>2):
                            break;
                    GPIO.output(5,GPIO.HIGH)
                    GPIO.cleanup()
                if(index2==8):#打开2号箱
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(6,GPIO.OUT)
                    startTime=time.time();
                    while(True):
                        endTime=time.time();
                        if(endTime-startTime>2):
                            break;
                    GPIO.output(6,GPIO.HIGH)
                    GPIO.cleanup()
                if(index2==9):#打开3号箱
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(13,GPIO.OUT)
                    startTime=time.time();
                    while(True):
                        endTime=time.time();
                        if(endTime-startTime>2):
                            break;
                    GPIO.output(13,GPIO.HIGH)
                    GPIO.cleanup()
                if(index2==10):#打开4号箱
                    GPIO.setmode(GPIO.BCM)
                    GPIO.setup(19,GPIO.OUT)
                    startTime=time.time();
                    while(True):
                        endTime=time.time();
                        if(endTime-startTime>2):
                            break;
                    GPIO.output(19,GPIO.HIGH)
                    GPIO.cleanup()

        if(index==5):#添加管理员
#             confirmSpeak1="@TextToSpeech#请扫描系统管理员人脸信息$"
#             ser.write(confirmSpeak1.encode("GBK"))
            confidenceTimes=0
            unconfidenceTimes=0
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer/trainer.yml')
            cascadePath = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath);
            font = cv2.FONT_HERSHEY_SIMPLEX
            #iniciate id counter
            id = 0
            names = ['None', 'Jason', 'Tom', 'Ilza', 'Z', 'W']
            minW = 0.1*cap.get(3)
            minH = 0.1*cap.get(4)
            while True:
                ret, img =cap.read()
                img = cv2.flip(img, -1) # Flip vertically
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 5,
                    minSize = (int(minW), int(minH)),
                )
                for(x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])                    # Check if confidence is less them 100 ==> "0" is perfect match 
                    if (confidence < 100):
                        id = names[id]
                        confidenceTimes=confidenceTimes+1
                        #confidence = "  {0}%".format(round(100 - confidence))
                    else:
                        id = "unknown"
                        unconfidenceTimes=unconfidenceTimes+1
                        #confidence = "  {0}%".format(round(100 - confidence))
                    cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.imshow('camera',img)
                if(confidenceTimes>20):
                    confirmSpeak1="@TextToSpeech#系统管理员身份确认，请添加新的管理员人脸信息$"
                    ser.write(confirmSpeak1.encode("GBK"))
                    GoOn1=1
                    time.sleep(6)
                    break
                if(unconfidenceTimes>20):
                    confirmSpeak2="@TextToSpeech#身份验证失败$"
                    ser.write(confirmSpeak2.encode("GBK"))
                    GoOn1=0
                    break
                k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
            #cap.release()
            cv2.destroyAllWindows()
            if(GoOn1==0):
                break
            #########1.以下是收集人脸数据
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face_id = input('\n enter user id end press <return> ==>  ')
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            count = 0
            confirmSpeak1="@TextToSpeech#正在收集人脸信息，请稍候$"
            ser.write(confirmSpeak1.encode("GBK"))
            while(True):
                ret, img = cap.read()
                img = cv2.flip(img, -1) # flip video image vertically
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('image', img)
                k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 30: # Take 30 face sample and stop video
                     break
            # Do a bit of cleanup
            print("\n [INFO] Exiting Program and cleanup stuff")
            #cam.release()
            cv2.destroyAllWindows()
            
            
            #########2.以下是训练模型
            confirmSpeak1="@TextToSpeech#正在训练数据集，请稍候$"
            ser.write(confirmSpeak1.encode("GBK"))
            time.sleep(2)
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
            # function to get the images and label data
            print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces,ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            # Save the model into trainer/trainer.yml
            recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
            # Print the numer of faces trained and end program
            print("done")
            #print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
            confirmSpeak1="@TextToSpeech#管理员添加完毕!$"
            ser.write(confirmSpeak1.encode("GBK"))