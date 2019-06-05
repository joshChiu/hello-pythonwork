
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import keras.backend.tensorflow_backend as KTF

# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

#进行配置，使用30%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.layers import Input, Embedding, LSTM, Dense

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import scipy.misc
import math
import time
from datetime import datetime as dt



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



# PSNR值(峰值信號雜訊比 (Peak Signal to Noise Ratio)) RGB
def psnr(original, contrast):
    [width, height, color] = original.shape
    error = np.zeros((width, height, color), dtype=int)  # uint8'
    error = (original[0:width, 0:height, 0:color] - contrast[0:width, 0:height, 0:color]) ** 2

    print("error[0,0:5,:]=", error[0, 0:5, :])
    print("original[0,0:5,:]=", original[0, 0:5, :])
    print("contrast[0,0:5,:]=", contrast[0, 0:5, :])
    print("original[0,0:5,:]-contrast[0,0:5,:]=", "error[0,0:5,:]=", original[0, 0:5, :] - contrast[0, 0:5, :])

    print("np.sum(error).info=", np.sum(error), "/width*height*color= ", width * height * color)
    mse = np.sum(error) / (width * height * color)

    print("error.info", error.dtype, error.shape)
    print("255**2 = ", 255 ** 2, "& mse=", mse)
    print("255**2/mse=", 255 ** 2 / mse)

    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return (20 * math.log10(PIXEL_MAX)) - (10 * math.log10(mse))




# from datetime import datetime as dt
# print(dt.today())


#x=np.random.randint(0,1000,size=1)
# x=np.random.randint(0,100,size=5)
# print(x)

imF = Image.open('baboon.png')
print('imF.shape1= ',imF.size[0], imF.size[1])
imAll=np.array(imF)  #打开图像并转化为数字矩阵
print('imAll.shape2= ',imAll.shape)


a=8
# Full Resize image 預處理
[widthA,heightA,color]=imAll.shape
resize_image = imF.resize((widthA//a, heightA//a), Image.BILINEAR)
resize_image.save("ResizeD4.png")  # 2=256 ,4=128 , 8=64


#Read image
im = Image.open('ResizeD4.png')  # baboonR1010 , baboon.png
print('im_Image.shape= ',im.size[0], im.size[1])
imog=np.array(im)  #打开图像并转化为数字矩阵

# Resize image
k=2
[widthG,heightG,color]=imog.shape
resize_image = im.resize((widthG//k, heightG//k), Image.BILINEAR)
resize_image.save("Resize.jpg")


im2 = Image.open('Resize.jpg')
imog2=np.array(im2)
pix = im2.load()
width = im2.size[0]
height = im2.size[1]
print('im2_Resize.jpg= k ',k,im2.size[0], im2.size[1] )


fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title("Prior image Size") # title
plt.xlabel("resize_image=256/%s =%s"%(a,im.size[0]))

plt.axis('on')

plt.subplot(1, 2, 2)
plt.imshow(im2)
plt.title("DownSample Size") # title
plt.xlabel("Star Day= %s "%(dt.today()))
plt.show()
plt.axis('on')

print('////////////////------------Prior image finish---------////////////////////////////////////')

m=2
n=3
xyLable=np.empty(shape=[0, m])
rgbLable=np.empty(shape=[0, n])
nom=255
for x in range(width):
    for y in range(height):
        r, g, b = pix[x, y]
        xy = np.array([[x, y]], np.int)
        rgb = np.array([[r, g , b]], np.int)

        xyLable =np.vstack((xyLable, xy))
        rgbLable=np.vstack((rgbLable, rgb))
#        print(x)

xyLable.shape

print("xyLable.shape",xyLable.shape)
print("rgbLable.shape",rgbLable.shape)

print('xyLable[0:30]=',xyLable[0:30])
print('rgbLable[0:30]=',rgbLable[0:30])

# index = [i for i in range(len(xyLable))]
# np.random.shuffle(index)
data = xyLable
label= rgbLable


v=im2.size[0]*im2.size[1]
x_train=data[0:v] #
# x_test=data[30000:]   #

y_train=label[0:v] #
# y_test=label[30000:]   #

# print('x_train.shape,y_train.shape)= ',x_train.shape,y_train.shape)
# print('x_test.shape,y_test.shape)= ',x_test.shape,y_test.shape)
# #
# ################################# _mlp.py

x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
x_train /= width
# x_test /= height

y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')
y_train /= 254.0
# y_test /= 255


print(x_train.shape, '= X train samples')
# print(x_test.shape[0], '= X test samples')

print(y_train.shape,'y train samples')

print('x_train[:3] =',x_train[:30])
print('y_train[:3] =',y_train[:30])

print('len(y_train)= ',len(y_train))
#
# convert class vectors to binary class matrices
# 線性迴歸分析模型預測
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape, 'categorical after')

tStart = time.time()#計時開始

model = Sequential()
model.add(Dense(units=10, # hide=10
                activation='relu',
                kernel_initializer='normal',
                input_shape=(2,)))
# model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='sigmoid'))

model.summary()

# print(model.summary())


# batch_size=1
model.compile(loss='mse',       # categorical_crossentropy' , mse
              optimizer=RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
              # optimizer='sgd')
print('/////////////////////////////////////////////////////////////////////////')
print(y_train[0:10])
print('/////////////////////////////////////////////////////////////////////////')

epochs=5000
c=100
history = model.fit(x_train[0:c], y_train[0:c], # x_train[0:600]
                    batch_size=128,
                    shuffle=True,               # 要不要打乱数据 (打乱比较好)
                    epochs=epochs,
                    # num_workers=2,              # 多线程来读数据
                    verbose=1)
                    #validation_data=(x_train[0:c], y_train[0:c]))

end = time.time()
elapsed = end - tStart
print ("Time taken: ", ('%.3f'%(elapsed/60)), "mins.") # (seconds/1 , min /60)

# score = model.evaluate(x_train, y_train, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# aa = np.array([[1, 1.5]], np.float)
# bb = np.array([[0, 0.5]], np.float)
#
# cc =np.vstack((aa, bb))
# RGB = model.predict(cc)
# print(RGB)
#
# # RGB = model.predict([0,0.5])
# # RGB = model.predict([0,1])
# # plt.scatter(x_test, y_test)
# # plt.plot(x_test, Y_pred)
# # plt.show()
#
# # predict = model.predict(X[:200,:])
#
#
# xyInput=np.empty(shape=[0, m])
# rgbPredict=np.empty(shape=[0, n])
#
# k=1
# # g=np.empty((width*k,height*k),3)
# for x in range(width*k):
#     for y in range(height*k):
#         xinp = (x//k) + (x % k) / k
#         yinp = (y//k) + (y % k) / k
#         xy = np.array([[xinp, yinp]], np.float)
#         xyInput = np.vstack((xyInput, xy))
#
# xyInput = xyInput.astype('float32')
# xyInput /= width
#
# print('<===============================================>')
# print("k ,width & height =" ,k,width,height)
# print("xyInput[:5]",xyInput[:5])
# print('xyInput.shape',xyInput.shape)
# print('<-1---------------------------------------------->')
#
# g= model.predict(xyInput)
# print("shape & model.predict",g.shape,g[:5])
# g=np.reshape(g, (width*k,height*k, 3), order='F')  # C ,F
# g *= 255//1
# print("shape & model.predictReshape",g.shape,g[0,0:5])
#
# print('<-2---------------------------------------------->')
# p_rgbLable=g.astype(int) # 'uint8'
# print("p_rgbLable.Bicubic",p_rgbLable.dtype,p_rgbLable.shape,p_rgbLable[0,0:5])
# print("rgbLable.oriang",rgbLable.dtype,rgbLable.shape,rgbLable[0:5])
#
#
# print('<-3---------------------------------------------->')
# # PIL.save( "InterBaboonE5k_D2-T.png", "PNG" )
#
# im3 = Image.open('InterBaboonE5k_D2-T.png')
# im3Arr=np.array(im3)  #打开图像并转化为数字矩阵
#
#
# d=psnr(imog2,p_rgbLable)  # imog2
# print("PSNR",'%.5f'%(d))
# PSNR='%.3f'%(d)
# print("epochs ; batch_size =",epochs ,",",batch_size )
# end = time.time()
# elapsed = end - tStart
# print ("Time taken: ", ('%.3f'%(elapsed/60)), "mins.") # (seconds/1 , min /60)
#
# fig, ax = plt.subplots(nrows=1, ncols=2)
# plt.subplot(1, 2, 1)
# # plt.set_title('area1')
# plt.imshow(im2) #im2
# plt.title("DownSample Size") # title
# plt.xlabel("Time taken: min \n : %s hours" %'%.2f'%(elapsed/3600))#x mins
# # plt.ylabel("y's;abel")#y轴上的名字
# plt.axis('on')
#
#
# plt.subplot(1, 2, 2)
# plt.imshow(p_rgbLable)
# plt.title("Predictive Image,k=%s"%(k) )# title
# plt.xlabel("epochs= %s & batch_size= %s \n PSNR is %s "%(epochs, batch_size,PSNR))#x轴上的名字
# # plt.xlabel("My long label with $\Sigma_{C}$ math \n continues here")
# # plt.ylabel("PSNR is %s"%'%.3f'%(d))#y轴上的名字
# plt.show()
# plt.axis('on')
#
# # ("The slope of the line is %s and the max of y is %s"%(k, max(y)))