
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# 사전 학습된 모델 불러오기
model = VGG16(weights='imagenet')

# 분류를 위한 이미지 불러오기
image = cv2.imread("D:/vscode/cat.jpg")
# 이미지 리사이징
image = cv2.resize(image,dsize=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
image = preprocess_input(image)

# class predict
yhat = model.predict(image)

# class decodeing
label = decode_predictions(yhat)

# class alloc
label = label[0][0]

print("%s (%.2f%%)" % (label[1], label[2]*100))