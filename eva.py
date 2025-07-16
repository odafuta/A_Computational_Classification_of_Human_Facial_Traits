from ultralytics import YOLO
import os
import joblib
from main import imgEva

model = YOLO('best.pt')
pca = joblib.load('./pca_110.pkl')
svm = joblib.load('./svm_c1.5_com110.pkl')
imgEva()

image_path = 'eva.jpg'
results = model(image_path)

pred = results[0]
class_index = pred.probs.top1
class_name = pred.names[class_index]

print(class_name)
