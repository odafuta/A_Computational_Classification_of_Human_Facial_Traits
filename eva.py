from ultralytics import YOLO
import os
import joblib
from main import imgEva

pca = joblib.load('./pca_110.pkl')
svm = joblib.load('./svm_c1.5_com110.pkl')
def eva(type):
    model = YOLO('best.pt')
    pca = joblib.load('./pca_110.pkl')
    sr = 0
    yr = 0

    for i in range(1, 11):
        if i < 10:
            image_path = './test/' + str(type) + '_human_0' + str(i) + '.jpg'
        else:
            image_path = './test/' + str(type) + '_human_' + str(i) + '.jpg'
        results = model(image_path)

        pred = results[0]
        class_index = pred.probs.top1
        class_name = pred.names[class_index]
        if class_name in ('fox', 'tiger'):
            class_name = 'wild'
        id = imgEva(image_path)
        if id in ('fox_like', 'tiger_like'):
            id = 'wild'
        r = 'Real animal: ' + str(type)
        s = ' -> svm eva: ' + id
        y = ' -> yolo eva: ' + class_name
        n = '\n'
        if class_name == type:
            yr -=- 1
        if id == type + '_like':
            sr -=- 1
        file.write(r + s + y + n)
    file.write('svm rate: ' + str(sr / 10) + ' -> yolo rate: ' + str(yr / 10) + n)
with open("eva_res.txt", 'w') as file:
    eva('cat')
    eva('dog')
    eva('wild')
