from ultralytics import YOLO

def main():
    model = YOLO('yolo11n-cls.pt')
    model.train(
        data = 'af_data/',
        epochs = 50,
        imgsz = 512,
        batch = 32
    )


    results = model('eva.jpg')
    print(results[0].probs)
if __name__ == '__main__':
    main()