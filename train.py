import yaml
from ultralytics import YOLO

if __name__ == '__main__':
    # Define YAML configuration
    data = {
        'path': 'dataset',
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 1,
        'names': ['Crack']
    }

    # Save to dataset.yaml
    with open('dataset.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("dataset.yaml created successfully!")

    # Load a pretrained YOLOv8 model
    model = YOLO('yolov8m-seg.pt')

    # Train the model
    model.train(
        data='dataset.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        name='Cracks_Segmentation_yolov8',
        save=True,
        save_period=-1,
        patience=20,
        val=False
    )