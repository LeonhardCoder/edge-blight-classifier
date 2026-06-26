CLASSES = ["healthy", "early_blight", "late_blight"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
CRITICAL_CLASS = "late_blight"

IMG_SIZE = 224
RESIZE_SIZE = int(round(1.14 * IMG_SIZE))  # 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WARMUP_RUNS = 50
SEED = 42
