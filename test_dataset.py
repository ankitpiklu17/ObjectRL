from utils.dataset import VOCDataset

dataset = VOCDataset("/home/ankit/datasets/VOC/images/train2007")

img, path = dataset.sample()

print("Image path:", path)

label_path = path.replace("images", "labels").replace(".jpg", ".txt")
print("Label path:", label_path)