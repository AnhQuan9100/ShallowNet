# import the necessary packages
from preprocessing import imagetoarraypreprocessor
from preprocessing import simplepreprocessor
from datasets import simpledatasetloader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import argparse

#  Load_model_shallownet.py -d <folder chứa ảnh phân loại>
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="Nhập vào folder để phân loại")
#ap.add_argument("-m", "--model", required=True,help="Nhập vào pre-trained model")
args = vars(ap.parse_args())

# Khởi tạo danh sách nhãn
classLabels = ["cat", "dog", "panda"]

#Lấy danh sách các hình ảnh trong tập dữ liệu sau đó lấy mẫu ngẫu nhiên
# ảnh theo chỉ số để đưa vào đường dẫn hình ảnh
print("[INFO] Đang nạp ảnh mẫu để phân lớp...")
imagePaths = np.array(list(paths.list_images(args["dataset"]))) #xác định số file trong dataset
idxs = np.random.randint(0, len(imagePaths), size=(10,)) # Trả về 10 idxs ngẫu nhiên
imagePaths = imagePaths[idxs]
print(imagePaths)

sp = simplepreprocessor.SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng


# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# load the pre-trained network
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("model.hdf5")

# make predictions on the images
print("[INFO] Đang dự đoán để phân lớp...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

# Lặp qua tất cả các file ảnh trong imagePaths
# Nạp ảnh ví dụ --> Vẽ dự đoán --> Hiển thị ảnh
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "label: {}".format(classLabels[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)







