# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageToArrayPreprocessor
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Cách dùng Trainning_model_shallownet.py -d datasets/animals

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="Nhập folder chứa tập dữ liệu")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))

# Bước 1. Chuẩn bị dữ liệu
# Khởi tạo tiền xử lý ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
iap = ImageToArrayPreprocessor() # Gọi hàm để chuyển ảnh sang mảng

# Nạp dataset từ đĩa rồi co dãn mức xám của pixel trong vùng [0,1]
print("[INFO] Nạp ảnh...")

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Chia tách dữ liệu vào 02 tập, training: 75% và testing: 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
# Chuyển dữ liệu nhãn ở số nguyên vào biểu diễn dưới dạng vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Bước 2. Xây dựng model mạng
# Tạo bộ tối ưu hóa cho mô hỉnh
opt = SGD(learning_rate=0.005)

# Khởi tạo mô hình mạng, biên dịch mô hình
print("[INFO] Tạo mô hình...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Bước 3. trainning mạng
print("[INFO] training mạng ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=32, epochs=100, verbose=1)
model.save("model.hdf5")



# Bước 4. Đánh giá mạng
print("[INFO] Đánh giá mạng...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat", "dog", "panda"]))

# Vẽ kết quả trainning: sự mất mát (loss) quá trình trainning và độ chính xác (accuracy)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="Mất mát khi trainning")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="Độ chính xác khi trainning")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Biểu đồ hiển thị mất mát trong Training và độ chính xác")
plt.xlabel("Epoch #")
plt.ylabel("Mất mát/Độ chính xác")
plt.legend()
plt.show()