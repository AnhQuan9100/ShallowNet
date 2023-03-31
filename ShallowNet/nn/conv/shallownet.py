# import các gói thư viện cần thiết
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # khởi tạo mô hình
        # height, width, depth: tương ứng 3 chiều của dữ liệu ảnh đầu vào
        # classes: tổng số lớp mà mạng dự đoán, phụ thuộc vào dữ liệu
        # Dữ liệu Animal: 3 lớp; Dữ liệu CIFAR-10: 10 lớp

        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Định nghĩa mạng CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",input_shape = inputShape))
        model.add(Activation("relu"))

        # Bộ phân lớp sử dụng hàm softmax
        model.add(Flatten())       # Chuyển thành vector
        model.add(Dense(classes))  # Định nghĩa Full Connected layer
        model.add(Activation("softmax"))   # Kích hoạt hàm softmax để phân lớp
        # Trả về model kiến trúc mạng
        return model

