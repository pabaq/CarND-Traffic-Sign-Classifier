from model import *

lenet5_rgb = [
    # in: 32 x 32 x 3
    Conv2d(name="conv1",
           shape=(5, 5, 3, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16 = 400
    Flatten(size=400),
    # 400
    Dense(name="fc3",
          shape=(400, 120),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc4",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc5",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet5_single_channel = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16 = 400
    Flatten(size=400),
    # 400
    Dense(name="fc3",
          shape=(400, 120),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc4",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc5",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6a_layers = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(5, 5, 16, 400),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 1 x 1 x 400
    Flatten(size=400),
    # 400 (with dropout)
    Dense(name="fc4",
          shape=(400, 120),
          activation="Relu",
          dropout=True),
    # 120 (with dropout)
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6b_layers = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(3, 3, 16, 50),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 3 x 3 x 50
    Flatten(size=450),
    # 450 (with dropout)
    Dense(name="fc4",
          shape=(450, 120),
          activation="Relu",
          dropout=True),
    # 120 (with dropout)
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6a_layers_concat_c2c3 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(5, 5, 16, 400),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 1 x 1 x 400
    # conv2: 10 x 10 x 16 -> 1600
    # conv3: 1 x 1 x 400 -> 400
    # concat: 1600 + 400 = 2000
    Concat(layers=["conv2", "conv3"]),
    # 2000
    Dense(name="fc4",
          shape=(2000, 120),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6a_layers_concat_p2c3 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(5, 5, 16, 400),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 1 x 1 x 400
    # pool2: 5 x 5 x 16 -> 400
    # conv3: 1 x 1 x 400 -> 400
    # concat: 400 + 400 = 800
    Concat(layers=["pool2", "conv3"]),
    # 800
    Dense(name="fc4",
          shape=(800, 120),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6b_layers_concat_c2c3 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(3, 3, 16, 50),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 3 x 3 x 50
    # conv2: 10 x 10 x 16 -> 1600
    # conv3: 3 x 3 x 50 -> 450
    # concat: 1600 + 450 = 2050
    Concat(layers=["conv2", "conv3"]),
    # 2050 (with dropout)
    Dense(name="fc4",
          shape=(2050, 120),
          activation="Relu",
          dropout=True),
    # 120 (with dropout)
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet6b_layers_concat_p2c3 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16
    Conv2d(name="conv3",
           shape=(3, 3, 16, 50),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 3 x 3 x 50
    # pool2: 5 x 5 x 16 -> 400
    # conv3: 3 x 3 x 50 -> 450
    # concat: 400 + 450 = 850
    Concat(layers=["pool2", "conv3"]),
    # 850 (with dropout)
    Dense(name="fc4",
          shape=(850, 120),
          activation="Relu",
          dropout=True),
    # 120 (with dropout)
    Dense(name="fc5",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc6",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet5a_concat_p1p2 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16 = 400
    # pool1: 14 x 14 x 6 = 1176
    # pool2: 5 x 5 x 16 = 400
    Concat(layers=["pool1", "pool2"]),
    # 1576
    Dense(name="fc3",
          shape=(1576, 120),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc4",
          shape=(120, 84),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc5",
          shape=(84, 43),
          activation=None)]  # out: 43

lenet5b_concat_p1p2 = [
    # in: 32 x 32 x 1
    Conv2d(name="conv1",
           shape=(5, 5, 1, 6),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 28 x 28 x 6
    Pool(name="pool1",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 14 x 14 x 6
    Conv2d(name="conv2",
           shape=(5, 5, 6, 16),
           strides=[1, 1, 1, 1],
           padding="VALID",
           activation="Relu"),
    # 10 x 10 x 16
    Pool(name="pool2",
         shape=(1, 2, 2, 1),
         strides=(1, 2, 2, 1),
         padding="VALID",
         pooling_type="MAX"),
    # 5 x 5 x 16 = 400
    # pool1: 14 x 14 x 6 = 1176
    # pool2: 5 x 5 x 16 = 400
    Concat(layers=["pool1", "pool2"]),
    # 1576
    Dense(name="fc3",
          shape=(1576, 256),
          activation="Relu",
          dropout=True),
    # 120
    Dense(name="fc4",
          shape=(256, 128),
          activation="Relu",
          dropout=True),
    # 84
    Dense(name="fc5",
          shape=(128, 43),
          activation=None)]  # out: 43