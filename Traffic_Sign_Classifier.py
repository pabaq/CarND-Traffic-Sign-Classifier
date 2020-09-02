import pandas as pd

from utilities import *
from model import *
from layers import *

# Load data
train_data = x_train, y_train = load_data("data/train.p")
valid_data = x_valid, y_valid = load_data("data/valid.p")
test_data = x_test, y_test = load_data("data/test.p")
new_test_data = x_test_new, y_test_new = load_data(
    "additional_signs/additional_signs.p")

# Load the sign names into a pandas DataFrame
sign_names = pd.read_csv("signnames.csv", index_col=0)

if __name__ == "__main__":

    plot_distributions(y_train, y_valid, y_test)
    plot_images(x_train, y_train)
    plot_images(x_train, y_train, prep=True)
    plot_new_images(x_test_new, y_test_new)
    plot_new_images(x_test_new, y_test_new, prep=True)

    # Basic LeNet
    flag = False
    if flag:
        tf.reset_default_graph()
        lenet = Model('LeNet-5')
        lenet.compile(layers=lenet5_rgb,
                      initializer='RandomNormal',
                      activate_dropout=False)

        loss, train_acc, valid_acc = lenet.train(
            train_data=(x_train, y_train),
            valid_data=(x_valid, y_valid),
            optimizer='GradientDescent',
            learning_rate=0.001,
            batch_size=128,
            epochs=30)

        collector = Collector()
        collector.collect(lenet, loss, train_acc, valid_acc)
        plot_pipeline("LeNet-5_Basic", collector)

    # Optimizer Pipeline
    flag = False
    if flag:

        # Parameters
        layers = lenet5_rgb
        initializer = 'RandomNormal'
        optimizers = ['GradientDescent', 'Adam', 'Adagrad']
        learning_rate = 0.001
        batch_size = 128
        epochs = 30

        collector = Collector()
        for optimizer in optimizers:
            print(f"\nOptimizer = {optimizer}")

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=False)
            loss, train_acc, valid_acc = lenet.train(
                train_data=train_data,
                valid_data=valid_data,
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet, loss, train_acc, valid_acc)

        plot_pipeline("LeNet-5_Optimizer", collector)

    # Normalization pipeline
    flag = False
    if flag:

        # Parameters
        initializer = 'RandomNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        batch_size = 128
        epochs = 30

        normilization_kwargs = [
            OrderedDict(scale=None, clahe=False),
            OrderedDict(scale='norm', clahe=False),
            OrderedDict(scale='std', clahe=False),
            OrderedDict(scale=None, clahe=True),
            OrderedDict(scale='std', clahe=True)
        ]

        lenet_layers = [
            lenet5_rgb,
            lenet5_rgb,
            lenet5_rgb,
            lenet5_single_channel,
            lenet5_single_channel
        ]

        collector = Collector()
        for kwargs, layers in zip(normilization_kwargs, lenet_layers):

            print(f"\npreprocess(x, scale='{kwargs['scale']}', "
                  f"clahe={kwargs['clahe']})")

            x_train_pre = preprocess(x_train, **kwargs)
            x_valid_pre = preprocess(x_valid, **kwargs)

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=False)
            loss, train_acc, valid_acc = lenet.train(
                train_data=(x_train_pre, y_train),
                valid_data=(x_valid_pre, y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet, loss, train_acc, valid_acc, **kwargs)

        plot_pipeline("LeNet-5_Normalization", collector)

    # Initializer pipeline
    flag = False
    if flag:

        # Parameters
        layers = lenet5_single_channel
        initializers = ["RandomNormal",
                        "TruncatedNormal",
                        "HeNormal",
                        "XavierNormal"]
        optimizer = 'Adam'
        learning_rate = 0.001
        batch_size = 128
        epochs = 30

        collector = Collector()
        for initializer in initializers:
            print(f"\nInitializer = {initializer}")

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=False)
            loss, train_acc, valid_acc = lenet.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet, loss, train_acc, valid_acc)

        plot_pipeline("LeNet-5_Initializer", collector)

    # Learning rates pipeline
    flag = False
    if flag:

        # Parameters
        layers = lenet5_single_channel
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        batch_size = 128
        epochs = 30

        collector = Collector()
        for learning_rate in learning_rates:
            print(f"\nLearning rate = {learning_rate}")

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=False)
            loss, train_acc, valid_acc = lenet.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet, loss, train_acc, valid_acc)

        plot_pipeline("LeNet-5_Learning_Rates", collector)

    # Batch size pipeline
    flag = False
    if flag:

        # Parameters
        layers = lenet5_single_channel
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        batch_sizes = [32, 64, 128, 256]
        epochs = 30

        collector = Collector()
        for batch_size in batch_sizes:
            print(f"\nBatch size = {batch_size}")

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=False)
            loss, train_acc, valid_acc = lenet.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet, loss, train_acc, valid_acc)

        plot_pipeline("LeNet-5_Batch_Sizes", collector)

    # Dropout pipeline
    flag = False
    if flag:

        # Parameters
        layers = lenet5_single_channel
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_probs = [1.0, 0.75, 0.5, 0.25]
        batch_size = 128
        epochs = 30

        collector = Collector()
        for keep_prob in keep_probs:
            print(f"\nkeep_prob = {keep_prob}")

            tf.reset_default_graph()
            lenet = Model('LeNet-5')
            lenet.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=True)
            loss, train_acc, valid_acc = lenet.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                keep_prob=keep_prob)

            collector.collect(lenet, loss, train_acc, valid_acc)

        plot_pipeline("LeNet_Dropout", collector)

    # Convolution depth pipeline
    flag = False
    if flag:

        # Parameters
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_prob = 0.5
        batch_size = 128
        epochs = 30
        multiplicators = [1, 3, 6, 9]

        collector = Collector()
        for multi in multiplicators:
            lenet5_single_channel_extended_conv_depth = [
                # in: 32 x 32 x 1
                Conv2d(name="conv1",
                       shape=(5, 5, 1, 6 * multi),
                       strides=[1, 1, 1, 1],
                       padding="VALID",
                       activation="Relu"),
                # 28 x 28 x (6 | 18 | 36 | 54)
                Pool(name="pool1",
                     shape=(1, 2, 2, 1),
                     strides=(1, 2, 2, 1),
                     padding="VALID",
                     pooling_type="MAX"),
                # 14 x 14 x (6 | 18 | 36 | 54)
                Conv2d(name="conv2",
                       shape=(5, 5, 6, 16 * multi),
                       strides=[1, 1, 1, 1],
                       padding="VALID",
                       activation="Relu"),
                # 10 x 10 x (16 | 48 | 96 | 144)
                Pool(name="pool2",
                     shape=(1, 2, 2, 1),
                     strides=(1, 2, 2, 1),
                     padding="VALID",
                     pooling_type="MAX"),
                # 5 x 5 x (16 | 48 | 96 | 144) = 400 | 1200 | 2400 | 3600
                Flatten(size=400 * multi),
                # 400 | 1200 | 2400 | 3600
                Dense(name="fc3",
                      shape=(400 * multi, 120 * multi),
                      activation="Relu",
                      dropout=True),
                # shape: (120 | 360 | 720 | 1080) (with dropout)
                Dense(name="fc4",
                      shape=(120 * multi, 84 * multi),
                      activation="Relu",
                      dropout=True),
                # shape: (84 | 252 | 504 | 756)
                Dense(name="fc5",
                      shape=(84 * multi, 43),
                      activation=None)]  # out: 43

            print(f"\ndepth multiplicator = {multi}")

            tf.reset_default_graph()
            lenet_extdepth = Model(f'LeNet-5')
            lenet_extdepth.compile(
                layers=lenet5_single_channel_extended_conv_depth,
                initializer=initializer,
                activate_dropout=True)
            loss, train_acc, valid_acc = lenet_extdepth.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                keep_prob=keep_prob,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(lenet_extdepth, loss, train_acc, valid_acc,
                              multi=multi)

        plot_pipeline("LeNet-5_Extendended_Conv_Depth", collector)

    # Additional convolution layers
    flag = False
    if flag:

        # Parameters
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_prob = 0.5
        batch_size = 128
        epochs = 30

        names = ["LeNet-5", "LeNet-6a", "LeNet-6b"]
        layers_list = [lenet5_single_channel, lenet6a_layers, lenet6b_layers]

        collector = Collector()
        for name, layers in zip(names, layers_list):
            print(f"\n{name}")

            tf.reset_default_graph()
            model = Model(f'{name}')
            model.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=True)
            loss, train_acc, valid_acc = model.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                keep_prob=keep_prob,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(model, loss, train_acc, valid_acc)

        plot_pipeline("LeNet_Additional_Layers", collector)

    # Concatenating layers
    flag = False
    if flag:

        # Parameters
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_prob = 0.5
        batch_size = 128
        epochs = 30

        names = ["LeNet-5",
                 "LeNet-6a_concat_c2c3",
                 "LeNet-6a_concat_p2c3",
                 "LeNet-6b_concat_c2c3",
                 "LeNet-6b_concat_p2c3"]
        layers_list = [lenet5_single_channel,
                       lenet6a_layers_concat_c2c3,
                       lenet6a_layers_concat_p2c3,
                       lenet6b_layers_concat_c2c3,
                       lenet6b_layers_concat_p2c3]

        collector = Collector()
        for name, layers in zip(names, layers_list):
            print(f"\n{name}")

            tf.reset_default_graph()
            model = Model(f'{name}')
            model.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=True)
            loss, train_acc, valid_acc = model.train(
                train_data=(preprocess(x_train), y_train),
                valid_data=(preprocess(x_valid), y_valid),
                optimizer=optimizer,
                learning_rate=learning_rate,
                keep_prob=keep_prob,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(model, loss, train_acc, valid_acc)

        plot_pipeline("LeNet_Concat", collector)

    # Variants
    flag = False
    if flag:

        # Parameters
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_prob = 0.5
        batch_size = 128
        epochs = 30

        names = [
            "LeNet-5a_concat_p1p2",
            "LeNet-5b_concat_p1p2",
            # "MyNet-5c"
        ]
        layers_list = [
            lenet5a_concat_p1p2,
            lenet5b_concat_p1p2,
            # lenet5_layers_with_dropout_single_channel
            # mynet5a,
            # mynet5b,
            # mynet5c
        ]

        trains = [
            (preprocess(x_train), y_train),
            (preprocess(x_train), y_train),
        ]

        valids = [
            (preprocess(x_valid), y_valid),
            (preprocess(x_valid), y_valid),
        ]

        collector = Collector()
        for name, layers, train, valid in zip(names, layers_list, trains,
                                              valids):
            print(f"{name}")

            tf.reset_default_graph()
            model = Model(f'{name}')
            model.compile(layers=layers,
                          initializer=initializer,
                          activate_dropout=True)
            loss, train_acc, valid_acc = model.train(
                train_data=train,
                valid_data=valid,
                optimizer=optimizer,
                learning_rate=learning_rate,
                keep_prob=keep_prob,
                epochs=epochs,
                batch_size=batch_size)

            collector.collect(model, loss, train_acc, valid_acc)

        plot_pipeline("Variants", collector)

    # Final Model
    flag = False
    if flag:
        # Parameters
        initializer = 'TruncatedNormal'
        optimizer = 'Adam'
        learning_rate = 0.001
        keep_prob = 0.5
        batch_size = 128
        epochs = 50

        tf.reset_default_graph()
        lenet = Model('LeNet-5_Final')
        lenet.compile(layers=lenet5_single_channel,
                      initializer=initializer,
                      activate_dropout=True)

        loss, train_acc, valid_acc = lenet.train(
            train_data=(preprocess(x_train), y_train),
            valid_data=(preprocess(x_valid), y_valid),
            optimizer=optimizer,
            learning_rate=learning_rate,
            keep_prob=keep_prob,
            epochs=epochs,
            batch_size=batch_size,
            save=True)

        collector = Collector()
        collector.collect(lenet, loss, train_acc, valid_acc)
        plot_pipeline("LeNet-5_Final", collector)

    # Evaluate test set
    flag = False
    data = preprocess(x_test_new), y_test_new
    if flag:
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            lenet = Model()
            lenet.restore(checkpoint="models/LeNet-5_Final.ckpt-28")
            acc = lenet.evaluate(*data)
            print(f"Accuracy: {acc:.4f}")

    # Predict new test images
    flag = False
    if flag:
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            lenet = Model()
            lenet.restore(checkpoint="models/LeNet-5_Final.ckpt-28")
            acc = lenet.evaluate(preprocess(x_test_new), y_test_new)
            print(f"\nAccuracy: {acc:.4f}\n")
            top_k_probs, top_k_preds = lenet.predict(
                preprocess(x_test_new), k=3)

        plot_predictions(x_test_new, y_test_new, top_k_probs, top_k_preds,
                         sign_names)
