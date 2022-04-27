This project is part of Udacity's [Self-Driving-Car Nanodegree][Course]. The project 
resources and build instructions can be found [here][Project].

## Traffic sign classification with CNN

In this project we will train and validate several CNN architectures with the goal of 
classifying traffic sign images using the [German Traffic Sign Dataset][Dataset]. 
Subsequently, we will try out the best architecture on random images of traffic signs 
that we collected from the web.

The complete code can be found in the following modules:

- [``utilities.py``][utilities]: helper functions for the loading, preprocessing and 
  plotting of the data and the investigation results.
- [``model.py``][model]: defintion of a [``Model``][modelclass] class, making the 
  investigation of several network architectures and parameter variations more comfortable.
- [``layers.py``][layers]: the layer defintions of the investigated networks. 

It is also possible to walk through the project using the [Traffic Sign Classifier jupyter 
notebook][notebook] in this repository.

The project consists of the following steps:

1. Exploration of the available data set.
2. Initial investigation of the basic ``LeNet-5`` network.
3. Investigation of the influence of several model (hyper) parameters and model 
   architectures
4. Training and testing of the final network.
5. Predictions on unknown traffic sign images collected from the internet.

## Data exploration
The distributions of the training, validation and test data sets of the [German Traffic 
Sign Dataset][Dataset] are comparable, however, they are far from being uniform. 
Some classes are present much more frequent then others.

![][histogram]

Let's have a look on the traffic sign images and classes. The images are of shape 32x32 and 
in the RGB color space. They were taken under varying light conditions. Some can easily be 
recognized, others are even hard to notice. We will preprocess these images in one of the 
subsequent investigations pipelines.

![][samples]

| ClassId | SignName                                        |
|:------- |:----------------------------------------------- | 	
|0 	      |  Speed limit (20km/h)                                
|1 	      |  Speed limit (30km/h)                                
|2 	      |  Speed limit (50km/h)
|3 	      |  Speed limit (60km/h)
|4 	      |  Speed limit (70km/h)
|5 	      |  Speed limit (80km/h)
|6 	      |  End of speed limit (80km/h)
|7 	      |  Speed limit (100km/h)
|8 	      |  Speed limit (120km/h)
|9 	      |  No passing
|10       |  No passing for vehicles over 3.5 metric tons
|11       |  Right-of-way at the next intersection
|12       |  Priority road
|13       |  Yield
|14       |  Stop
|15       |  No vehicles
|16       |  Vehicles over 3.5 metric tons prohibited
|17       |  No entry
|18       |  General caution
|19       |  Dangerous curve to the left
|20       |  Dangerous curve to the right
|21       |  Double curve
|22       |  Bumpy road
|23       |  Slippery road
|24       |  Road narrows on the right
|25       |  Road work
|26       |  Traffic signals
|27       |  Pedestrians
|28       |  Children crossing
|29       |  Bicycles crossing
|30       |  Beware of ice/snow
|31       |  Wild animals crossing
|32       |  End of all speed and passing limits
|33       |  Turn right ahead
|34       |  Turn left ahead
|35       |  Ahead only
|36       |  Go straight or right
|37       |  Go straight or left
|38       |  Keep right
|39       |  Keep left
|40       |  Roundabout mandatory
|41       |  End of no passing
|42       |  End of no passing by vehicles over 3.5 metric ...


## The LeNet-5 network
![][lenet5]

The basic ``LeNet-5`` network presented in [Gradient-Based Learning Applied to Document 
Recognition][LenetPaper] by Yann LeCun et al. is used as a starting point for the 
following investigations. We will build it by making use of the [``Model``][modelclass] 
class defined in [``model.py``][model] and the [``lenet5_rgb``][lenet5_rgb] layers defined 
in [``layers.py``][layers]. The basic ``LeNet-5`` archictecture is defined as follows: 

```python
# layers.py

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
          activation=None)]  # out: 43 logits
```

It will take as input an image of shape ``32 x 32 x 3`` and its last layer will output the 
43 traffic traffic sign ``logits``.

The implemented methods of the [``Model``][model] class allow the [``compiling``][model_compile],
[``training``][model_train] and subsequent [``evaluation``][model_evaluate] of the network. Let's
build basic ``LeNet-5`` and train it on the traffic sign samples with the folowing first set of
parameters:

- **Training Variables Initializer**: Random Normal Initializer (with the defaults: mean=0, stddev=0.1)
- **Dropout**: We will set the dropout in the Dense Layers inactive for the first training
- **Training and Validation Data**: We will use unprocessed data for the first training
- **Optimizer**: Gradient Descent
- **Learning Rate**: 0.001
- **Mini Batch Size**: 128
- **Epochs**: 30

We will vary these parameters in the subsequent analyses to see their effects on the network's performance.

````python
from model import Model
from utilities import Collector, plot_pipeline

# Basic LeNet
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
````
```
Epoch 10/30:   Train Loss: 3.7002   Train Acc: 0.0498   Valid Acc: 0.0451  
Epoch 20/30:   Train Loss: 3.1954   Train Acc: 0.1768   Valid Acc: 0.1567  
Epoch 30/30:   Train Loss: 2.5000   Train Acc: 0.3713   Valid Acc: 0.3472  
```

![][basic]

It can be seen that the Gradient Descent Optimizer makes a quite slow progress.
Next, let us check how the model behaves by varying some of its (hyper) parameters.

## Model parameter analysis

### Optimizers
In the first investigation we will vary the optimizers using ``Gradient Descent``, 
``Adam`` and ``Adagrad``.

````python
# Parameters
layers = lenet5_rgb
initializer = 'RandomNormal'
optimizers = ['GradientDescent', 'Adam', 'Adagrad']
learning_rate = 0.001
batch_size = 128
epochs = 30

collector = Collector()
for optimizer in optimizers:
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(lenet, loss, train_acc, valid_acc)

plot_pipeline("LeNet-5_Optimizer", collector)
````
```
Optimizer = GradientDescent
Epoch 10/30:   Train Loss: 3.6991   Train Acc: 0.0502   Valid Acc: 0.0451
Epoch 20/30:   Train Loss: 3.1851   Train Acc: 0.1845   Valid Acc: 0.1630
Epoch 30/30:   Train Loss: 2.3898   Train Acc: 0.4253   Valid Acc: 0.3859

Optimizer = Adam
Epoch 10/30:   Train Loss: 0.0971   Train Acc: 0.9758   Valid Acc: 0.8585
Epoch 20/30:   Train Loss: 0.0524   Train Acc: 0.9900   Valid Acc: 0.8952
Epoch 30/30:   Train Loss: 0.0303   Train Acc: 0.9864   Valid Acc: 0.8810

Optimizer = Adagrad
Epoch 10/30:   Train Loss: 2.5005   Train Acc: 0.4063   Valid Acc: 0.3422
Epoch 20/30:   Train Loss: 1.8471   Train Acc: 0.5405   Valid Acc: 0.4605
Epoch 30/30:   Train Loss: 1.4762   Train Acc: 0.6237   Valid Acc: 0.5317
```

![][optimizer]

The ``Adam`` optimizer is doing a pretty good job. We will use it as the default 
optimizer for the rest of the project. 

### Input Data Normalization
Next let's preprocess the input data. The function [``preprocess``][preprocess] in 
[``utilities.py``][utilities] does this job for us. It performs scaling and contrast limited 
adaptive histogram equalization ([CLAHE][Clahe]) on the input images.

````python
def preprocess(x, scale='std', clahe=True):
    """ Preprocess the input features.

    Args:
        x:
            batch of input images
        clahe:
            perform a contrast limited histogram equalization before scaling
        scale:
            'normalize' the data into a range of 0 and 1 or 'standardize' the
            data to zero mean and standard deviation 1

    Returns:
        The preprocessed input features, eventually reduced to single channel
    """

    if clahe is True:
        x = np.array([np.expand_dims(rgb2clahe(img), 2) for img in x])

    x = np.float32(x)

    if scale is not None and scale.lower() in ['norm', 'normalize']:
        x /= x.max()
    elif scale is not None and scale.lower() in ['std', 'standardize']:
        mean, std = x.mean(), x.std()
        x = (x - mean) / (std + np.finfo(float).eps)

    return x
````

The output on the class samples shown above looks as follows 

![][preprocessing]

We can see that the edges and the content of the signs get highlighted independently of the 
lightning situation in the original images. Let's investigate the effect of each of the 
preprocessing parameters on the performance of the network. Since the output of the CLAHE 
operation is a gray image, we will introduce [``lenet5_single_channel``][lenet5_single_channel]
layers that can handle single channel input images. All other layer parameters stay the same. 

````python
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

    print(f"\npreprocess(x, scale='{kwargs['scale']}', clahe={kwargs['clahe']})")

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
````
````
preprocess(x, scale=None, clahe=False)
Epoch 10/30:   Train Loss: 0.0853   Train Acc: 0.9827   Valid Acc: 0.8816
Epoch 20/30:   Train Loss: 0.0374   Train Acc: 0.9872   Valid Acc: 0.8710
Epoch 30/30:   Train Loss: 0.0232   Train Acc: 0.9929   Valid Acc: 0.8975

preprocess(x, scale=norm, clahe=False)
Epoch 10/30:   Train Loss: 0.0509   Train Acc: 0.9891   Valid Acc: 0.9143
Epoch 20/30:   Train Loss: 0.0194   Train Acc: 0.9973   Valid Acc: 0.9068
Epoch 30/30:   Train Loss: 0.0100   Train Acc: 0.9994   Valid Acc: 0.9206

preprocess(x, scale=std, clahe=False)
Epoch 10/30:   Train Loss: 0.0289   Train Acc: 0.9913   Valid Acc: 0.9190
Epoch 20/30:   Train Loss: 0.0036   Train Acc: 0.9997   Valid Acc: 0.9388
Epoch 30/30:   Train Loss: 0.0114   Train Acc: 0.9972   Valid Acc: 0.9454

preprocess(x, scale=None, clahe=True)
Epoch 10/30:   Train Loss: 0.0719   Train Acc: 0.9812   Valid Acc: 0.8762
Epoch 20/30:   Train Loss: 0.0280   Train Acc: 0.9869   Valid Acc: 0.8891
Epoch 30/30:   Train Loss: 0.0259   Train Acc: 0.9953   Valid Acc: 0.9075

preprocess(x, scale=std, clahe=True)
Epoch 10/30:   Train Loss: 0.0177   Train Acc: 0.9947   Valid Acc: 0.9415
Epoch 20/30:   Train Loss: 0.0033   Train Acc: 0.9964   Valid Acc: 0.9467
Epoch 30/30:   Train Loss: 0.0000   Train Acc: 1.0000   Valid Acc: 0.9546
````

![][normalization]

If the input images are both standardized and pass the CLAHE operation, the network seems 
to show the best performance. We will keep this as the default image preprocessing pipeline.  

### Learning Parameters Initializer
Next we will have a look on the influence of the Variable initializer. 

````python
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(lenet, loss, train_acc, valid_acc)

plot_pipeline("LeNet-5_Initializer", collector)
````
```
Initializer = RandomNormal
Epoch 10/30:   Train Loss: 0.0179   Train Acc: 0.9862   Valid Acc: 0.9179
Epoch 20/30:   Train Loss: 0.0081   Train Acc: 0.9992   Valid Acc: 0.9456
Epoch 30/30:   Train Loss: 0.0047   Train Acc: 0.9996   Valid Acc: 0.9508

Initializer = TruncatedNormal
Epoch 10/30:   Train Loss: 0.0149   Train Acc: 0.9976   Valid Acc: 0.9510
Epoch 20/30:   Train Loss: 0.0126   Train Acc: 0.9989   Valid Acc: 0.9626
Epoch 30/30:   Train Loss: 0.0018   Train Acc: 0.9996   Valid Acc: 0.9630

Initializer = HeNormal
Epoch 10/30:   Train Loss: 0.0142   Train Acc: 0.9981   Valid Acc: 0.9420
Epoch 20/30:   Train Loss: 0.0108   Train Acc: 0.9941   Valid Acc: 0.9274
Epoch 30/30:   Train Loss: 0.0000   Train Acc: 1.0000   Valid Acc: 0.9535

Initializer = XavierNormal
Epoch 10/30:   Train Loss: 0.0103   Train Acc: 0.9951   Valid Acc: 0.9497
Epoch 20/30:   Train Loss: 0.0135   Train Acc: 0.9924   Valid Acc: 0.9351
Epoch 30/30:   Train Loss: 0.0000   Train Acc: 1.0000   Valid Acc: 0.9658
````

![][initializer]

Well, it doesn't seem to make much of a difference which of the shown initializers we use.
We'll keep the ``TruncatedNormal`` Initializer as standard for the following analyses.


### Learning Rates
One of the most important parameters is the learning rate of the optimizer. Let's have a look.

````python
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(lenet, loss, train_acc, valid_acc)

plot_pipeline("LeNet-5_Learning_Rates", collector)
````
````
Learning rate = 0.1
Epoch 10/30:   Train Loss: 3.4891   Train Acc: 0.0569   Valid Acc: 0.0544
Epoch 20/30:   Train Loss: 3.4905   Train Acc: 0.0552   Valid Acc: 0.0544
Epoch 30/30:   Train Loss: 3.4907   Train Acc: 0.0552   Valid Acc: 0.0544

Learning rate = 0.01
Epoch 10/30:   Train Loss: 0.1236   Train Acc: 0.9818   Valid Acc: 0.9397
Epoch 20/30:   Train Loss: 0.1735   Train Acc: 0.9766   Valid Acc: 0.9265
Epoch 30/30:   Train Loss: 0.1845   Train Acc: 0.9840   Valid Acc: 0.9415

Learning rate = 0.001
Epoch 10/30:   Train Loss: 0.0146   Train Acc: 0.9881   Valid Acc: 0.9290
Epoch 20/30:   Train Loss: 0.0103   Train Acc: 0.9976   Valid Acc: 0.9526
Epoch 30/30:   Train Loss: 0.0160   Train Acc: 0.9972   Valid Acc: 0.9574

Learning rate = 0.0001
Epoch 10/30:   Train Loss: 0.2365   Train Acc: 0.9398   Valid Acc: 0.8658
Epoch 20/30:   Train Loss: 0.1028   Train Acc: 0.9768   Valid Acc: 0.9045
Epoch 30/30:   Train Loss: 0.0555   Train Acc: 0.9876   Valid Acc: 0.9190
````

![][learning]

A learning rate that is too big, leads to no learning at all. It seems that a learning rate of 
``0.001`` makes a good starting choice. We will keep this rate for the rest of the project.

  
### Batch size
Next the mini batch size.

````python
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
        batch_size=batch_size,  
        verbose=1)

    collector.collect(lenet, loss, train_acc, valid_acc)

plot_pipeline("LeNet-5_Batch_Sizes", collector)
````
````
Batch size = 32
Epoch 10/30:   Train Loss: 0.0174   Train Acc: 0.9965   Valid Acc: 0.9562
Epoch 20/30:   Train Loss: 0.0132   Train Acc: 0.9974   Valid Acc: 0.9542
Epoch 30/30:   Train Loss: 0.0060   Train Acc: 0.9974   Valid Acc: 0.9562

Batch size = 64
Epoch 10/30:   Train Loss: 0.0147   Train Acc: 0.9889   Valid Acc: 0.9454
Epoch 20/30:   Train Loss: 0.0156   Train Acc: 0.9969   Valid Acc: 0.9490
Epoch 30/30:   Train Loss: 0.0049   Train Acc: 0.9994   Valid Acc: 0.9560

Batch size = 128
Epoch 10/30:   Train Loss: 0.0135   Train Acc: 0.9949   Valid Acc: 0.9494
Epoch 20/30:   Train Loss: 0.0108   Train Acc: 0.9917   Valid Acc: 0.9383
Epoch 30/30:   Train Loss: 0.0059   Train Acc: 0.9995   Valid Acc: 0.9626

Batch size = 256
Epoch 10/30:   Train Loss: 0.0248   Train Acc: 0.9940   Valid Acc: 0.9308
Epoch 20/30:   Train Loss: 0.0008   Train Acc: 1.0000   Valid Acc: 0.9474
Epoch 30/30:   Train Loss: 0.0002   Train Acc: 1.0000   Valid Acc: 0.9454
````

![][batch]

The smaller the batch size, the slower the learning. In this case, it doesn't seem to have 
that much of an influence on the performance. We will stay with a batch size of 128, since 
it allows us a little bit faster training.


### Dropout
Next, let's finally activate the dropout layers. As we can see above, dropout layers are 
introduced in the 3rd and 4th Dense layer of the ``Lenet-5`` network. We will activate them 
and have a look on the influence of the ``keep_prob`` probability. 

````python
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
        keep_prob=keep_prob, 
        verbose=1)

    collector.collect(lenet, loss, train_acc, valid_acc)

plot_pipeline("LeNet_Dropout", collector)
````
````
keep_prob = 1.0
Epoch 10/30:   Train Loss: 0.0134   Train Acc: 0.9970   Valid Acc: 0.9515
Epoch 20/30:   Train Loss: 0.0082   Train Acc: 0.9979   Valid Acc: 0.9463
Epoch 30/30:   Train Loss: 0.0001   Train Acc: 1.0000   Valid Acc: 0.9676

keep_prob = 0.75
Epoch 10/30:   Train Loss: 0.0688   Train Acc: 0.9958   Valid Acc: 0.9580
Epoch 20/30:   Train Loss: 0.0324   Train Acc: 0.9993   Valid Acc: 0.9723
Epoch 30/30:   Train Loss: 0.0219   Train Acc: 0.9999   Valid Acc: 0.9705

keep_prob = 0.5
Epoch 10/30:   Train Loss: 0.2520   Train Acc: 0.9863   Valid Acc: 0.9590
Epoch 20/30:   Train Loss: 0.1453   Train Acc: 0.9943   Valid Acc: 0.9669
Epoch 30/30:   Train Loss: 0.1063   Train Acc: 0.9983   Valid Acc: 0.9732

keep_prob = 0.25
Epoch 10/30:   Train Loss: 1.2227   Train Acc: 0.8211   Valid Acc: 0.7821
Epoch 20/30:   Train Loss: 1.0344   Train Acc: 0.8741   Valid Acc: 0.8454
Epoch 30/30:   Train Loss: 0.9459   Train Acc: 0.8859   Valid Acc: 0.8587
````

![][dropout]

A drop out rate of 50% gives us the best results. We'll set it as default for the rest of 
the project.


### Convolution depth
Next we will leverage the depth of the convolution layers by a multiplicator. This will 
lead to much more learning parameters. Let's see if it is worth.

````python
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
        # 120 | 360 | 720 | 1080
        Dense(name="fc4",
              shape=(120 * multi, 84 * multi),
              activation="Relu",
              dropout=True),
        # 84 | 252 | 504 | 756
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(lenet_extdepth, loss, train_acc, valid_acc,
                      multi=multi)

plot_pipeline("LeNet-5_Extendended_Conv_Depth", collector)
````
````
depth multiplicator = 1
Epoch 10/30:   Train Loss: 0.2535   Train Acc: 0.9851   Valid Acc: 0.9601
Epoch 20/30:   Train Loss: 0.1420   Train Acc: 0.9959   Valid Acc: 0.9737
Epoch 30/30:   Train Loss: 0.1078   Train Acc: 0.9980   Valid Acc: 0.9751

depth multiplicator = 3
Epoch 10/30:   Train Loss: 0.0465   Train Acc: 0.9997   Valid Acc: 0.9766
Epoch 20/30:   Train Loss: 0.0231   Train Acc: 0.9996   Valid Acc: 0.9798
Epoch 30/30:   Train Loss: 0.0160   Train Acc: 0.9997   Valid Acc: 0.9789

depth multiplicator = 6
Epoch 10/30:   Train Loss: 0.0264   Train Acc: 0.9997   Valid Acc: 0.9807
Epoch 20/30:   Train Loss: 0.0136   Train Acc: 1.0000   Valid Acc: 0.9764
Epoch 30/30:   Train Loss: 0.0151   Train Acc: 1.0000   Valid Acc: 0.9794

depth multiplicator = 9
Epoch 10/30:   Train Loss: 0.0217   Train Acc: 0.9998   Valid Acc: 0.9800
Epoch 20/30:   Train Loss: 0.0165   Train Acc: 0.9994   Valid Acc: 0.9796
Epoch 30/30:   Train Loss: 0.0104   Train Acc: 1.0000   Valid Acc: 0.9796
````

![][convdepth] 

The network seems to benefit from the extended depth. However, the increase in performance 
is bought by an explosion of paramaters that need to be trained. We will stay with the basic
convolution depth, since it shows a good performance with a fraction of parameters.


### Addational Convolution layer
Let us investigate if an addtional 3rd convolution layer gives us a better compromise between
parameter and performance increase. The new layers are defined in [``layers.py``][layers]. 
The difference of the [``lenet6a_layers``][lenet6a_layers] and 
[``lenet6b_layers``][lenet6a_layers] is the convolution filter (``5x5`` vs ``3x3``) and the
convolution depth (``400`` vs ``50``). They were chosen in a way that a comparable amount of
parameters are flattened before entering the Dense layers.

````python
# lenet6a_layers 
...
# 5 x 5 x 16
Conv2d(name="conv3",
       shape=(5, 5, 16, 400),
       strides=[1, 1, 1, 1],
       padding="VALID",
       activation="Relu"),
# 1 x 1 x 400
Flatten(size=400),
...

# lenet6b_layers 
...
# 5 x 5 x 16
Conv2d(name="conv3",
       shape=(3, 3, 16, 50),
       strides=[1, 1, 1, 1],
       padding="VALID",
       activation="Relu"),
# 3 x 3 x 50 = 450
Flatten(size=450),
...
````
````python
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(model, loss, train_acc, valid_acc)

plot_pipeline("LeNet_Additional_Layers", collector)
````
````
LeNet-5
Epoch 10/30:   Train Loss: 0.2538   Train Acc: 0.9859   Valid Acc: 0.9626
Epoch 20/30:   Train Loss: 0.1423   Train Acc: 0.9955   Valid Acc: 0.9680
Epoch 30/30:   Train Loss: 0.1102   Train Acc: 0.9980   Valid Acc: 0.9769

LeNet-6a
Epoch 10/30:   Train Loss: 0.1090   Train Acc: 0.9952   Valid Acc: 0.9560
Epoch 20/30:   Train Loss: 0.0491   Train Acc: 0.9990   Valid Acc: 0.9562
Epoch 30/30:   Train Loss: 0.0256   Train Acc: 0.9995   Valid Acc: 0.9549

LeNet-6b
Epoch 10/30:   Train Loss: 0.2093   Train Acc: 0.9864   Valid Acc: 0.9546
Epoch 20/30:   Train Loss: 0.1119   Train Acc: 0.9967   Valid Acc: 0.9617
Epoch 30/30:   Train Loss: 0.0824   Train Acc: 0.9984   Valid Acc: 0.9662
````

![][convlayer] 

The ``Lenet-5`` network still shows a better performance with fewer parameters. 


### Concatenating Layers
Lastly we will have a look on the effect of concatenating layers as shown in 
[Traffic sign recognition with multi-scale Convolutional Networks][LecunPaper]. The output 
of the 2nd and 3rd convolutional layers are concatenated and led into the Dense layer for 
classification. 

![][lecun]

In this investigation we will use the LeNet-6a and LeNet-6b networks shown above to perform 
this kind of concatenation. We will have a look on following variants.  

Concatenation of the outputs of
- the 2nd and 3rd convolutional layers
- the 2nd pooling and 3rd convolutional layer

As always the layer defintions can be found in [``layers.py``][layers].


````python
# Concatenation of 2nd and 3rd convolutional layers 
...
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
...

# Concatenation of 2nd pooling and 3rd convolutional layer
...
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
...
````
````python
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
        batch_size=batch_size, 
        verbose=1)

    collector.collect(model, loss, train_acc, valid_acc)

plot_pipeline("LeNet_Concat", collector)
````
````
LeNet-5
Epoch 10/30:   Train Loss: 0.2559   Train Acc: 0.9850   Valid Acc: 0.9610
Epoch 20/30:   Train Loss: 0.1449   Train Acc: 0.9961   Valid Acc: 0.9723
Epoch 30/30:   Train Loss: 0.1100   Train Acc: 0.9983   Valid Acc: 0.9730

LeNet-6a_concat_c2c3
Epoch 10/30:   Train Loss: 0.1545   Train Acc: 0.9952   Valid Acc: 0.9560
Epoch 20/30:   Train Loss: 0.0643   Train Acc: 0.9993   Valid Acc: 0.9669
Epoch 30/30:   Train Loss: 0.0479   Train Acc: 0.9997   Valid Acc: 0.9642

LeNet-6a_concat_p2c3
Epoch 10/30:   Train Loss: 0.1399   Train Acc: 0.9954   Valid Acc: 0.9590
Epoch 20/30:   Train Loss: 0.0692   Train Acc: 0.9995   Valid Acc: 0.9649
Epoch 30/30:   Train Loss: 0.0464   Train Acc: 0.9995   Valid Acc: 0.9746

LeNet-6b_concat_c2c3
Epoch 10/30:   Train Loss: 0.1805   Train Acc: 0.9950   Valid Acc: 0.9560
Epoch 20/30:   Train Loss: 0.0972   Train Acc: 0.9991   Valid Acc: 0.9698
Epoch 30/30:   Train Loss: 0.0725   Train Acc: 0.9996   Valid Acc: 0.9717

LeNet-6b_concat_p2c3
Epoch 10/30:   Train Loss: 0.2171   Train Acc: 0.9918   Valid Acc: 0.9576
Epoch 20/30:   Train Loss: 0.1192   Train Acc: 0.9975   Valid Acc: 0.9642
Epoch 30/30:   Train Loss: 0.0886   Train Acc: 0.9991   Valid Acc: 0.9637
````

![][concat]

As with the previous shown architectures the extended complexity does not lead to an 
additional benefit in performance. 


## The final network
Since the investigated architectures did not show a benefit over the ``LeNet-5`` network, 
we will use the latter with the adjusted parameters shown in the previous sections for the final 
training and testing. The final training will be performed for 50 epochs. 

````python
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
````
![][final]

### Evaluation of the test set
````python
tf.reset_default_graph()
with tf.Session(config=config) as session:
    lenet = Model()
    lenet.restore(checkpoint="models/LeNet-5_Final.ckpt-47")
    acc = lenet.evaluate(preprocess(x_test), y_test)
    print(f"Accuracy: {acc:.4f}")
````
````
Accuracy: 0.9583
````

The evalution of the unseen test set leads to an accuracy of **95.83%**.

## Predictions on unknown images
Now let's see how the network performs on traffic sign images gathered from in the internet. 

![][newimages]

````python
# Predict new test images
tf.reset_default_graph()
with tf.Session(config=config) as session:
    lenet = Model()
    lenet.restore(checkpoint="models/LeNet-5_Final.ckpt-47")
    acc = lenet.evaluate(preprocess(x_test_new), y_test_new)
    print(f"\nAccuracy on new test signs: {acc * 100:.2f}%\n")            
    top_k_probs, top_k_preds = lenet.predict( preprocess(x_test_new), k=3)

plot_predictions(x_test_new, y_test_new, top_k_probs, top_k_preds, sign_names)
````
````
Accuracy on new test signs: 70.45%
````

The accuracy is quite low compared to the original test set, since some of the images were 
chosen to be edge cases and to challenge the network.

![][predictions]

## Observations and improvements
- In some cases the network is able to correctly predict partly occluded or polluted signs.
- The network does a good job if the signs are centered, viewed from the front and do fill a 
  major part of the area.
- It has its problems with signs that are located outside of the center, viewed from a 
  perspective or are far away.
- The latter problem could be reduced by extending the training data with augmented data.
- As can be seen by the miss prediction of the traffic signals sign, the network could 
  enventually benefit from color information. The pipeline chosen in this project uses a 
  colorless input. Seeing only gray the traffic lights share great similarity with the general 
  caution sign.
- The last two stop signs were taken from the paper [Robust Physical-World Attacks on Deep 
  Learning Visual Classification][Attack] which handles the vulnerablity of neural networks to small-magnitude perturbations added to the input. The network poorly falls on one of these 
  examples.
- Although the network shows its uncertainty with the unknown images, a plausibility check 
  using additional available information could lead to a greater robustness of the predictions.


## References
[[1]][LenetPaper] Yann LeCun, Leon Bottou, Y. Bengio, and Patrick Haffner (1998). Gradient-Based
Learning Applied to Document Recognition. *Proceedings of the IEEE* **86** (11): 2278-2324

[[2]][LecunPaper] Pierre Sermanet and Yann LeCun. Traffic sign recognition with multi-scale 
Convolutional Networks. *International Joint Conference on Neural Networks, San Jose, CA, 
United States*, 2011

[[3]][Attack] Kevin Eykholt et al. Robust Physical-World Attacks on Deep Learning Models. 
*CVPR 2018*


[Course]: https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
[Project]: https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
[Dataset]: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
[LenetPaper]: https://www.researchgate.net/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition
[Clahe]: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
[LecunPaper]: https://www.researchgate.net/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks 
[Attack]: https://arxiv.org/abs/1707.08945 


[notebook]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb
[utilities]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/utilities.py
[model]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/model.py
[modelclass]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/model.py#L177
[layers]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/layers.py
[lenet5_rgb]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/layers.py#L3
[model_compile]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/model.py#L185
[model_train]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/model.py#L207
[model_evaluate]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/model.py#L345
[preprocess]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/utilities.py#L89
[lenet5_single_channel]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/layers.py#L45
[lenet6a_layers]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/layers.py#L87
[lenet6b_layers]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/blob/master/layers.py#L135

[histogram]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/histograms.png "Histograms"
[samples]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/class_samples.png "Traffic sign classes"
[preprocessing]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/class_samples_preprocessed.png "Preprocessing"
[lenet5]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5.png "LeNet-5"
[basic]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Basic.png "Basic LeNet-5"
[optimizer]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Optimizer.png "Optimizer"
[normalization]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Normalization.png "Normalization"
[initializer]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Initializer.png "Initializer"
[learning]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Learning_Rates.png "Learning Rates"
[batch]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Batch_Sizes.png "Batch size"
[dropout]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet_Dropout.png "Dropout"
[convdepth]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Extendended_Conv_Depth.png "Convolution Depth"
[convlayer]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet_Additional_Layers.png "Additional Convolution Layer"
[lecun]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeCun.jpg "Traffic Sign Recognition with Multi-Scale Convolutional Networks"
[concat]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet_Concat.png "Concatenating Layers"
[final]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/LeNet-5_Final.png "Final Network"
[newimages]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/new_signs.png "New Signs"
[predictions]: https://github.com/pabaq/CarND-Traffic-Sign-Classifier/raw/master/images/predictions.png "Predictions"