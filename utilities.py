import os
import pickle
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.style.use('seaborn')


class Collector:
    def __init__(self):
        self.losses = []
        self.train_accs = []
        self.valid_accs = []
        self.train_pars = []
        self.titles = []

    def collect(self, model, losses, train_accs, valid_accs, **kwargs):
        self.losses.append(losses)
        self.train_accs.append(train_accs)
        self.valid_accs.append(valid_accs)
        self.titles.append(self.create_title(model, **kwargs))

    @staticmethod
    def create_title(model, **kwargs):
        title = f"{model.name} ({model.trainable_parameters} pars)    "
        train_pars = model.recent_train_pars
        initializer = train_pars.pop('initializer')
        optimizer = train_pars.pop('optimizer')
        lr = train_pars.pop('learning_rate')
        keep_prob = train_pars.pop('keep_prob')
        title += f"  Initializer = {initializer}  "
        title += f"  {optimizer}(learning_rate = {lr})  "
        if model.dropout_active is True:
            title += f"  keep_prob = {keep_prob}  "
        if kwargs:
            train_pars.update(kwargs)
        for k, v in train_pars.items():
            title += f"  {k} = {v}  "
        return title


def load_data(pickle_file):
    """ Load the data from given path.

    Args:
        pickle_file: pickle file of the requested data
    Returns:
        A tuple of (features, labels) for the given `pickle_file`
    """

    with open(pickle_file, mode='rb') as f:
        data = pickle.load(f)

    return data['features'], data['labels']


def rgb2clahe(rgb_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Perform a contrast limited adaptive histogram equalization on given image.

    The clahe operation is performed on the grayscale version of the given rgb
    frame.

    Args:
        rgb_img:
            current undistorted rgb frame
        clip_limit:
            threshold for contrast limiting
        tile_grid_size:
            size of the grid for the histogram equalization. The image will be
            divided into equally sized rectangular tiles. tile_grid_size defines
            the number of tiles in row and column.
    Returns:
        a gray image as result of the application of clahe
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    gray_clahe = clahe.apply(gray)
    return gray_clahe


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


def plot_distributions(y_train, y_valid, y_test):
    """ Create histograms for the train, validation and test sample labels.

    Args:
        y_train: train sample labels
        y_valid: validation sample labels
        y_test: test sample labels
    """

    labelsets = dict(training=y_train, validation=y_valid, test=y_test)
    n_data = [len(y_train), len(y_valid), len(y_test)]
    n_total = sum(n_data)

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), dpi=100)

    for i, (setname, labelset) in enumerate(labelsets.items()):
        labels, counts = np.unique(labelset, return_counts=True)
        ax = axes[i]
        ax.bar(labels, counts, align='center', width=1)
        n = n_data[i]
        pct = n / n_total * 100
        ax.set_title(f"{setname.capitalize()} samples ({n}, {pct:2.1f}%)",
                     size=22)
        ax.set_xlabel("ClassId", size=16)
        ax.set_ylabel("count", size=16)
        ax.set_xticks(labels)
        ax.set_xlim(left=-1, right=43)
        ax.tick_params("both", labelsize=16)

    plt.tight_layout()
    fig.savefig("./images/histograms")


def plot_images(x, y, prep=False):
    """ Plot one image of each class of the given data sample.

    Args:
        x: train, validation or test images
        y: corresponding labels
        prep: if True the image will pass the preprocessing pipeline
    """

    if prep:
        x = preprocess(x, scale='std', clahe=True)

    rows = 5
    cols = 9
    fig, axes = plt.subplots(rows, cols, figsize=(16, 11), dpi=100)

    labels, indices = np.unique(y, return_index=True)
    for label, index in zip(labels, indices):
        ax = axes[label // cols, label % cols]
        ax.set_title(label, size=20)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if prep:
            ax.imshow(x[index].reshape(32, 32), cmap='gray')
        else:
            ax.imshow(x[index])

    axes[-1, -2].set_visible(False)
    axes[-1, -1].set_visible(False)

    plt.tight_layout()
    name = "class_samples_preprocessed" if prep else "class_samples"
    fig.savefig(f"./images/{name}")


def plot_new_images(x, y, prep=False):
    """ Plot one image of each class of the given data sample.

    Args:
        x: train, validation or test images
        y: corresponding labels
        prep: if True the image will pass the preprocessing pipeline
    """

    if prep:
        x = preprocess(x, scale='std', clahe=True)

    cols = 9
    rows = np.ceil(x.shape[0] / cols).astype(int)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 11), dpi=100)

    for i, (image, label) in enumerate(zip(x, y)):
        ax = axes[i // cols, i % cols]
        ax.set_title(label, size=20)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if prep:
            ax.imshow(image.reshape(32, 32), cmap='gray')
        else:
            ax.imshow(image)

    for i in range(1, rows * cols - x.shape[0] + 1):
        axes[-1, -i].set_visible(False)

    plt.tight_layout()
    name = "new_signs_preprocessed" if prep else "new_signs"
    fig.savefig(f"./images/{name}")


def plot_pipeline(name, collector):
    """  """

    losses = collector.losses
    train_accs = collector.train_accs
    valid_accs = collector.valid_accs
    titles = collector.titles

    epochs = np.arange(1, len(losses[0]) + 1)

    rows = len(titles)
    height = len(titles) ** 1.2 / rows * 11
    fig, axes = plt.subplots(rows, 1, figsize=(16, height), dpi=100)

    for i, title in enumerate(titles):

        loss = losses[i]
        train_acc = train_accs[i]
        valid_acc = valid_accs[i]

        ax = axes[i] if rows > 1 else axes
        ax.set_title(f"{title}", size=16, pad=30)
        ax.set_yticks(np.linspace(0, 4, 5))
        ax.set_ylim(0, 4)
        ax.set_ylabel("Loss", size=16)
        plt.xticks(fontsize=12)
        loss = ax.plot(epochs, np.squeeze(loss), color='k',
                       label='Training Loss')
        ax.tick_params(axis='both', which='major', labelsize=12)

        ax2 = ax.twinx()
        ax2.grid(None)
        ax2.set_yticks(np.linspace(0, 1, 5))
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Accuracy", size=16)
        train = ax2.plot(epochs, np.squeeze(train_acc), color='r',
                         label='Training Accuracy')

        # Maximum validation accuracy
        y_max = np.max(valid_acc)
        xpos = valid_acc.index(y_max)
        x_max = epochs[xpos]
        valid = ax2.plot(
            epochs, np.squeeze(valid_acc), color='b',
            label=f'Validation Accuracy (max: {y_max:.3f} @ epoch {x_max})')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        # Legend
        lines = loss + train + valid
        keys = [line.get_label() for line in lines]
        ax.legend(lines, keys, fontsize=14, loc='lower center', ncol=3,
                  bbox_to_anchor=(0.5, 1 - rows * 0.01),
                  bbox_transform=ax.transAxes)

        if i == len(titles) - 1:
            ax.set_xlabel("epochs", size=16)
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    fig.savefig(f"./images/{name}")


def create_new_test_set(image_directory):
    images = [(file, mpimg.imread(f"{image_directory}/{file}"))
              for file in os.listdir(image_directory)]

    x = []
    y = []

    for file, image in images:

        # crop
        height, width, _ = image.shape
        if width > height:
            left_crop = (width - height) // 2
            right_crop = width - left_crop
            img = image[:, left_crop:right_crop]
        elif height > width:
            top_crop = (height - width) // 2
            bot_crop = height - top_crop
            img = image[top_crop:bot_crop, :]
        else:
            img = image

        # resize
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

        x.append(img)
        y.append(int(f"{file.split('_')[0]}"))
        mpimg.imsave(f"additional_signs/{file.replace('jpg', 'png')}", img)

    # store
    data = dict(features=np.array(x, dtype=np.uint8),
                labels=np.array(y, dtype=np.uint8))
    with open('additional_signs/additional_signs.p', 'wb') as handle:
        pickle.dump(data, handle)


def plot_predictions(x, y, top_k_probs, top_k_preds, sign_names,
                     name="predictions"):
    cols = 2
    rows = np.ceil(x.shape[0] / cols).astype(int)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(16, 40), dpi=100)

    for i, (img, label, probs, preds) in enumerate(
            zip(x, y, top_k_probs, top_k_preds)):
        row = i // cols
        col = i % cols
        ax = axes[row, col * 2]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set_title(sign_names.loc[label].values[0], wrap=True)
        ax.imshow(img)

        width = 30
        true_label = sign_names.loc[label].values[0]
        if len(true_label) > width:
            true_label = true_label[0:width - 3] + "..."
        # red color if missclassified
        title_color = "r" if label != preds[0] else "k"
        for j, (prob, pred) in enumerate(zip(probs, preds.astype(int))):
            if label != preds[0]:
                pred_color = "r" if label != pred else "g"
            else:
                pred_color = 'k'
            sign_name = sign_names.loc[pred].values[0]
            if len(sign_name) > width:
                sign_name = sign_name[0:width - 3] + "..."
            ax.text(36, 5, f"{label:02}: {true_label}", size=20,
                    color=title_color)
            ax.text(36, 12 + 6 * j, f"{pred:02}: {sign_name:{width}.{width}}",
                    size=16, color=pred_color)
            ax.text(120, 12 + 6 * j, f"{prob * 100:.2f}%", ha="right", size=16,
                    color=pred_color)

        ax = axes[row, col * 2 + 1]
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')

    plt.subplots_adjust(left=-0.02,
                        bottom=0.02,
                        right=0.95,
                        top=0.98,
                        wspace=0.1,
                        hspace=0.1)

    fig.savefig(f"./images/{name}")


if __name__ == "__main__":
    create_new_test_set("additional_signs/preprocessed/")
