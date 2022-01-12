import torchvision
import torch
import numpy as np

DATA_DIR = './data/'
from torchvision.transforms import ToPILImage


##########
# Binary MNIST
# From Arjovsky et al.: https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##########

def _shuffle(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x.numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(y.numpy())

    return x, y


def _train_val_MNIST():
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)

    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    mnist_train = _shuffle(*mnist_train)

    return mnist_train, mnist_val


def load_control_experimental_MNIST(spurious_value=0.1):
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)

    mnist_x, mnist_y = _shuffle(mnist.data, mnist.targets)

    # binarize
    mnist_y = (mnist_y < 5).float()

    mnist_min = (mnist_x[:30000], mnist_y[:30000])
    mnist_maj = (mnist_x[30000:], mnist_y[30000:])

    # make spurious
    mnist_min = append_spurious_channel(*mnist_min, sp_corr=0, sp_val=spurious_value)
    mnist_maj = append_spurious_channel(*mnist_maj, sp_corr=1, sp_val=spurious_value)

    control_data = {
        'images': torch.vstack([mnist_min[0], mnist_maj[0]]),
        'labels': torch.cat([mnist_min[1], mnist_maj[1]]).unsqueeze(1)
    }

    experimental_data = {
        'images': torch.vstack([mnist_min[0]] + 10 * [mnist_maj[0]]),
        'labels': torch.cat([mnist_min[1]] + 10 * [mnist_maj[1]]).unsqueeze(1)
    }

    return control_data, experimental_data


def load_test_MNIST(spurious_value=0.1):
    mnist_test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)
    mnist_test_x, mnist_test_y = mnist_test.data, mnist_test.targets
    mnist_test_y = (mnist_test_y < 5).float()

    mnist_test_x, mnist_test_y = append_spurious_channel(mnist_test_x, mnist_test_y, sp_corr=0.5, sp_val=spurious_value)

    test_data = {
        'images': mnist_test_x,
        'labels': mnist_test_y.unsqueeze(1)
    }

    return test_data


def append_spurious_channel(images, labels, sp_corr, sp_val):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(1 - sp_corr, len(labels)))
    images = torch.stack([images, images], dim=1)
    # display(ToPILImage()(images[0]).convert('RGB'))
    # set the second channel to colors
    images[torch.tensor(range(len(images))), 1, :, :] *= 0
    sp_val *= 255
    images[colors == 1.0, 1, :, :] += int(sp_val)
    # display(ToPILImage()(images[0]).convert('RGB'))
    return (images.float() / 255.), labels


def load_binary_MNIST(spurious_correlation=None):
    mnist_train, mnist_val = _train_val_MNIST()

    if spurious_correlation is not None:
        mnist_train = make_environment(mnist_train[0], mnist_train[1], 1 - spurious_correlation)
        mnist_val = make_environment(mnist_val[0], mnist_val[1], 0.5)
    else:
        def binarize(images, labels):
            # Assign a binary label based on the digit;
            labels = (labels < 5).float()
            return {
                'images': (images.float() / 255.),
                'labels': labels[:, None]
            }

        mnist_train = binarize(mnist_train[0], mnist_train[1])
        mnist_val = binarize(mnist_val[0], mnist_val[1])

    return mnist_train, mnist_val


def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # Assign a binary label based on the digit;
    labels = (labels < 5).float()
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    # You either zero out the first or the second of the 3 channels (r or g)
    # display(ToPILImage()(images[0]))
    images = torch.stack([images, images, images], dim=1)
    # display(ToPILImage()(images[0]).convert('RGB'))
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    # display(ToPILImage()(images[0]).convert('RGB'))
    return {
        'images': (images.float() / 255.),
        'labels': labels[:, None]
    }
