# -*- coding: utf-8 -*-
"""
Author: Brandstetter, Schäfl
Date: 16-03-2020

This file is part of the "Hands on AI II" lecture material. The following copyright statement applies
to all code within this file.

Copyright statement: 
This material, no matter whether in printed or electronic form, may be used for personal and non-commercial
educational use only. Any reproduction of this manuscript, no matter whether as a whole or in parts, no matter whether
in printed or in electronic form, requires explicit prior acceptance of the authors.
"""
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
import pandas as pd
import PIL
import scipy
import seaborn as sns
import sklearn
import sys
import torch
import torch.nn as nn

from distutils.version import LooseVersion
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import art3d, proj3d
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional, Sequence, Tuple, Union


class Arrow3D(FancyArrowPatch):
    """
    Class provided by Vihang Patil [1].

    [1] https://www.jku.at/en/institute-for-machine-learning/about-us/team/vihang-patil/
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# noinspection PyUnresolvedReferences
def check_module_versions() -> None:
    """
    Check Python version as well as versions of recommended (partly required) modules.

    :return: None
    """
    python_check = '(\u2713)' if sys.version_info >= (3, 7) else '(\u2717)'
    numpy_check = '(\u2713)' if LooseVersion(np.__version__) >= LooseVersion(r'1.17') else '(\u2717)'
    pandas_check = '(\u2713)' if LooseVersion(pd.__version__) >= LooseVersion(r'1.0') else '(\u2717)'
    pytorch_check = '(\u2713)' if LooseVersion(torch.__version__) >= LooseVersion(r'1.3') else '(\u2717)'
    sklearn_check = '(\u2713)' if LooseVersion(sklearn.__version__) >= LooseVersion(r'0.22') else '(\u2717)'
    scipy_check = '(\u2713)' if LooseVersion(scipy.__version__) >= LooseVersion(r'1.4') else '(\u2717)'
    matplotlib_check = '(\u2713)' if LooseVersion(matplotlib.__version__) >= LooseVersion(r'3.0.0') else '(\u2717)'
    seaborn_check = '(\u2713)' if LooseVersion(sns.__version__) >= LooseVersion(r'0.10.0') else '(\u2717)'
    pil_check = '(\u2713)' if LooseVersion(PIL.__version__) >= LooseVersion(r'6.0.0') else '(\u2717)'
    print(f'Installed Python version: {sys.version_info.major}.{sys.version_info.minor} {python_check}')
    print(f'Installed numpy version: {np.__version__} {numpy_check}')
    print(f'Installed pandas version: {pd.__version__} {pandas_check}')
    print(f'Installed PyTorch version: {torch.__version__} {pytorch_check}')
    print(f'Installed scikit-learn version: {sklearn.__version__} {sklearn_check}')
    print(f'Installed scipy version: {scipy.__version__} {scipy_check}')
    print(f'Installed matplotlib version: {matplotlib.__version__} {matplotlib_check}')
    print(f'Installed seaborn version: {sns.__version__} {seaborn_check}')
    print(f'Installed PIL version: {PIL.__version__} {pil_check}')


def load_fashion_mnist() -> pd.DataFrame:
    """
    Load Fashion-MNIST data set [1].

    [1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.
        Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

    :return: Fashion-MNIST data set
    """
    fashion_mnist_data = datasets.fetch_openml(name=r'Fashion-MNIST')
    feature_names = [f'PX_{_}' for _ in range(len(fashion_mnist_data[r'feature_names']))]
    data = pd.DataFrame(fashion_mnist_data[r'data'], columns=feature_names).astype(dtype=np.float32)
    data[r'item_type'] = fashion_mnist_data[r'target']
    return data


def apply_pca(data: pd.DataFrame, n_components: int, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Apply principal component analysis (PCA) on specified data set and down-project project data accordingly.

    :param data: data set to down-project
    :param n_components: amount of (top) principal components involved in down-projection
    :param target_column: if specified, append target column to resulting, down-projected data set
    :return: down-projected data set
    """
    assert (type(n_components) == int) and (n_components >= 1)
    assert type(data) == pd.DataFrame
    assert ((type(target_column) == str) and (target_column in data)) or (target_column is None)
    if target_column is None:
        projected_data = PCA(n_components=n_components).fit_transform(data)
        projected_data = pd.DataFrame(np.ascontiguousarray(projected_data))
    else:
        projected_data = PCA(n_components=n_components).fit_transform(data.drop(columns=[target_column]))
        projected_data = pd.DataFrame(np.ascontiguousarray(projected_data))
        projected_data[target_column] = np.ascontiguousarray(data[target_column])
    return projected_data


def split_data(data: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data set into training and testing subsets.

    :param data: data set to split
    :param test_size: relative size of the test subset
    :return: training as well as testing subsets
    """
    assert (data is not None) and (type(data) == pd.DataFrame)
    assert (test_size is not None) and (type(test_size) == float) and (0 < test_size < 1)
    return train_test_split(data, test_size=test_size)


def plot_points_2d(data: pd.DataFrame, target_column: Optional[str] = None, legend: bool = True, **kwargs) -> None:
    """
    Visualize data points in a two-dimensional plot, optionally color-coding according to target column.

    :param data: data set to visualize
    :param target_column: optional target column to be used for color-coding
    :param legend: flag for displaying a legend
    :param kwargs: optional keyword arguments passed to matplotlib
    :return: None
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] in [2, 3])
    assert (target_column is None) or (
            (data.shape[1] == 3) and (type(target_column) == str) and (target_column in data))
    assert type(legend) == bool
    sns.set()
    plt.subplots(**kwargs)
    legend = r'full' if legend else False
    if target_column is None:
        sns.scatterplot(x=data[0], y=data[1], hue=None, legend=legend)
    else:
        data_stripped = data.drop(columns=[target_column])
        sns.scatterplot(x=data_stripped[0], y=data_stripped[1], hue=data[target_column], legend=legend)
    plt.show()
    sns.reset_orig()


# noinspection PyUnresolvedReferences
def plot_decision_boundaries(data: pd.DataFrame, classifier: ClassifierMixin, target_column: Optional[str] = None,
                             granularity: float = 10.0, legend: bool = True, **kwargs) -> None:
    """
    Visualize decision boundaries of specified classifier in a two-dimensional plot.

    :param data: data set for which to visualize decision boundaries
    :param classifier: classifier used to compute decision boundaries
    :param target_column: optional target column to be used for color-coding (defaults to last column)
    :param granularity: granularity of visualized color mesh
    :param legend: flag for displaying a legend
    :param kwargs: optional keyword arguments passed to matplotlib
    :return: None
    """
    assert (type(data) == pd.DataFrame) and (data.shape[1] == 3)
    assert (target_column is None) or ((type(target_column) == str) and (target_column in data))
    assert type(legend) == bool
    sns.set()

    # Prepare data and mesh grid for plotting.
    if target_column is None:
        data_stripped = data
        hue = data[data.columns[2]]
        cmap = ListedColormap(sns.cubehelix_palette(len(set(data[data.columns[2]]))).as_hex())
    else:
        data_stripped = data.drop(columns=target_column)
        hue = data[target_column]
        cmap = ListedColormap(sns.cubehelix_palette(len(set(target_column))).as_hex())
    xx, yy = np.meshgrid(np.arange(data_stripped[0].min() - 1, data_stripped[0].max() + 1, granularity),
                         np.arange(data_stripped[1].min() - 1, data_stripped[1].max() + 1, granularity))
    target = classifier.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])).astype(dtype=np.float32).reshape(xx.shape)

    # Plot color mesh of decision boundaries.
    fig, ax = plt.subplots(**kwargs)
    ax.pcolormesh(xx, yy, target, cmap=cmap)

    # Plot invisible auxiliary scatter plot in order to display a legend.
    if legend:
        sns.scatterplot(x=data_stripped[0][0], y=data_stripped[0][1], hue=hue, cmap=cmap, alpha=0.5, legend=r'full')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    sns.reset_orig()


def plot_function(x0: Sequence[Union[float, int]], data: pd.DataFrame, loss_function: Callable[[np.array], np.array],
                  loss_function_gradient: Callable[[np.array], np.array], **kwargs) -> None:
    """
    Plot specified function (mapping from R2 to R1) as well as specified gradient. Function based on implementation
    provided by Vihang Patil [1].

    [1] https://www.jku.at/en/institute-for-machine-learning/about-us/team/vihang-patil/

    :param x0: data point at which to start gradient descent
    :param data: data set used to evaluate loss function
    :param loss_function: function to evaluate on specified data
    :param loss_function_gradient: gradient of function to evaluate on specified data
    :param kwargs: optional keyword arguments passed to matplotlib
    :return: None
    """
    assert (data is not None) and (type(data) == pd.DataFrame)
    assert (r'x' in data) and (r'y' in data) and (len(data[r'x']) == len(data[r'y']))
    assert loss_function is not None
    assert loss_function_gradient is not None
    fig = plt.figure(**kwargs)
    ax = axes3d.Axes3D(fig)

    # Compute loss surface of specified loss function.
    data_x = data[r'x'].values.astype(dtype=np.float32)
    data_y = data[r'y'].values.astype(dtype=np.float32)
    data_loss = loss_function(np.meshgrid(data_x, data_y))

    _ = ax.plot_surface(data_x, data_y, data_loss, rstride=1, cstride=1, cmap=r'coolwarm', linewidth=0.0, alpha=0.9)

    # Evaluate loss function on specified data point.
    x0 = np.array(x0)
    y0 = loss_function(x0)
    ax.plot([x0[0]], [x0[1]], [y0], r'ro')

    # Compute and plot gradient information.
    dy0 = loss_function_gradient(x0)
    length = 0.3
    dx = dy0 * length / np.linalg.norm(dy0)
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle=r'-|>', color=r'g', shrinkA=0, shrinkB=0)
    arrow1 = Arrow3D([x0[0], x0[0] + dx[0]], [x0[1], x0[1] + dx[1]], [y0, y0 + np.dot(dy0, dx)], **arrow_prop_dict)
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle=r'-|>', color=r'r', shrinkA=0, shrinkB=0)
    arrow2 = Arrow3D([x0[0], x0[0] - dx[0]], [x0[1], x0[1] - dx[1]], [y0, y0 + np.dot(dy0, -dx)], **arrow_prop_dict)
    ax.add_artist(arrow1)
    ax.add_artist(arrow2)

    # Plot circle around gradient arrows.
    n = 100
    angle = np.linspace(0, 2 * np.pi, n)
    dx = np.concatenate((np.cos(angle).reshape(n, 1), np.sin(angle).reshape(n, 1)), axis=1) * length
    xs = x0 + dx
    y = np.array([y0 + np.dot(dy0, (x - x0)) for x in xs])
    plt.plot(xs[:, 0], xs[:, 1], y, color='k')
    plt.show()


def animate_gradient_descent(x0: Sequence[Union[float, int]], data: pd.DataFrame, n_updates: int,
                             learning_rate: float, momentum: float,
                             loss_function: Callable[[np.array], np.array],
                             loss_function_gradient: Callable[[np.array], np.array],
                             file_name: str = r'gradient_descent.gif', **kwargs) -> None:
    """
    Create animation of gradient descent.

    :param x0: data point at which to start gradient descent
    :param data: data set used to evaluate loss function
    :param n_updates: amount of update steps
    :param learning_rate: step size used by gradient descent
    :param momentum: momentum term of gradient descent
    :param loss_function: function to evaluate on specified data
    :param loss_function_gradient: gradient of function to evaluate on specified data
    :param file_name: name of animation file
    :param kwargs: optional keyword arguments passed to matplotlib
    :return: None
    """
    assert (x0 is not None) and (len(x0) == 2)
    assert (data is not None) and (type(data) == pd.DataFrame)
    assert (n_updates is not None) and (type(n_updates) in [int, float]) and (n_updates > 0)
    assert (learning_rate is not None) and (type(learning_rate) in [int, float])
    assert (momentum is not None) and (type(momentum) in [int, float])
    assert (r'x' in data) and (r'y' in data) and (len(data[r'x']) == len(data[r'y']))
    assert loss_function is not None
    assert loss_function_gradient is not None
    assert (file_name is not None) and (type(file_name) == str)

    def _gen_line() -> np.array:
        """
        Optimize function by gradient descent and save accompanying information. Function based on
        implementation provided by Vihang Patil [1].

        [1] https://www.jku.at/en/institute-for-machine-learning/about-us/team/vihang-patil/
        :return: gradient descent information
        """
        _x = np.array(x0)
        _data = np.empty((3, n_updates + 1))
        _data[:, 0] = np.concatenate((_x, [loss_function(_x)]))
        _v = np.zeros_like(_x)
        for _step in range(1, n_updates + 1):
            _grad = loss_function_gradient(_x)
            _v = momentum * _v - learning_rate * _grad
            _x += _v
            _data[:, _step] = np.concatenate((_x, [loss_function(_x)]))
        return _data

    def _update_line(_num: int, _data: np.array, _line: art3d.Line3D) -> Tuple[art3d.Line3D]:
        """
        Update line instance used for animation. Function based on implementation provided by Vihang Patil [1].

        [1] https://www.jku.at/en/institute-for-machine-learning/about-us/team/vihang-patil/

        :param _num: current time/update step
        :param _data: gradient descent data used for updating animation
        :param _line: line instance used in animation
        :return: updated line instance
        """
        _line.set_data(_data[:2, :_num])
        _line.set_3d_properties(_data[2, :_num])
        return _line,

    fig = plt.figure(**kwargs)
    ax = axes3d.Axes3D(fig)
    plt.close()

    # Compute loss surface of specified loss function.
    data_x = data[r'x'].values.astype(dtype=np.float32)
    data_y = data[r'y'].values.astype(dtype=np.float32)
    data_loss = loss_function(np.meshgrid(data_x, data_y))

    _ = ax.plot_surface(data_x, data_y, data_loss, rstride=1, cstride=1,
                        cmap=r'coolwarm', linewidth=1, antialiased=True, alpha=0.9)

    # Optimize specified function.
    data_optim = _gen_line()

    # Creating line objects.
    line = ax.plot([], [], [], r'-', linewidth=6, color=r'k')[0]

    # Setting the axes properties.
    # ax.view_init(30, -160)

    ax.set_xlim3d([data_x.min() - 1.0, data_x.max() + 1.0])
    ax.set_xlabel(r'X')

    ax.set_ylim3d([data_y.min() - 1.0, data_y.max() + 1.0])
    ax.set_ylabel(r'Y')

    ax.set_zlim3d([data_loss.min(), data_loss.max()])
    ax.set_zlabel(r'Z')

    # Creating and storing the gradient descent animation.
    gradient_descent_animation = animation.FuncAnimation(
        fig, _update_line, fargs=(data_optim, line), frames=n_updates + 1, interval=1000 / 30, blit=True)
    gradient_descent_animation.save(f'resources/{file_name}', writer=r'pillow', fps=30)


# noinspection PyUnresolvedReferences
def train_network(model: torch.nn.Module, data_loader: DataLoader,
                  optimizer: torch.optim.Optimizer, device: torch.device = r'cpu') -> None:
    """
    Train specified network for one epoch on specified data loader.

    :param model: network to train
    :param data_loader: data loader to be trained on
    :param optimizer: optimizer used to train network
    :param device: device on which to train network
    :return: None
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_index, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_network(model: torch.nn.Module, data_loader: DataLoader,
                 device: torch.device = r'cpu') -> Tuple[float, float]:
    """
    Test specified network on specified data loader.

    :param model: network to test on
    :param data_loader: data loader to be tested on
    :param device: device on which to test network
    :return: cross-entropy loss as well as accuracy
    """
    model.eval()
    loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += float(criterion(output, target).item())
            pred = output.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.view_as(pred)).sum().item())

    return loss / len(data_loader.dataset), correct / len(data_loader.dataset)
