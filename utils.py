import os
import torch
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from IPython import display

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def gen_abspath(directory: str, rel_path: str) -> str:
    """
    Generate the absolute path by combining the given directory with a relative path.

    :param directory: The specified directory, which can be either an absolute or a relative path.
    :param rel_path: The relative path with respect to the 'dir'.
    :return: The resulting absolute path formed by concatenating the absolute directory
             and the relative path.
    """
    abs_dir = os.path.abspath(directory)
    return os.path.join(abs_dir, rel_path)


def read_csv(
    file_path: str,
    sep: str = ',',
    header: int = 0,
    on_bad_lines: str = 'warn',
    encoding: str = 'utf-8',
    dtype: dict = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read a CSV file from the specified path.
    """
    return pd.read_csv(file_path,
                       header=header,
                       sep=sep,
                       on_bad_lines=on_bad_lines,
                       encoding=encoding,
                       dtype=dtype,
                       **kwargs)


def eval_binary(
    y_true,
    y_label
):
    """
    Evaluate a binary classification task.
    """

    # Metrics that require the predicted labels (y_label)
    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_label)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_label)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_label)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    tn, fp, fn, tp = cm.ravel()

    print(f'accuracy: {acc:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall: {recall:.5f}')
    print(f'f1_score: {f1:.5f}')
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(f'confusion matrix:\n{cm}')


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    @staticmethod
    def use_svg_display():
        """Use the svg format to display a plot in Jupyter.

        Defined in :numref:`sec_calculus`"""
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
    
    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.

        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
        axes.set_xscale(xscale), axes.set_yscale(yscale)
        axes.set_xlim(xlim),     axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class MLP:

    def __init__(self,
                 input_channel,
                 output_channel,
                 hidden_num_1=128,
                 hidden_num_2=32,
                 batch_size=256,
                 num_epochs=20,
                 dropout_prob=0.7):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_num_1 = hidden_num_1
        self.hidden_num_2 = hidden_num_2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob

        self.embd_col_name = None
        self.label_col_name = None
        self.net = None

    @staticmethod
    def series2tensor(s: pd.Series, dtype):
        return torch.tensor(s.tolist(), dtype=dtype)

    def split_dateset(self, df, test_size=0.2, random_state=42):
        if self.embd_col_name is None or self.label_col_name is None:
            raise Exception('`embd_col_name` or `label_col_name` is None.')

        X, y = df[self.embd_col_name], df[self.label_col_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train = self.series2tensor(X_train, dtype=torch.float32)
        X_test = self.series2tensor(X_test, dtype=torch.float32)
        y_train = self.series2tensor(y_train, dtype=torch.long)
        y_test = self.series2tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_iter = DataLoader(train_dataset,
                                self.batch_size,
                                shuffle=True,
                                num_workers=4)
        test_iter = DataLoader(test_dataset,
                               self.batch_size,
                               shuffle=True,
                               num_workers=4)

        X_tensor = self.series2tensor(X, dtype=torch.float32)

        return X_tensor, y, train_iter, test_iter

    def init_model(self):
        net = nn.Sequential(nn.Linear(self.input_channel, self.hidden_num_1),
                            nn.ReLU(),
                            nn.Linear(self.hidden_num_1, self.hidden_num_2),
                            nn.ReLU(),
                            nn.Dropout(self.dropout_prob),
                            nn.Linear(self.hidden_num_2, self.output_channel),
                            nn.Softmax(dim=1))

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.01)
        net.apply(init_weights)

        loss = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(net.parameters())

        return net, loss, optimizer

    def train_epoch(self, train_iter, loss, updater):

        if isinstance(self.net, torch.nn.Module):
            self.net.train()

        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = self.net(X)
            l = loss(y_hat, y)

            updater.zero_grad()
            l.mean().backward()
            updater.step()

            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_iter, test_iter, loss, num_epochs, updater):
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, loss, updater)
            test_acc = evaluate_accuracy(self.net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))

        train_loss, train_acc = train_metrics

        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }

    def predict(self, X, net):
        pred_list = []

        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i+self.batch_size]
            with torch.no_grad():
                output = net(X_batch).argmax(axis=1)
            pred_list += output.tolist()

        return pred_list

    def main(self,
             df,
             embd_col_name,
             label_col_name):
        self.embd_col_name = embd_col_name
        self.label_col_name = label_col_name

        net, loss, optimizer = self.init_model()
        self.net = net

        X_tensor, _, train_iter, test_iter = self.split_dateset(df, test_size=0.2)
        metrics = self.train(train_iter=train_iter,
                             test_iter=test_iter,
                             loss=loss,
                             num_epochs=self.num_epochs,
                             updater=optimizer)

        return self.predict(X_tensor, self.net), metrics

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)
