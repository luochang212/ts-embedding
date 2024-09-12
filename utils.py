import os
import math
import collections
import torch
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
from IPython import display

from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
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
            test_acc = self.evaluate_accuracy(test_iter)
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


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "Sequence length exceeds max_len"

        return x + self.pe[:, :seq_len].to(x.device)


class TransformerClassifier(nn.Module):
    """Transformer + MLP 分类器"""

    def __init__(self, input_dim, model_dim, nhead, num_layers, hidden_dim, num_classes):
        super().__init__()

        # 输入嵌入层和位置编码
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, src, src_key_padding_mask=None, need_mlp=True):
        # 输入嵌入和位置编码
        src = self.embedding(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)

        # Transformer Encoder (使用 padding mask 来忽略填充的部分)
        transformer_out = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Average pooling over the sequence dimension, ignoring padding
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (batch_size, sequence_length, 1)
            transformer_out = transformer_out * mask  # Mask the padded positions
            sum_out = transformer_out.sum(dim=1)  # Sum over the sequence length
            avg_out = sum_out / mask.sum(dim=1).clamp(min=1)  # Divide by the actual sequence lengths
        else:
            avg_out = transformer_out.mean(dim=1)  # Average pooling over sequence length

        if need_mlp:
            return self.mlp(avg_out)

        return avg_out


class TSTransformer:
    """Time Series Transformer Model"""

    def __init__(self,
                 input_dim,
                 model_dim,
                 nhead,
                 num_layers,
                 hidden_dim,
                 num_classes,
                 num_epochs,
                 batch_size):
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
       
        self.loss = nn.CrossEntropyLoss(reduction='none') # 损失函数
        self.optimizer = None  # 优化器
        self.net = None  # 神经网络

    @staticmethod
    def collate_fn(batch):
        """对变长序列进行 padding"""
        X_batch, y_batch = zip(*batch)

        X_batch_padded = pad_sequence(X_batch, batch_first=True, padding_value=0.0)
        y_batch_tensor = torch.stack(y_batch)

        lengths = torch.tensor([len(x) for x in X_batch])
        src_key_padding_mask = torch.arange(X_batch_padded.size(1)) \
            .expand(len(X_batch), X_batch_padded.size(1)) >= lengths.unsqueeze(1)

        return X_batch_padded, y_batch_tensor, src_key_padding_mask

    @staticmethod
    def create_dataset(X_train, y_train, X_test, y_test, batch_size):
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)

        # 判断时间序列是不是定长序列，如果不是定长序列，添加 padding
        is_fixed = True
        first_len = X_train.shape[1]
        for X in X_train[1:]:
            if(X.shape[0] != first_len):
                is_fixed = False
                break

        print(f'Fixed length: {is_fixed}')
        collate_fn = None if is_fixed else collate_fn

        # 用 DataLoader 加载数据
        train_iter = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        test_iter = DataLoader(test_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               collate_fn=collate_fn)

        return train_iter, test_iter, is_fixed

    @staticmethod
    def create_model(input_dim, model_dim, nhead, num_layers, hidden_dim, num_classes, device):
        # 定义模型
        net = TransformerClassifier(input_dim=input_dim,
                                    model_dim=model_dim,
                                    nhead=nhead,
                                    num_layers=num_layers,
                                    hidden_dim=hidden_dim,
                                    num_classes=num_classes)

        # 权重初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)
        net.apply(init_weights)

        # 如果有 GPU，把模型复制到 GPU
        return net.to(device), torch.optim.Adam(net.parameters(), lr=0.01)

    def evaluate_accuracy(self, data_iter, device):
        """计算在指定数据集上模型的精度"""
        if isinstance(self.net, torch.nn.Module):
            self.net.eval()  # 将模型设置为评估模式
        metric = Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for data in data_iter:
                if self.is_fixed:
                    X_batch, y_batch = data
                    X_batch = X_batch.to(device)
                    y = y_batch.to(device)
                    src_key_padding_mask = None
                else:
                    X_batch, y_batch, src_key_padding_mask = data
                    X_batch = X_batch.to(device)
                    y = y_batch.to(device)
                    src_key_padding_mask = src_key_padding_mask.to(device)

                # 计算梯度并更新参数
                y_hat = self.net(X_batch, src_key_padding_mask=src_key_padding_mask)
                metric.add(accuracy(y_hat, y), y.numel())

        return metric[0] / metric[1]

    def train_epoch(self, train_iter, device):
        """训练模型一个迭代周期"""

        # 将模型设置为训练模式
        if isinstance(self.net, torch.nn.Module):
            self.net.train()
        self.net.to(device)
    
        # 训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for data in train_iter:

            if self.is_fixed:
                X_batch, y_batch = data
                X_batch = X_batch.to(device)
                y = y_batch.to(device)
                src_key_padding_mask = None
            else:
                X_batch, y_batch, src_key_padding_mask = data
                X_batch = X_batch.to(device)
                y = y_batch.to(device)
                src_key_padding_mask = src_key_padding_mask.to(device)

            # 计算梯度并更新参数
            y_hat = self.net(X_batch, src_key_padding_mask=src_key_padding_mask)
            l = self.loss(y_hat, y)

            # 使用 PyTorch 内置的优化器和损失函数
            self.optimizer.zero_grad()
            l.mean().backward()
            self.optimizer.step()

            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_iter, test_iter, num_epochs, device):
        """训练模型"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, device)
            test_acc = self.evaluate_accuracy(test_iter, device)
            animator.add(epoch + 1, train_metrics + (test_acc,))

        train_loss, train_acc = train_metrics

        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }

    def predict(self, X):
        pred_list = []

        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i+self.batch_size]
            with torch.no_grad():
                output = self.net(X_batch).argmax(axis=1)
            pred_list += output.tolist()

        return pred_list

    def save_model(self, model_path):
        """保存模型权重"""
        torch.save(self.net.state_dict(), model_path)

    def load_model(self, model_path):
        net = TransformerClassifier(input_dim=self.input_dim,
                                    model_dim=self.model_dim,
                                    nhead=self.nhead,
                                    num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim,
                                    num_classes=self.num_classes,
                                    num_epochs=self.num_epochs,
                                    batch_size=self.batch_size)

        net.load_state_dict(torch.load(model_path, weights_only=True))
        return net

    def main(self, X_train, y_train, X_test, y_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = X_train.shape[2] if self.input_dim is None else None

        train_iter, test_iter, self.is_fixed = self.create_dataset(X_train,
                                                    y_train,
                                                    X_test,
                                                    y_test,
                                                    self.batch_size)
        
        self.net, self.optimizer = self.create_model(input_dim=self.input_dim,
                                     model_dim=self.model_dim,
                                     nhead=self.nhead,
                                     num_layers=self.num_layers,
                                     hidden_dim=self.hidden_dim,
                                     num_classes=self.num_classes,
                                     device=device)

        # 训练
        metrics = self.train(train_iter=train_iter,
                             test_iter=test_iter,
                             num_epochs=self.num_epochs,
                             device=device)

        return self.predict(torch.tensor(X_test, dtype=torch.float32)), metrics

    def __call__(self, *args, **kwargs):
        return self.main(*args, **kwargs)


class Convert:

    def __init__(self, lst: list, file_path='./item_list.pkl'):
        self.lst = lst
        self.file_path = file_path

        # 对 lst 进行编码
        self.item_list = None
        self.item_dict = None
        self.parse()

    @staticmethod
    def to_dict(item_list):
        item_dict = dict()
        for i, item in enumerate(item_list):
            item_dict[item] = i
        return item_dict

    def parse(self):
        """对列表元素编码"""

        # 统计字符串频率
        frequency = collections.Counter(self.lst)

        # 按字符串频率，倒序排列
        sorted_items = sorted(frequency.items(),
                              key=lambda e: e[1],
                              reverse=True)

        self.item_list = [e[0] for e in sorted_items]
        self.item_dict = self.to_dict(self.item_list)

    def reset(self):
        self.item_list = self.read()
        self.item_dict = self.to_dict(self.item_list)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        return self.item_list[index]

    def encoder(self, item):
        return self.item_dict.get(item)

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.item_list, f)

    def read(self):
        with open(self.file_path, 'rb') as f:
            item_list = pickle.load(f)
        return item_list
