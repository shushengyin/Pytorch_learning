import  torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import math
#一、data load
#num_steps:序列的长度
batch_size, num_steps = 32, 35
#返回数据集的一个索引，以及一个字典（索引和char的映射）
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# 二、init params
def get_params(vocab_size, num_hiddens, device):
    """
    vocab_size:在语言模型中，输入和输出在训练语料的词表上
    num_hidden:hidden staste 的数量
    return the initial params of RNN model
    """
    num_inputs = num_outputs = vocab_size
    # 说明：在进行one_hot编码后，输入是一个vocab_size长度的向量；对于输出而言，就是一个分类问题
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
#三、Model
#1. init_run_state :renturn zero of the shape(n,h) and device
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )#元组
# 2. run：
def rnn(inputs, state, params):
    """
    inputs: X which shape is (时间步数量,批量大小,词表大小)
    state: Hidden staste
    params: wight of model
    return output which shape is (batch_sizeXtime_step,q) and new Hidden state is (n,h)
    """

    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状:(批量大小,词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) +
                       torch.mm(H, W_hh) +
                       b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)# output shape (batch_sizeXtime_step,q)
# 3.将上面的函数封装成类
class RNNModelScratch: #@save
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,  get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn
    def __call__(self, X, state):
        # input_X:shape=(batch_size,time steps)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        #output_X:shape=(time_steps,batch_size,28)--> f:  input of run
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
# 四、train
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)#首先将hidden state 进行初始化，清零
    outputs = [vocab[prefix[0]]]# 读取输入字符串中，第一个char对应的index
    # 函数：get_input: get pre char (the last char)  ，the first is vocab[prefix[0]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]], device=device), (1, 1))#shape：1x1
    for y in prefix[1:]:  # 预热期：just new the hidden state
        _, state = net(get_input(), state) # return output of run
        outputs.append(vocab[y]) # new outputs[-1]
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)# y is pred
        outputs.append(int(y.argmax(dim=1).reshape(1))) # new outputs[-1]
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(),
                      get_params,  init_rnn_state, rnn)
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())