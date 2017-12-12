from contextlib import contextmanager
from inspect import isfunction
import numpy as np
import pandas as pd
import tensorboardX as tb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from progressbar import ProgressBar, Bar, ETA, Percentage
from .core import USE_CUDA, Variable
from .modules import *


def get_parameter(module, path):
    path = path.split(".")
    for subpath in path:
        module = getattr(module, subpath)
    return module


class Trainer:
    def __init__(self, module: nn.Module, loss_fn, optimizer: optim.Optimizer, logdir: str=None):
        self.module = module
        if USE_CUDA:
            module.cuda()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.loss_parameters = []
        self.loss = 0
        self.module.register_forward_pre_hook(lambda m, n: self.clear_cache()) # Clear cache every time the module is called
        self.parameter_watcher = []
        self.output_watcher = []
        self.writer = tb.SummaryWriter(logdir)

    def clear_cache(self):
        self.loss = 0

    def train_once(self, dataloader):
        self.module.train()
        for x, y in dataloader:
            x, y = Variable(x), Variable(y)
            self.clear_cache()
            loss = self.get_loss(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, dataloader, num_epochs, validate_data=None):
        progressbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()])
        for epoch in progressbar(range(num_epochs)):
            self.train_once(dataloader)
            if validate_data is not None:
                self.show_metrics(epoch, validate_data)

    def show_metrics(self, epoch, dataset):
        self.module.eval()
        x, y = zip(*dataset)
        x, y = torch.stack(x), torch.stack(y)
        x, y = Variable(x, volatile=True), Variable(y, volatile=True)
        with self.watch(epoch):
            loss = self.get_loss(x, y)
            if isinstance(loss, autograd.Variable):
                loss = loss.data[0]
            self.writer.add_scalar("loss", loss, epoch)

    def save(self, filename):
        torch.save(self.module.state_dict(), filename)

    def add_parameter_loss(self, parameter: autograd.Variable, func):
        self.loss_parameters.append((parameter, func))

    def add_output_loss(self, module: nn.Module, func):
        if module not in self.module.modules():
            raise RuntimeError("The module of an output loss must be a children of the root module")

        def hook(module, input, output):
            self.loss += func(output)

        handler = module.register_forward_hook(hook)
        return handler

    def get_loss(self, x, y):
        """
        Total Loss = Parameter Loss + Output Loss + Loss_fn
        """
        loss = self.loss_fn(self.module(x), y)
        return self.loss + sum(func(parameter) for parameter, func in self.loss_parameters) + loss

    def watch_parameter(self, parameter, name: str, func='hist'):
        """
        Log the distribution of a specific parameter in the module

        Parameters
        ==========
        parameter: nn.Parameter
            The parameter to log
        name: str
            identifier of the watch
        func: str
            One of {'hist', 'mean', 'std'} or a function. If 'hist' and histogram of the parameter
            is logged, otherwise a scalar of the transformed value if logged.
        """
        self.parameter_watcher.append((func, name, parameter))

    def watch_output(self, module, name: str, func='hist'):
        """
        Log the distribution of the output in the module

        Parameters
        ==========
        module: nn.Module
            The output of which to be watched
        name: str
            identifier of the watch
        func: str
            One of {'hist', 'mean', 'std'} or a function. If 'hist' and histogram of the parameter
            is logged, otherwise a scalar of the transformed value if logged.
        """
        if module not in self.module.modules() and module is not self.module:
            raise RuntimeError("The module of an output watcher must be a children"
                               " of the root module or the root module itself")
        self.output_watcher.append((func, name, module))

    @contextmanager
    def watch(self, epoch: int):
        def hook_factory(name, func, epoch):
            def hook(module, input, output):
                if func != "hist" or isfunction(func):
                    if isfunction(func):
                        value = func(output.data)
                    else:
                        value = getattr(output.data, func)()
                    if isinstance(value, (np.ndarray, pd.Series)):
                        self.writer.add_scalars(name, pd.Series(value).to_dict(), epoch)
                    else:
                        self.writer.add_scalar(name, value, epoch)
                elif func == "hist":
                    self.writer.add_histogram(name, output.data.cpu().numpy(), epoch, bins='auto')
            return hook

        hooks = []
        for func, name, module in self.output_watcher:
            handler = module.register_forward_hook(hook_factory(name, func, epoch))
            hooks.append(handler)
        
        yield

        for func, name, parameter in self.parameter_watcher:
            if func != "hist" or isfunction(func):
                if isfunction(func):
                    value = func(parameter.data)
                else:
                    value = getattr(parameter.data, func)()
                if isinstance(value, (np.ndarray, pd.Series, list, tuple)):
                    self.writer.add_scalars(name, pd.Series(value).to_dict(), epoch)
                else:
                    self.writer.add_scalar(name, value, epoch)
            elif func == "hist":
                self.writer.add_histogram(name, parameter.data.cpu().numpy(), epoch, bins='auto')

        for hook in hooks:
            hook.remove()
