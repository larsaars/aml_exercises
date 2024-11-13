import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class Logger:
    """
    A class used to log and print metrics during training of a model.

    Attributes
    ----------
    n_metrics : int
        The number of metrics to be logged.
    format_str : str
        The string format to print the metrics during training.
    epoch_metrics : list
        The metrics for each epoch.
    avg_grads : list
        The average magnitude of gradients for a model.

    Methods
    -------
    log_epoch(size=1)
        Logs the metrics for the current epoch.
    log_grads(model)
        Logs the average magnitude of gradients of the specified model
    info()
        Prints the metrics for the last epoch.
    get_epoch_metrics(index)
        Returns the specified metric for all epochs.
    get_num_epochs()
        Returns the number of epochs.
    """

    def __init__(self, n_metrics, format_str):
        """
        Constructs all the necessary attributes for the Logger object.

        Parameters
        ----------
            n_metrics : int
                The number of metrics to be logged.
            format_str : str
                The string format to print the metrics.
        """
        self.format_str = format_str
        self.n_metrics = n_metrics
        self.epoch_metrics = []
        self.avg_grads = []

    def log_epoch(self, metrics):
        """
        Logs the metrics for the current epoch.

        Parameters
        ----------
            metrics : list or numpy.ndarray
                The metrics for the current batch.
        """
        self.epoch_metrics.append(metrics)


    def info(self):
        """
        Prints the metrics of the last epoch.
        """
        epoch = len(self.epoch_metrics)
        print(self.format_str.format(epoch, *self.epoch_metrics[-1]))


    def get_epoch_metrics(self, index):
        """
        Returns the specified (normalized) metric for all epochs.

        Parameters
        ----------
        index : int
            The index of the metric.
        normalized : bool, optional
            Flag indicating whether the metric is normalized (default is False).

        Returns
        -------
        numpy.ndarray
            The specified metric for all epochs, normalized if the 'normalized' flag is set to True.
        """
        assert 0 <= index < self.n_metrics, "index should be a valid index"
        return np.array([self.epoch_metrics[i][index] for i in range(len(self.epoch_metrics))])

    
    def get_num_epochs(self):
        """
        Returns the number of epochs.

        Returns
        -------
            int
                The number of epochs.
        """
        return len(self.epoch_metrics)


class Plotter:
    
    def __init__(self, logger, font_sale = 1.3):
        _config_plots(font_sale)
        self.logger = logger
        self.epochs = np.arange(1, logger.get_num_epochs() + 1)
        self.metrics = np.array(logger.epoch_metrics)

    def plot_results(self, idx_metrics, ylabel='', labels=None):
        idx_metrics = [idx_metrics] if isinstance(idx_metrics, int) else idx_metrics
        labels=['']*len(idx_metrics) if labels is None else labels
        assert len(labels) == len(idx_metrics), 'Labels and idx_metrics must have the same length!'
        plt.figure(figsize=(8,6))
        for idx, label in zip(idx_metrics, labels):
            plt.plot(self.epochs, self.metrics[:,idx], lw=2, label=label)
        plt.xlabel('epochs')
        plt.ylabel(ylabel)
        if any(labels): 
            plt.legend()
        plt.show()

    def plot_imgs(X, ncols=5, transform=None, figscale=1.5):
        n = len(X)
        nrows = math.ceil(n / ncols)
        figscale = figscale * max(nrows, ncols)
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(figscale, figscale), squeeze=False)
        axs = axs.flatten()
        for i in range(n):
            img = X[i]
            if len(img.shape) == 4: 
                img = img[0, :, :, :] 
            if transform:
                img = transform(img)
            axs[i].imshow(img, cmap='gray')
            axs[i].xaxis.set_visible(False)
            axs[i].yaxis.set_visible(False)
        for ax in axs[n:]:
            ax.set_visible(False)
        plt.show()


def _config_plots(fontscale=2, figsize=(9,6)):
    '''
    Configures matplotlib font sizes for plotting.
    Parameters:
        fontscale (int, optional): A multiplier for base font sizes. Defaults to 2.
        figsize (tuple, optional): The default figure size for plots. Defaults to (9, 6).
    '''
    plt.rcParams['figure.figsize'] = figsize
    
    SMALL_SIZE = fontscale*16
    MEDIUM_SIZE = fontscale*18
    BIGGER_SIZE = fontscale*22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
