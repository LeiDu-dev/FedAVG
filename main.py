import random

from model.data import loader
from model.server import server
from model.client import client
from model.plot import plot


def federated_learning():
    # hyper parameter
    n_client = 100
    n_sample = 10
    n_epoch = 100
    n_iter = 1
    batch_size = 128

    # dataset
    print('Initialize Dataset...')
    data_loader = loader('mnist', batch_size=batch_size)
    # data_loader = loader('cifar10', batch_size=batch_size)

    # initialize server
    print('Initialize Server...')
    s = server(size=n_client, data_loader=data_loader.get_loader([]))

    # initialize client
    print('Initialize Client...')
    clients = []
    for i in range(n_client):
        clients.append(client(rank=i, data_loader=data_loader.get_loader(
            random.sample(range(0, 10), 8)
        )))

    # federated learning
    for e in range(n_epoch):
        current_index = random.sample(range(0, 100), n_sample)
        print('\n========================== Epoch {:>2} =========================='.format(e + 1))
        for i in current_index:
            clients[i].run(n_iter=n_iter)
        s.aggregate(current_index)

    # plot
    plot()


if __name__ == '__main__':
    federated_learning()
