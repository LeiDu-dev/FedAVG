import torch

from torch import nn
from torch import optim
from torch.autograd import Variable

from model.lenet import lenet5


class client(object):
    def __init__(self, rank, data_loader):
        # seed
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # rank
        self.rank = rank

        # data loader
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

        # result
        self.accuracy = []

    @staticmethod
    def __load_global_model():
        global_model_state = torch.load('./cache/global_model_state.pkl')
        model = lenet5().cuda()
        model.load_state_dict(global_model_state)
        return model

    def __test(self, model):
        test_loss = 0
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()

                output = model(data)

                test_loss += nn.CrossEntropyLoss()(output, target).item()
                test_loss /= len(self.test_loader.dataset)

                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()

        return test_loss, test_correct / len(self.test_loader.dataset)

    def __train(self, model, optimizer):
        model.train()
        for data, target in self.train_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()

            optimizer.zero_grad()
            output = model(data)

            loss = nn.CrossEntropyLoss()(output, target)

            loss.backward()
            optimizer.step()
        return model

    def run(self, n_iter):
        model = self.__load_global_model()
        # optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
        optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.5)

        for i in range(n_iter):
            model = self.__train(model=model, optimizer=optimizer)
            test_loss, test_acc = self.__test(model)
            print('[Rank {:>2}  Iter {:>2}]  test_loss: {:.6f}, test_accuracy: {:.4f}'.format(
                self.rank,
                i,
                test_loss,
                test_acc
            ))
        print()
        torch.save(model.state_dict(), './cache/model_state_{}.pkl'.format(self.rank))
