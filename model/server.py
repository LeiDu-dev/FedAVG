import copy

import torch

from torch.autograd import Variable

from model.lenet import lenet5


class server(object):
    def __init__(self, size, data_loader):
        self.size = size
        self.test_loader = data_loader[1]
        self.model = self.__init_server()
        self.accuracy = []

    @staticmethod
    def __init_server():
        model = lenet5().cuda()
        torch.save(model.state_dict(), './cache/global_model_state.pkl')
        return model

    def __test(self, model):
        test_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        return test_correct / len(self.test_loader.dataset)

    @staticmethod
    def __load_model_states(sample_index):
        model_states = []
        for i in sample_index:
            model_states.append(torch.load('./cache/model_state_{}.pkl'.format(i)))
        return model_states

    @staticmethod
    def __model_averaging(model_states):
        global_model_state = copy.deepcopy(model_states[0])
        for key in global_model_state.keys():
            for i in range(1, len(model_states)):
                global_model_state[key] += model_states[i][key]
            global_model_state[key] = torch.div(global_model_state[key], len(model_states))
        return global_model_state

    def aggregate(self, sample_index):
        model_states = self.__load_model_states(sample_index)
        global_model_state = self.__model_averaging(model_states)

        self.model.load_state_dict(global_model_state)
        test_acc = self.__test(self.model)
        print('\n[Global model]  test_accuracy: {:.2f}%\n'.format(test_acc * 100.))
        self.accuracy.append(test_acc)

        torch.save(self.accuracy, './cache/accuracy.pkl')
        torch.save(global_model_state, './cache/global_model_state.pkl')
