import torch
import numpy as np



class Layer:

    def __init__(self, pretrained, **kwargs):

        if not pretrained:

            input_count = kwargs.get("input_count")
            neuron_count = kwargs.get("neuron_count")

            ##### Initialize weights
            # self.weights = torch.rand(size=(neuron_count, input_count), dtype=torch.float64)  # Random Initialization

            # stddev = np.sqrt(2 / (input_count + neuron_count))
            # self.weights = torch.normal(0, stddev, size=(neuron_count, input_count), dtype=torch.float64)  # Xavier Initialization

            stddev = np.sqrt(2 / input_count)
            self.weights = torch.normal(0, stddev, size=(neuron_count, input_count), dtype=torch.float64)  # He Initialization

            self.biases = torch.zeros(size=(neuron_count, 1), dtype=torch.float64)

        else:
            self.weights = kwargs.get("pretrained_weights")
            self.biases = kwargs.get("pretrained_biases")


        self.weights.requires_grad_()
        self.biases.requires_grad_()

        # self.weights.retain_grad()
        # self.biases.retain_grad()



        self.activations = None
        self.nonlineartiy = kwargs.get("nonlinearity")
        self.index = kwargs.get("index")



    def __repr__(self):
        return (f"__________________________________________\n"
                f"Layer {self.index}\nWeights:\n{self.weights}\nBiases:\n{self.biases}\nActivation:\n{self.nonlineartiy}\n"
                f"__________________________________________")


    @staticmethod
    def reLU(k):
        return k * (k > 0)

    @staticmethod
    def leakyReLU(k):
        alpha = 0.01
        k[k < 0] *= alpha
        return k

    @staticmethod
    def sigmoid(k):
        return 1/(1 + torch.exp(k))

    @staticmethod
    def none(k):
        return k


    def activate(self, z):
        return getattr(self, self.nonlineartiy)(z)


    def feed(self, x):

        z = torch.matmul(self.weights, x) + self.biases
        self.activations = self.activate(z)

        """self.activations = torch.nn.functional.relu(z)"""



        # print("----------------------------")
        # print("weights:")
        # print(self.weights)
        # print(self.weights.size())
        # print("----------------------------")
        #
        #
        #
        # print("biases:")
        # print(self.biases)
        # print(self.biases.size())
        # print("----------------------------")
        #
        #
        # print("input:")
        # print(x)
        # print(x.size())
        # print("----------------------------")
        #
        # print("z:")
        # print(z)
        # print(z.size())
        # print("----------------------------")
        #
        # #
        # print("activations:")
        # print(self.activations)
        # print(self.activations.size())
        # print("----------------------------")


        return self.activations





