from MLP import MLP
from Data import genStressTensorData, fetchModelParametersFromFile, plotResults
import torch
import numpy as np




test_cauchy = True

inputDataFeatureCount = 9  # MUST ALWAYS BE 9
outputLayerNeuronCount = 1
seed = np.random.randint(100, size=(1,)).item()


if test_cauchy:

    datasetSize = 10_000

    (inputData, targetData) = genStressTensorData(n=inputDataFeatureCount, m=datasetSize, p=outputLayerNeuronCount, seed=seed)

    nn = MLP(pretrained=True, input_feature_count=inputDataFeatureCount, model_params=fetchModelParametersFromFile())
    # for l in nn.layers:
    #     print(l)
    prediction = nn.inference(inputData)


    if datasetSize == 1:
        print("input:")
        print(inputData)

        print(f"prediction: {prediction.item()}")

        print(f"truth: {targetData.item()}")

    else:
        num_correct = (torch.abs(targetData - prediction) < 0.5)
        percent_correct = (torch.sum(num_correct)/datasetSize)*100
        print(f"percent correct = {percent_correct.item()}%")


else:

    datasetSize = 500_000

    (inputData, targetData) = genStressTensorData(n=inputDataFeatureCount, m=datasetSize, p=outputLayerNeuronCount, seed=73)

    # Specify parameters for all layers except the input layer
    neuronCounts =        [256, 128, 64, 32, 16, 8, outputLayerNeuronCount]
    activationFunctions = ["leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "leakyReLU", "sigmoid"]

    modelConfig = {
        "neuron_counts": neuronCounts,
        "activation_functions": activationFunctions
    }

    nn = MLP(pretrained=False, input_feature_count=inputDataFeatureCount, model_config=modelConfig)
    # for l in nn.layers:
    #     print(l)

    (epochPlt, lossPlt) = nn.train(inputData, targetData, epochs=datasetSize)

    plotResults(epochPlt, lossPlt)
