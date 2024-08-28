import torch
import numpy as np
torch.set_printoptions(threshold=torch.inf)
import glob, os, re
import matplotlib.pyplot as plt, matplotlib.pylab as pylab



def plotResults(epochPlt, lossPlt):

    epochPlt = torch.tensor(epochPlt)
    lossPlt = torch.tensor(lossPlt)
    print(f"mean loss unfiltered = {lossPlt.mean()}")

    loss_filter = lossPlt > lossPlt.mean()
    mask = torch.ones(loss_filter.size(0), dtype=torch.bool)
    mask[loss_filter] = False
    # Apply the mask
    epochPlt = epochPlt[mask]
    lossPlt = lossPlt[mask]
    print(f"mean loss filtered = {lossPlt.mean()}")


    plt.figure(1)
    marker_size = 1
    f = plt.scatter(epochPlt[:], lossPlt[:], s=marker_size)
    plt.xlabel("epoch")
    plt.ylabel("loss")


    z = np.polyfit(epochPlt, lossPlt, 5)
    p = np.poly1d(z)
    pylab.plot(epochPlt, p(epochPlt), "r--")


    plt.show()



def fetchModelParametersFromFile():

    modelParams = {}

    # Define the directory and pattern
    directory = '' #os.getcwd()  # Replace with the directory path

    # Use glob to get all files matching the pattern
    weight_pattern = "layer_*_weights_*.csv"  # Pattern to match
    weight_files = glob.glob(os.path.join(directory, weight_pattern))
    weight_files.sort()

    bias_pattern = "layer_*_biases_*.csv"  # Pattern to match
    bias_files = glob.glob(os.path.join(directory, bias_pattern))
    bias_files.sort()


    for (w_file, b_file) in zip(weight_files, bias_files):

        weights = torch.tensor(np.genfromtxt(w_file, delimiter=','))
        biases = torch.tensor(np.genfromtxt(b_file,  delimiter=',')).reshape(-1, 1)

        regex_pattern = r"layer_(\d+)_weights_(.*?)\.csv"
        match = re.search(regex_pattern, w_file)

        index = match.group(1)
        activation = match.group(2)

        modelParams.update({f"Layer {index}": [weights, biases, activation, index] })

    return modelParams





def genData(n, m, p, seed):
    # Generate the 4x500 data tensor with random integers from 0 to 9

    # random.seed(seed)  # Set the random seed to ensure reproducibility
    # numbers = list(range(n*m))  # Create a list of numbers from 0 to n*m
    # random.shuffle(numbers)  # Shuffle the list
    # input_data = torch.tensor(numbers, dtype=torch.float64).reshape(n, m)/(n*m)

    input_data = torch.randn(size=(n, m), dtype=torch.float64)

    # Create the 2x100 target tensor
    target_data = torch.empty(size=(p, m), dtype=torch.float64)

    # Fill the target tensor based on the conditions
    target_data[0] = (input_data > 1.9)

    probability = torch.sum(target_data)/m
    print(probability.item())

    return input_data, target_data



def genStressTensorData(n, m, p, seed):

    torch.manual_seed(seed)


    input_data = torch.empty(size=(n, m), dtype=torch.float64)
    target_data = torch.empty(size=(p, m), dtype=torch.float64)
    max_principal_stresses = torch.empty(size=(p, m), dtype=torch.float64)


    for i in range(m):

        A = torch.randn(3, 3)
        sigma = (A + A.T) / 2  # convert A to a symmetric tensor

        eigenvalues, _ = torch.linalg.eig(sigma)

        real_float_eigenvalues = eigenvalues.real.to(dtype=torch.float64)

        max_principal_stresses[0, i] = torch.max(real_float_eigenvalues)

        input_data[:, i] = sigma.flatten()


    # yield_stress = 1.4668714587274254
    yield_stress = torch.mean(max_principal_stresses)
    target_data[0] = (max_principal_stresses > yield_stress)


    # print(f"yield stress = {yield_stress}")
    # print(f"max princpal stress = {max_principal_stresses}")

    # probability = torch.sum(target_data)/m
    # print(probability.item())


    return input_data, target_data







if __name__ == "__main__":

    seed = np.random.randint(100, size=(1,)).item()

    # inputData, targetData = genData(n=3, m=3, p=1, seed=seed)
    # print(inputData)
    # print(targetData)


    inputData, targetData = genStressTensorData(n=9, m=8, p=1, seed=73)
    # print("seed = " + str(seed))
    print(inputData)
    print(targetData)


    # for (i, k) in zip(inputData.T, targetData.T):
    #     print(i)
    #     print(k)
    #     print("--------------")
