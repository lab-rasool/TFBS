import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader

from data import ChipDataLoader, chipseq_dataset
from model import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ChipData = (
    "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_AC.seq.gz"
)
best_hyperparameters_path = (
    "/mnt/f/Projects/GenomicAttributions/models/deepBIND/best_hyperpamarameters.pth"
)
checkpoint_path = "/mnt/f/Projects/GenomicAttributions/models/deepBIND/MyModel_2.pth"
chipseq_test = (
    "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_B.seq.gz"
)


def logsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = 10 ** ((math.log10(b) - math.log10(a)) * x + math.log10(a))
    return y


def sqrtsampler(a, b):
    x = np.random.uniform(low=0, high=1)
    y = (b - a) * math.sqrt(x) + a
    return y


def train():
    chipseq = ChipDataLoader(ChipData)
    alldataset = chipseq.load_data()

    train_dataset_pad = alldataset
    size = int(len(train_dataset_pad) / 3)
    firstvalid = train_dataset_pad[:size]
    secondvalid = train_dataset_pad[size : size + size]
    thirdvalid = train_dataset_pad[size + size :]
    firsttrain = secondvalid + thirdvalid
    secondtrain = firstvalid + thirdvalid
    thirdtrain = firstvalid + secondvalid

    train1_dataset = chipseq_dataset(firsttrain)
    train2_dataset = chipseq_dataset(secondtrain)
    train3_dataset = chipseq_dataset(thirdtrain)
    valid1_dataset = chipseq_dataset(firstvalid)
    valid2_dataset = chipseq_dataset(secondvalid)
    valid3_dataset = chipseq_dataset(thirdvalid)

    batchSize = 64

    train_loader1 = DataLoader(
        dataset=train1_dataset, batch_size=batchSize, shuffle=True
    )
    train_loader2 = DataLoader(
        dataset=train2_dataset, batch_size=batchSize, shuffle=True
    )
    train_loader3 = DataLoader(
        dataset=train3_dataset, batch_size=batchSize, shuffle=True
    )
    valid1_loader = DataLoader(
        dataset=valid1_dataset, batch_size=batchSize, shuffle=False
    )
    valid2_loader = DataLoader(
        dataset=valid2_dataset, batch_size=batchSize, shuffle=False
    )
    valid3_loader = DataLoader(
        dataset=valid3_dataset, batch_size=batchSize, shuffle=False
    )

    train_dataloader = [train_loader1, train_loader2, train_loader3]
    valid_dataloader = [valid1_loader, valid2_loader, valid3_loader]

    best_AUC = 0
    learning_steps_list = [4000, 8000, 12000, 16000, 20000]
    for number in range(5):
        config = {
            "nummotif": 16,
            "motiflen": 24,
            "poolType": random.choice(["max", "maxavg"]),
            "neuType": random.choice(["hidden", "nohidden"]),
            "dropprob": random.choice([0.5, 0.75, 1.0]),
            "sigmaConv": logsampler(10**-7, 10**-3),
            "sigmaNeu": logsampler(10**-5, 10**-2),
            "learning_rate": logsampler(0.0005, 0.05),
            "momentum_rate": sqrtsampler(0.95, 0.99),
            "beta1": logsampler(10**-10, 10**-3),
            "beta2": logsampler(10**-10, 10**-3),
            "beta3": logsampler(10**-10, 10**-3),
        }
        random_neuType = config["neuType"]

        model_auc = [[], [], []]
        for kk in range(3):
            model = ConvNet(config).to(device)

            if random_neuType == "nohidden":
                optimizer = torch.optim.SGD(
                    [model.wConv, model.wRect, model.wNeu, model.wNeuBias],
                    lr=model.learning_rate,
                    momentum=model.momentum_rate,
                    nesterov=True,
                )

            else:
                optimizer = torch.optim.SGD(
                    [
                        model.wConv,
                        model.wRect,
                        model.wNeu,
                        model.wNeuBias,
                        model.wHidden,
                        model.wHiddenBias,
                    ],
                    lr=model.learning_rate,
                    momentum=model.momentum_rate,
                    nesterov=True,
                )

            train_loader = train_dataloader[kk]
            valid_loader = valid_dataloader[kk]
            learning_steps = 0
            while learning_steps <= 20000:
                model.mode = "training"
                auc = []
                for i, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)
                    # Forward pass
                    output = model(data)
                    if model.neuType == "nohidden":
                        loss = (
                            F.binary_cross_entropy(torch.sigmoid(output), target)
                            + model.beta1 * model.wConv.norm()
                            + model.beta3 * model.wNeu.norm()
                        )

                    else:
                        loss = (
                            F.binary_cross_entropy(torch.sigmoid(output), target)
                            + model.beta1 * model.wConv.norm()
                            + model.beta2 * model.wHidden.norm()
                            + model.beta3 * model.wNeu.norm()
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    learning_steps += 1

                    if learning_steps % 4000 == 0:
                        with torch.no_grad():
                            model.mode = "test"
                            auc = []
                            for i, (data, target) in enumerate(valid_loader):
                                data = data.to(device)
                                target = target.to(device)
                                # Forward pass
                                output = model(data)
                                pred_sig = torch.sigmoid(output)
                                pred = (
                                    pred_sig.cpu()
                                    .detach()
                                    .numpy()
                                    .reshape(output.shape[0])
                                )
                                labels = target.cpu().numpy().reshape(output.shape[0])

                                auc.append(metrics.roc_auc_score(labels, pred))
                            #                         print(np.mean(auc))
                            model_auc[kk].append(np.mean(auc))
                            print(
                                "AUC performance when training fold number ",
                                kk + 1,
                                "using ",
                                learning_steps_list[len(model_auc[kk]) - 1],
                                "learning steps = ",
                                np.mean(auc),
                            )
        print("-" * 50)

        for n in range(5):
            AUC = (model_auc[0][n] + model_auc[1][n] + model_auc[2][n]) / 3
            if AUC > best_AUC:
                best_AUC = AUC
                best_learning_steps = learning_steps_list[n]
                best_LearningRate = model.learning_rate
                best_LearningMomentum = model.momentum_rate
                best_neuType = model.neuType
                best_poolType = model.poolType
                best_sigmaConv = model.sigmaConv
                best_dropprob = model.dropprob
                best_sigmaNeu = model.sigmaNeu
                best_beta1 = model.beta1
                best_beta2 = model.beta2
                best_beta3 = model.beta3

    best_hyperparameters = {
        "best_poolType": best_poolType,
        "best_neuType": best_neuType,
        "best_learning_steps": best_learning_steps,
        "best_LearningRate": best_LearningRate,
        "best_LearningMomentum": best_LearningMomentum,
        "best_sigmaConv": best_sigmaConv,
        "best_dropprob": best_dropprob,
        "best_sigmaNeu": best_sigmaNeu,
        "best_beta1": best_beta1,
        "best_beta2": best_beta2,
        "best_beta3": best_beta3,
        "best_AUC": best_AUC,
    }
    print(best_hyperparameters)
    torch.save(
        best_hyperparameters,
        best_hyperparameters_path,
    )


def test():
    # ----------------- Testing the model -----------------
    best_hyperparameters = torch.load(best_hyperparameters_path)
    best_AUC = best_hyperparameters["best_AUC"]
    best_poolType = best_hyperparameters["best_poolType"]
    best_neuType = best_hyperparameters["best_neuType"]
    best_learning_steps = best_hyperparameters["best_learning_steps"]
    best_LearningRate = best_hyperparameters["best_LearningRate"]
    best_dropprob = best_hyperparameters["best_dropprob"]
    best_LearningMomentum = best_hyperparameters["best_LearningMomentum"]
    best_sigmaConv = best_hyperparameters["best_sigmaConv"]
    best_sigmaNeu = best_hyperparameters["best_sigmaNeu"]
    best_beta1 = best_hyperparameters["best_beta1"]
    best_beta2 = best_hyperparameters["best_beta2"]
    best_beta3 = best_hyperparameters["best_beta3"]

    config = {
        "nummotif": 16,
        "motiflen": 24,
        "poolType": best_poolType,
        "neuType": best_neuType,
        "dropprob": best_dropprob,
        "sigmaConv": best_sigmaConv,
        "sigmaNeu": best_sigmaNeu,
        "learning_rate": best_LearningRate,
        "momentum_rate": best_LearningMomentum,
        "beta1": best_beta1,
        "beta2": best_beta2,
        "beta3": best_beta3,
    }

    for number_models in range(6):
        model = ConvNet(config).to(device)

        if model.neuType == "nohidden":
            optimizer = torch.optim.SGD(
                [model.wConv, model.wRect, model.wNeu, model.wNeuBias],
                lr=model.learning_rate,
                momentum=model.momentum_rate,
                nesterov=True,
            )

        else:
            optimizer = torch.optim.SGD(
                [
                    model.wConv,
                    model.wRect,
                    model.wNeu,
                    model.wNeuBias,
                    model.wHidden,
                    model.wHiddenBias,
                ],
                lr=model.learning_rate,
                momentum=model.momentum_rate,
                nesterov=True,
            )

        # chipseq = ChipDataLoader(chipseq_test)
        chipseq = ChipDataLoader(ChipData)
        alldataset = chipseq.load_data()
        # batchSize = alldataset.__len__()
        batchSize = 5
        alldataset_dataset = chipseq_dataset(alldataset)
        alldataset_loader = DataLoader(
            dataset=alldataset_dataset, batch_size=batchSize, shuffle=False
        )

        train_loader = alldataset_loader
        valid_loader = alldataset_loader
        learning_steps = 0
        while learning_steps <= best_learning_steps:
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                # Forward pass
                output = model(data)

                if model.neuType == "nohidden":
                    loss = (
                        F.binary_cross_entropy(torch.sigmoid(output), target)
                        + model.beta1 * model.wConv.norm()
                        + model.beta3 * model.wNeu.norm()
                    )

                else:
                    loss = (
                        F.binary_cross_entropy(torch.sigmoid(output), target)
                        + model.beta1 * model.wConv.norm()
                        + model.beta2 * model.wHidden.norm()
                        + model.beta3 * model.wNeu.norm()
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_steps += 1

        with torch.no_grad():
            model.mode = "test"
            auc = []
            for i, (data, target) in enumerate(valid_loader):
                data = data.to(device)
                target = target.to(device)
                # Forward pass
                output = model(data)
                pred_sig = torch.sigmoid(output)
                pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                labels = target.cpu().numpy().reshape(output.shape[0])

                auc.append(metrics.roc_auc_score(labels, pred))
            #
            AUC_training = np.mean(auc)
            print("AUC for model ", number_models, " = ", AUC_training)
            if AUC_training > best_AUC:
                state = {
                    "conv": model.wConv,
                    "rect": model.wRect,
                    "wHidden": model.wHidden,
                    "wHiddenBias": model.wHiddenBias,
                    "wNeu": model.wNeu,
                    "wNeuBias": model.wNeuBias,
                }
                torch.save(state, checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model = ConvNet(config).to(device)
    model.wConv = checkpoint["conv"]
    model.wRect = checkpoint["rect"]
    model.wHidden = checkpoint["wHidden"]
    model.wHiddenBias = checkpoint["wHiddenBias"]
    model.wNeu = checkpoint["wNeu"]
    model.wNeuBias = checkpoint["wNeuBias"]

    with torch.no_grad():
        model.mode = "test"
        auc = []

        for i, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            # Forward pass
            output = model(data)
            pred_sig = torch.sigmoid(output)
            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels = target.cpu().numpy().reshape(output.shape[0])

            auc.append(metrics.roc_auc_score(labels, pred))

        AUC_training = np.mean(auc)

    test_data = alldataset
    test_dataset = chipseq_dataset(test_data)
    batchSize = test_dataset.__len__()
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)
    with torch.no_grad():
        model.mode = "test"
        auc = []

        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            # Forward pass
            output = model(data)
            pred_sig = torch.sigmoid(output)
            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels = target.cpu().numpy().reshape(output.shape[0])

            auc.append(metrics.roc_auc_score(labels, pred))

            # plot the ROC curve
            fpr, tpr, thresholds = metrics.roc_curve(labels, pred)
            plt.plot(fpr, tpr)

        AUC_training = np.mean(auc)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curve | AUC on test data {AUC_training}")
        plt.savefig("DeepBIND_ROC_curve.png")
        print("AUC on test data = ", AUC_training)


def main():
    # train()
    test()


if __name__ == "__main__":
    main()
