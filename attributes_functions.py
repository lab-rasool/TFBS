
import csv
import gzip
import logomaker
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift, Saliency
from captum.attr import visualization as viz
from torch.utils.data import DataLoader

from model import ConvNet
from data import chipseq_dataset

def seqtopad(sequence, motiflen, kind="DNA"):
    rows = len(sequence) + 2 * motiflen - 2
    S = np.zeros([rows, 4], dtype=np.float32)
    base = "ACGT" if kind == "DNA" else "ACGU"
    for i in range(rows):
        if i < motiflen - 1 or i >= len(sequence) + motiflen - 1:
            S[i] = 0.25
        elif sequence[i - motiflen + 1] in base:
            S[i, base.index(sequence[i - motiflen + 1])] = 1
    return S.T


class Chip_test:
    def __init__(self, filename, motiflen=24):
        self.file = filename
        self.motiflen = motiflen

    def openFile(self):
        test_dataset = []
        with gzip.open(self.file, "rt") as data:
            next(data)
            reader = csv.reader(data, delimiter="\t")
            for row in reader:
                test_dataset.append([seqtopad(row[2], self.motiflen), [int(row[3])]])
        return test_dataset

import torch.nn as nn
import torch.nn.functional as F

def shift_list(my_list, num):
    num = abs(num)
    my_list_ = my_list.copy()

    for i in range(num):
        if num > 0:
            my_list[:num] = my_list_[-num:]
            my_list[num:] = my_list_[:-num]
            # my_list.append(my_list.pop(0))
        elif num < 0:
            my_list[num:] = my_list_[:-num]
            my_list[:num] = my_list_[-num:]
            # my_list.insert(0, my_list.pop())
        else:
            my_list[:] = my_list_
    return my_list

def shift(arr, num):
    # result = np.empty_like(arr)
    result = torch.zeros_like(arr)
    if num > 0:
        result[:num] = arr[-num:]
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = arr[:-num]
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, embedding_size=32):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.embedding_size = embedding_size
        self.gate = nn.Linear(num_experts * embedding_size, num_experts)
        self.classifier = nn.Linear(embedding_size, 1)
        self.experts = nn.ModuleList(
            [nn.Linear(embedding_size, 32) for _ in range(num_experts)]
        )
    
    def forward(self, embeddings, return_gates=False):
        gating_weights = F.softmax(self.gate(embeddings), dim=1)
        embeddings = embeddings.view(-1, self.num_experts, self.embedding_size)
        gating_weights = gating_weights.unsqueeze(-1)
        combined_embedding = torch.mean(gating_weights * embeddings, dim=1)
        # print("gating_weights: ", gating_weights)
        return (self.classifier(combined_embedding), gating_weights) if return_gates else self.classifier(combined_embedding)

def run_moe(data, moe_model, nets, return_gates=False):
    total_preds, total_targets = [], []
    separate_data = True if type(data) == list else False
    if separate_data:
        embeddings = [model(data[i], return_embedding=True) for i, model in enumerate(nets)]
    else:
        embeddings = [model(data, return_embedding=True) for model in nets]
    concatenated = torch.cat(embeddings, dim=1)
    output = moe_model(concatenated, return_gates=return_gates)
    if return_gates:
        output, gating_weights = output
    pred_sig = torch.sigmoid(output)
    total_preds.extend(pred_sig.clone().detach().cpu().numpy())
    # total_targets.extend(target.clone().detach().cpu().numpy())

    return (pred_sig, gating_weights) if return_gates else pred_sig






## Generate Attribution Maps
def attribute_image_features(algorithm, input, **kwargs):
    tensor_attributions = algorithm.attribute(input, **kwargs)
    return tensor_attributions


## ShiftSmooth
def returnGradPred(img, net, magnitude=False, max_only=False, relu=False):
    
    img.requires_grad_(True)
    pred = net(img)
    # print("img.shape: ", img.shape, "pred.shape: ", pred.shape)

    Sc_dx = torch.autograd.grad(pred, img, 
                                create_graph=True, retain_graph=True,
                                )[0]

    grad = (torch.tensor(Sc_dx.clone().detach().cpu().numpy()))

    if max_only:
        max_grads = torch.max(abs(grad), dim=1)[0]
        mask = (abs(grad) == max_grads)
        Sc_dx[~mask] = 0

    if magnitude:
        Sc_dx = abs(Sc_dx)

    if relu:
        m = torch.nn.ReLU()
        Sc_dx = m(Sc_dx)
    
    return Sc_dx, pred

def returnGradPredMoE(img, moe_model, nets, magnitude=False, max_only=False, gate_scaling=False, relu=False):
    
    separate_data = True if type(img) == list else False

    if separate_data:
        for i in range(len(img)):
            img[i] = img[i].requires_grad_(True)
    else:
        img.requires_grad_(True)
    
    pred = run_moe(img, moe_model, nets, return_gates=gate_scaling)
    if gate_scaling:
        pred, gating_weights = pred
        # print("pred: ", pred, "gating_weights.shape: ", gating_weights.shape)
    n_outputs = len(img) if separate_data else 1
    Sc_dxs = []
    for i in range(n_outputs):
        im = img[i] if separate_data else img
        Sc_dx = torch.autograd.grad(pred, im, 
                                    create_graph=True, retain_graph=True,
                                    )[0]
        
        grad = (torch.tensor(Sc_dx.clone().detach().cpu().numpy()))

        if max_only:
            max_grads = torch.max(abs(grad), dim=1)[0]
            mask = (abs(grad) == max_grads)
            Sc_dx[~mask] = 0

        if magnitude:
            Sc_dx = abs(Sc_dx)

        if relu:
            m = torch.nn.ReLU()
            Sc_dx = m(Sc_dx)

        if separate_data:
            Sc_dxs.append(Sc_dx)
        else:
            Sc_dxs = Sc_dx
    if gate_scaling:
        # print("gating_weights.shape: ", gating_weights.shape, "torch.stack(Sc_dxs).shape:", torch.stack(Sc_dxs).shape)
        Sc_dxs = [gating_weights[0][i][0] * sc for i, sc in enumerate(Sc_dxs)]
    
    return (Sc_dxs, pred)

def GetAttShiftSmooth(
  x_value, net, nshiftlr=1,
  magnitude=False, max_only=False, moe_model=None, mask = None, og_img=None,
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
  exclude_nonmotif=None, 
#   exclude_nonmotif = ["GATAA", "TTATC"],
  visual_debug=False, relu=False):
    
    # device = net.device
    # device = x_value.device
    # softmax = torch.nn.Softmax(dim=1) if max_only else None
    x_np = torch.tensor(x_value)
    og_img = torch.tensor(og_img)
    
    k = 0
    total_gradients = torch.tensor(torch.zeros_like(og_img)).to(device)
    
    for i in range(nshiftlr*2 + 1):
        
        # print("x_np.shape: ", x_np.shape)
        x_shifted = torch.roll(torch.tensor(x_np.clone().detach().cpu().numpy()), i - nshiftlr, dims=2).to(device)
        if exclude_nonmotif is not None:
            seq = seq_to_string(x_shifted[0])
            contains_motif = False
            for motif in exclude_nonmotif:
                contains_motif = contains_motif or (motif in seq)
        else:
            contains_motif = True
        if contains_motif:        
            if mask is not None:
                x = torch.zeros_like(og_img).to(device)
                # print("x.shape: ", x.shape, "mask.shape: ", mask.shape, "x_shifted.shape: ", x_shifted.shape)
                # print("mask:", mask)
                x[:,:,~mask] = 0.25
                # x_shifted = x_shifted.reshape(x_shifted.shape[1], x_shifted.shape[2], x_shifted.shape[0])
                # x = x.reshape(x.shape[1], x.shape[2], x.shape[0])
                # print("x[:,mask].shape: ", x[:,:,mask].shape, "x_shifted.shape: ", x_shifted.shape)
                x[:,:,mask] = x_shifted#.reshape(x_shifted.shape[1], x_shifted.shape[2], x_shifted.shape[0])
                # print("x.shape: ", x.shape)
                x_shifted = x#.reshape(x.shape[2], x.shape[0], x.shape[1]).to(device)
                # print("x_shifted.shape: ", x_shifted.shape)
                view_x = x_shifted.reshape(x_shifted.shape[1], x_shifted.shape[2], x_shifted.shape[0]).clone().detach().cpu().numpy()
            
            if visual_debug:
                print("N = ", i - nshiftlr)
                crp_df_original_logo = create_logo(view_x, figsize=[50, 2.5])

            if moe_model is not None:
                gradient, pred = returnGradPredMoE(x_shifted.clone().detach().to(device), moe_model, nets=net, relu=relu) 
            else: 
                gradient, pred = returnGradPred(x_shifted.clone().detach().to(device), net=net, relu=relu)

            grad = torch.roll(torch.tensor(gradient.clone().detach().cpu().numpy()).to(device), (nshiftlr - i), dims=2)
            # print("grad.shape: ", grad.shape, "total_gradients.shape: ", total_gradients.shape)
            total_gradients += grad.clone().detach()
            k += 1
    
    grads = total_gradients.clone().detach() 
    if max_only:
        # grads = softmax(grads) * grads
        max_grads = torch.max(abs(grads), dim=1)[0]
        mask = (abs(grads) == max_grads)
        total_gradients[~mask] = 0
        

    if magnitude:
        total_gradients += abs(total_gradients)
    sq = 2 if magnitude else 1

    return ((total_gradients) / (k))#(nshiftlr*2 + 1))# ** sq


## Motifs
def create_logo(data, figsize, scale_data=False, visible_spines=None):
    if visible_spines is None:
        visible_spines = ["left", "bottom"]
    
    # Reshape and prepare the data
    output = {
        nucleotide: data[idx].reshape(data.shape[1]) for idx, nucleotide in enumerate("ACGT")
    }
    df = pd.DataFrame(output)

    # Optionally scale the data
    if scale_data:
        df /= df.max()

    # Create Logo object
    logo = logomaker.Logo(df, figsize=figsize)

    # Style the logo
    logo.style_spines(visible=False)
    logo.style_spines(spines=visible_spines, visible=True)
    logo.ax.set_ylabel("", labelpad=-1)
    logo.ax.xaxis.set_ticks_position("none")
    logo.ax.xaxis.set_tick_params(pad=-1)
    return logo

def create_motif(seq="00000", include_reverse=False):
    
    # Create a list of nucleotides
    nucleotides = ["A", "C", "G", "T"]

    motif_seq = torch.zeros((4, 5))
    
    for idx, nucleotide in enumerate(seq):
        motif_seq[nucleotides.index(nucleotide), idx] = 1
    
    if include_reverse:
        reverse_seq = torch.flip(motif_seq, [0, 1])
        return motif_seq, reverse_seq
    else:
        return motif_seq

def seq_to_string(seq):
    seq_str = ""
    for i in range(seq.shape[1]):
        if seq[0, i] == 1:
            seq_str += "A"
        elif seq[1, i] == 1:
            seq_str += "C"
        elif seq[2, i] == 1:
            seq_str += "G"
        elif seq[3, i] == 1:
            seq_str += "T"
    return seq_str