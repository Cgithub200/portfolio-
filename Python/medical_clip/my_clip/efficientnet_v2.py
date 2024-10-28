import csv
import math
from os import walk

import os
import torch.nn.functional as nnf
import torch
import torch.nn as nn
from math import ceil
from torch import optim, tensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
import torchvision.ops as ops


max_epoch = 300
num_classes = 30
batch_size = 70
version = "b0"



effversionv2 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")



to_go_to_excell = []

base_model_v2 = [
    # expanded_ratio, channels,layers,stride,kernelsize,fused
    [1, 16, 1, 1, 3,True],
    [4, 32, 2, 2, 3,True],#
    [4, 48, 2, 2, 3,True],#
    [4, 96, 3, 2, 3,False],#
    [6, 112, 5, 1, 3,False],
    [6, 192, 8, 2, 3,False],#
]

model_params = {
    "b0": (0, 0.3),  
    "b1": (0.5, 0.3),
    "b2": (1, 0.3),
    "b3": (2, 0.3),
    "b4": (3, 0.4),
    "b5": (4, 0.4),
    "b6": (5, 0.5),
    "b7": (6, 0.5),
}

Effsizes = {
    'b0': (224, 224), 'b1': (240, 240), 
    'b2': (260, 260), 'b3': (300, 300),
    'b4': (380, 380), 'b5': (456, 456),
    'b6': (528, 528), 'b7': (600, 600),
}

class cnn_block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(cnn_block, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.batchnorm(self.cnn(x)))


class EfficientChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Calculate channel-wise average pool
        x_avg = self.avg_pool(x)
        # Channel-wise convolutions
        x_out = self.conv1(x_avg)
        x_out = torch.relu(x_out)
        x_out = self.conv2(x_out)
        
        # Sigmoid activation
        x_out = self.sigmoid(x_out)
        
        # Element-wise multiplication
        return x * x_out


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expansion,
        reduction=4, 
        fused = False,
    ):
        super(MBConv, self).__init__()
        self.fused = fused
        self.residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expansion
        self.expand = in_channels != hidden_dim
        self.stride = stride
        self.survival = 0.7
        reduced_dim = int(in_channels / reduction)
        
        if self.expand:
            self.expand_conv = cnn_block(
                in_channels,
                hidden_dim,
                kernel_size=3 if self.fused else 1,
                stride=stride if self.fused else 1,
                padding=0,
            )

        if not self.fused:
            self.convfuse = cnn_block(
                hidden_dim,
                hidden_dim,
                kernel_size,
                stride,
                padding,
                groups=hidden_dim,
            ) 
        self.EffChanAtten = EfficientChannelAttention(hidden_dim, reduced_dim)
        self.b_conv = (nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 3 if self.fused and not self.expand else 1,stride if self.fused and not self.expand else 1,0, bias=False),
            nn.BatchNorm2d(out_channels),
        ))

    def padadd(self, x,front):   
        if self.fused and (self.expand if front == False else not self.expand):
            pad_stride = self.stride
            pad_kernel = 3
        else:
            pad_stride = 1
            pad_kernel = 1
        padding = ((pad_kernel-1)*pad_stride)//2 #int(ceil((outputsize - 1)*pad_stride-height+pad_kernel)/2)
        padding = (padding,padding,padding,padding)
        return nnf.pad(x, padding, "constant", 0)
    
    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival)
        return torch.div(x, self.survival) * binary
    
    def forward(self, inputs):
        if self.expand:
            data = self.padadd(inputs,True) if self.fused else inputs
            data = self.expand_conv(data)
        else:
            data = inputs 
        if not self.fused:
            data = self.convfuse(data)
        if self.residual:
            data = self.padadd(self.EffChanAtten(data),False) if self.fused and self.expand else self.EffChanAtten(data)
            #return ops.stochastic_depth(input = self.b_conv(data), p = self.survival , mode = "batch", training = self.training) + inputs
            return self.stochastic_depth(self.b_conv(data)) + inputs
        else:
            data = self.EffChanAtten(data)
            return self.b_conv(self.padadd(data,0) if self.fused and self.expand and effversionv2 else data)
        


class Eff_Net(nn.Module):
    def __init__(self, version, num_classes):
        super(Eff_Net, self).__init__()
        phi, dropout_rate = model_params[version]

        width_expansion = math.pow(1.1, phi)
        depth_expansion = math.pow(1.2, phi)

        last_channels = ceil(1280 * width_expansion)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.efficient_features(width_expansion, depth_expansion, last_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def efficient_features(self, width_expansion, depth_expansion, last_channels):
        channels = int(32 * width_expansion)
        in_channels = channels
        features = [cnn_block(3, channels, 3, stride=2, padding=1)]
        block = 0
        for expansion, channels, repeats, stride, kernel_size,fused in (base_model_v2):
            out_channels = 4 * ceil(int(width_expansion * channels) / 4)
            layer_repeat = ceil(depth_expansion * repeats)
            print(
            f"Conv block {block} out_channels {out_channels} layers_repeats {layer_repeat}"
            )
            block += 1
            for layer in range(layer_repeat):
                features.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        expansion=expansion,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  
                        fused=fused
                    )
                )
                print(
            f"MBConv in_channels {in_channels} out_channels {out_channels} expand_ratio {layer_repeat} stride {stride} kernel_size {kernel_size} padding {kernel_size // 2} fused {fused}"
            )
                in_channels = out_channels

        features.append(
            cnn_block(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
        )
        print(
            f"MBConv3 in_channels {in_channels} last_channels {last_channels} kernel_size {kernel_size}"
            )
        return nn.Sequential(*features)
    def forward(self, x,classify = False):
        x = self.pool(self.features(x))
        x = x.view(x.shape[0], -1)
        if classify:
            x = self.classifier(x)
        return x
    
def Eval_metrics(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    metric_dict = {}
    for myclass in range(num_classes):
        metric_dict[myclass] = {
            'TruePos': int(0),
            'FalsePos':int(0),
            'TrueNeg':int(0),
            'FalseNeg':int(0),
        }
    with torch.no_grad():
        tqdm_object = tqdm(loader, total=len(loader))
        for batch in tqdm_object:

            images = batch['image'].to(device)
            label_map = batch['label_map'].to(device)

     
            scores = model(images,classify = True)
            predictions = (scores > 0.5)
            #predictions = torch.tensor([[random.choice([True, False]) for _ in range(predictions.size(1))],[random.choice([True, False]) for _ in range(predictions.size(1))]]).to(device=device) 
            Correct = (predictions == label_map).all(dim=1)
            #print(f"scores {scores} targets {targets}")
            
            for myclass in range(num_classes):
                class_mask = (label_map[:, myclass] == 1).to(device=device) 
                metric_dict[myclass]['TruePos'] += ((predictions[:, myclass] == 1) & class_mask).sum().item()
                metric_dict[myclass]['FalsePos'] += ((predictions[:, myclass] == 1) & ~class_mask).sum().item()
                metric_dict[myclass]['TrueNeg'] += ((predictions[:, myclass] == 0) & ~class_mask).sum().item()
                metric_dict[myclass]['FalseNeg'] += ((predictions[:, myclass] == 0) & class_mask).sum().item()
                #print(f"targets {targets} predictions {predictions} myclass {myclass} class_mask {class_mask} TruePos {((predictions[:, myclass] == 1) & class_mask).sum().item()} FalsePos {((predictions[:, myclass] == 1) & ~class_mask).sum().item()} TrueNeg {((predictions[:, myclass] == 0) & ~class_mask).sum().item()} FalseNeg {((predictions[:, myclass] == 0) & class_mask).sum().item()}")
                
            num_correct += Correct.sum()
            num_samples += predictions.size(0)
    Precision = 0
    Recall = 0
    for myclass in range(num_classes):
        Precision += metric_dict[myclass]['TruePos']/(metric_dict[myclass]['TruePos'] + metric_dict[myclass]['FalsePos']) if (metric_dict[myclass]['TruePos'] + metric_dict[myclass]['FalsePos']) != 0 else 0
        Recall += metric_dict[myclass]['TruePos']/(metric_dict[myclass]['TruePos'] + metric_dict[myclass]['FalseNeg']) if (metric_dict[myclass]['TruePos'] + metric_dict[myclass]['FalseNeg']) != 0 else 0
    Precision /= num_classes
    Recall /= num_classes
    print(f"Precision: {float(Precision)*100:.2f} Recall: {float(Recall)*100:.2f} Accuracy: {float(num_correct)/float(num_samples)*100:.2f} score {num_correct} / {num_samples} metric_dict {metric_dict}")
    data = [f"{float(Precision)*100:.2f}",f"{float(Recall)*100:.2f}",f"{float(num_correct)/float(num_samples)*100:.2f}",f"{num_correct} / {num_samples}",f"{metric_dict}"]
    model.train()
    return data


from tqdm import tqdm
def run_eff_model(trainloader,evalloader,testloader,num_classes,class_weights): 
    class_weights = torch.tensor(class_weights).to(device=device)
    print ()
    print(f"class_weights: {class_weights}")
    PATH = "my_clip\eff_net\model.pth"
    #try:
       # model = torch.load(PATH)
        #print("model loaded")
    #except:
    model = Eff_Net(
            version = version,
            num_classes = num_classes,
        ).to(device)
    print("couldnt find model")
    model.train()
    learning_rate = 1e-4 if effversionv2 else 3e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    overall_loss = 0
    best_score = -1
    
    for i in range(max_epoch):
        tqdm_object = tqdm(trainloader, total=len(trainloader))
        for batch in tqdm_object:
            
            images = batch['image'].to(device)
            label_map = batch['label_map'].to(device)

            scores = model(images,classify = True)

            loss = criterion(scores, label_map)

            overall_loss += float(loss.item())  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(overall_loss)
        overall_loss = 0
        
        if (i % 10 == 9):
            score = [Eval_metrics(trainloader, model),Eval_metrics(evalloader, model),Eval_metrics(testloader, model)] 
            to_go_to_excell.append(score)

            if float(score[0][2]) == 100:# or float(score[2]) < best_score:
                break
            
            if float(score[1][2]) > best_score:
                torch.save(model, PATH)
                best_score = float(score[1][2])
            #elif float(score[1][2]) < best_score:
                #break

            with open('model_results_train.csv', 'w') as b:
                a = csv.writer(b,lineterminator='\n')
                a.writerows(to_go_to_excell)

    model.eval()
    return (model)


def get_trained_model():
    PATH = "my_clip\eff_net\model.pth"
    return torch.load(PATH).train()






import os

def shutdown_computer():
    if os.name == 'nt':  # for Windows
        os.system('shutdown /s /t 1')
    elif os.name == 'posix':  # for Linux/Unix
        os.system('sudo shutdown now')
    else:
        print("Unsupported OS")

from transformers import DistilBertTokenizer
if __name__ == "__main__":
    from datahandler import create_loaders,get_dfs,get_class_weights
     
    dir_path = os.path.join(os.getcwd(), "my_clip" , "static" , "roco-dataset", "data", "train", "radiology", "images")
    train_df, valid_df, test_df = get_dfs()

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    class_weights = get_class_weights(train_df)
    train_loader = create_loaders(train_df, tokenizer, batch_size, train=True) 
    valid_loader = create_loaders(dataframe = valid_df, train=False , batch_size = batch_size)
    test_loader = create_loaders(test_df, tokenizer, batch_size, train=False)

    model = run_eff_model(train_loader,valid_loader,test_loader,num_classes,class_weights)

    #Eval_metrics(trainloader, model)


    with open('model_results_train.csv', 'w') as b:
        a = csv.writer(b,lineterminator='\n')
        a.writerows(to_go_to_excell)

    


    
    #run_annoy(model)

