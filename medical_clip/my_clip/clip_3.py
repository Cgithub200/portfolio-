
import csv
import itertools
import os
import heapq
from matplotlib import pyplot as plt
from torch import nn
import matplotlib.pyplot as plt
import joblib
import math
import torch.nn.functional as F
import torch
import torch.nn.functional as nnf
from tqdm import tqdm
from transformers import BertTokenizer,BertModel
import torch.optim as optim

to_go_to_excell = []

try:
    from .datahandler import get_dfs,create_loaders
except:
    from datahandler import get_dfs,create_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 15
num_workers = 0
lr = 1e-3
to_excel = []
dir_path = os.path.join(os.getcwd(), "my_clip" , 'static' , "roco-dataset", "data", "train", "radiology", "images")
epochs = 60
max_seq_length = 250


try:
    from my_clip.efficientnet_v2 import get_trained_model,Eff_Net,cnn_block,EfficientChannelAttention,MBConv
except:
    from efficientnet_v2 import get_trained_model,Eff_Net,cnn_block,EfficientChannelAttention,MBConv

class BertEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(dim=1)
        attention_mask = attention_mask.squeeze(dim=1)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = torch.mean(output.last_hidden_state, dim=1)
        return text_embed

class simple_projection_head(nn.Module):
    def __init__(self, input_shape, output_shape, dropout=0.3):
        super().__init__()
        self.projection = nn.Linear(input_shape, output_shape)
        self.hidden_layer = nn.Sequential(
            nn.GELU(),  
            nn.Linear(output_shape, output_shape),  
            nn.Dropout(dropout)  
        )
        self.layer_norm = nn.LayerNorm(output_shape)
    def forward(self, x):
        projected = self.projection(x)
        x = self.hidden_layer(projected)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIP(nn.Module):
    def __init__(self,temperature=0.7,):
        super().__init__()
        #self.image_encoder = get_trained_model()
        self.image_encoder = Eff_Net( version = "b0",num_classes = 0,)
        self.img_proj = simple_projection_head(input_shape=1280 , output_shape=556)
        self.text_encoder = BertEncoder()
        self.text_proj = simple_projection_head(input_shape=768 , output_shape=556)
        self.temperature = temperature

    def forward(self, image,input_ids,attention_mask):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        img_embeds = self.img_proj(image_features)
        text_embeds = self.text_proj(text_features)

        images_sim = torch.matmul(img_embeds , img_embeds.transpose(0, 1))
        texts_sim = torch.matmul(text_embeds , text_embeds.transpose(0, 1))
        targets = (images_sim + texts_sim) / 2
        targets = nnf.softmax(targets, dim=-1) 
        logits = torch.matmul(text_embeds , img_embeds.transpose(0, 1)) / self.temperature 
        text_loss = F.cross_entropy(logits, targets)
        image_loss = F.cross_entropy(logits.transpose(0, 1), targets.transpose(0, 1))
        loss =  (image_loss + text_loss) / 2.0 
        return loss.mean()
    
def train_model(model, train_loader, optimizer):
    sum_loss = 0.0
    total_count = 0
    my_tqdm = tqdm(train_loader, total=len(train_loader))
    for batch in my_tqdm:
        local_batch_size = batch["image"].size(0)
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        loss = model(images,input_ids,attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * local_batch_size
        total_count += local_batch_size
        avg_loss = sum_loss / total_count
    return avg_loss

def img_text_find(images, input_ids, attention_mask , my_model , skip = False ,max_res = None):
    my_model.eval()
    with torch.no_grad(): 
        my_batch_size = images.size(0) if not skip else max_res
        images_encoded = my_model.img_proj(my_model.image_encoder(images)) if not max_res else images
        query_encoded = my_model.text_proj(my_model.text_encoder(input_ids=input_ids, attention_mask=attention_mask)) 
        images_encoded = nnf.normalize(images_encoded,p=2,dim=-1)
        query_encoded = nnf.normalize(query_encoded,p=2,dim=-1)
        dot_similarity = torch.matmul(query_encoded, images_encoded.transpose(0, 1)) #Get dot product
        values, indices = torch.topk(dot_similarity.squeeze(0), my_batch_size) #Return index of highest dot product
    return values, indices

def encode_img(image,my_model = None):
    with torch.no_grad():
        image = image.to(device)
        if not my_model:
            my_model = get_trained_clip()
        my_model = my_model.to(device).eval()
        image_encoder = my_model.image_encoder
        img_proj = my_model.img_proj
        images_encoded = img_proj(image_encoder(image))#Encode image and get projection
        return images_encoded

def tokenize_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = tokenizer(
            text, padding=True, truncation=True, max_length=max_seq_length
        )
    input_ids = torch.tensor(text['input_ids']).to(device)
    attention_mask = torch.tensor(text['attention_mask']).to(device)
    return input_ids,attention_mask
  
def get_trained_clip():
    PATH = "my_clip\clip_model\model.pth"
    model = CLIP()
    model.load_state_dict(torch.load(PATH))
    model = model.to(device=device).eval()
    return model

def main(batch_size):
    train_df, valid_df, test_df = get_dfs()
    
    train_loader = create_loaders(dataframe = train_df, train=True , batch_size = batch_size)
    valid_loader = create_loaders(dataframe = valid_df, train=False , batch_size = batch_size)
    test_loader = create_loaders(dataframe = test_df, train=False , batch_size = batch_size)
    #model = get_trained_clip().train()
    model = CLIP(temperature=0.75).to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": 1e-4},
        {"params": model.text_encoder.parameters(), "lr": 1e-6},
        {"params": itertools.chain(
            model.img_proj.parameters(), model.text_proj.parameters()
        ), "lr": 1e-3, "weight_decay": 1.5e-3}
    ]
    
    optimizer = torch.optim.AdamW(
        params, weight_decay=0.
    )

    PATH = "my_clip\clip_model\model.pth"
    best_eval_ndcg = 0
    data = ['train_loss','score','ndcg','mean_reciprocal_rank',     'valid_loss','eval_score','test_ndcg','test_mean_reciprocal_rank'] 
    to_go_to_excell.append(data)
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_model(model, train_loader, optimizer)
        model.eval()
        with torch.no_grad():
            train_ndcg,train_mean_reciprocal_rank,train_score = run_evaluation(train_loader,model)
            eval_ndcg,eval_mean_reciprocal_rank,eval_score = run_evaluation(valid_loader,model)
            test_ndcg,test_mean_reciprocal_rank,test_score = run_evaluation(test_loader,model)
        if best_eval_ndcg < eval_ndcg:
            best_eval_ndcg = eval_ndcg
            torch.save(model.state_dict(), PATH)
            print("new best")

        print(f"eval_ndcg {eval_ndcg} eval_mean_reciprocal_rank {eval_mean_reciprocal_rank} old best {best_eval_ndcg}")

        data = [train_loss,train_score,train_ndcg,train_mean_reciprocal_rank,     eval_score,eval_ndcg,eval_mean_reciprocal_rank   , test_score,test_ndcg,test_mean_reciprocal_rank] 
        to_go_to_excell.append(data)
        with open(f'clip_results_train.csv', 'w') as b:
            a = csv.writer(b,lineterminator='\n')
            a.writerows(to_go_to_excell)
    return model


def run_evaluation(loader,model):
    with torch.no_grad():
        my_tqdm = tqdm(loader, total=len(loader))
        Correct = 0
        Total = 0
        ndcg = 0
        mean_reciprocal_rank = 0
        for batch in my_tqdm:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_map = batch['label_map'].to(device)

            batch_size = label_map.size(0)
            values, indices = img_text_find(images, input_ids, attention_mask , model)    
            ndcg_b,mean_reciprocal_rank_b,Correct_b,Total_b = results_from_indices(indices,label_map,batch_size)
            ndcg += ndcg_b 
            Correct += Correct_b
            Total += Total_b
            mean_reciprocal_rank += mean_reciprocal_rank_b
        ndcg /= Total
        mean_reciprocal_rank /= Total
        score = Correct/Total if Total != 0 else 0
    return ndcg,mean_reciprocal_rank,score

def results_from_indices(indices,label_map,batch_size):
    Total = 0
    Correct = 0
    mean_reciprocal_rank = 0
    ndcg = 0
    for pos,index in enumerate(indices):
        current_lm = label_map[pos]
        first_lm = index.item() if index.numel() == 1 else index[0].item()
        if torch.all(current_lm == label_map[first_lm]):
            Correct += 1
        Total += 1
        dc_gain = 0
        ideal_dc_gain = 0
        sorted_relevence = []
        local_reciprocal = batch_size
        for pos2, index2 in enumerate(index):
            compair = label_map[index2]
            gain , relevence = ndcg_calc(current_lm,pos2,compair)
            dc_gain += gain
            sorted_relevence += [relevence]
            local_reciprocal = mrr_calc(current_lm,pos2,local_reciprocal,compair)  
        mean_reciprocal_rank += local_reciprocal
        sorted_relevence = sorted(sorted_relevence, reverse=True)
        
        for pos2, relevence in enumerate(sorted_relevence):
            ideal_dc_gain += relevence / math.log2(pos2 + 2)
        #print(f"dc_gain {dc_gain} ideal_dc_gain {ideal_dc_gain} sorted_relevence {sorted_relevence}")
        ndcg += dc_gain / (ideal_dc_gain if ideal_dc_gain != 0 else 1)
    return ndcg,mean_reciprocal_rank,Correct,Total

def mrr_calc(current_lm,pos,local_reciprocal,compair):
    if pos < local_reciprocal:
        if torch.all(current_lm == compair):
            local_reciprocal = 1/(pos + 1)
    return local_reciprocal

def ndcg_calc(current_lm,pos,compair):
    positives = torch.sum(compair == 1).item()
    relevence = torch.sum((current_lm == compair) & (compair == 1)).item()/ (positives if positives != 0 else 1)
    dc_gain = relevence / math.log2(pos + 2)
    return dc_gain , relevence

import optuna
def objective(trial):
    joblib.dump(study, f"study.pkl")
    #proj_wd = trial.suggest_int('projection_Weight_Decay')
    proj_lr = trial.suggest_float('projection_lr',1e-4,1e-2)
    proj_wd = trial.suggest_float('projection_wd',1e-4,1e-2)
    image_encoder_lr = trial.suggest_float('image_encoder_lr',1e-5,1e-3)
    text_encoder_lr = trial.suggest_float('text_encoder_lr',1e-6,1e-4)

    scheduler_factor = trial.suggest_float('factor',0.4,0.6)
    patience = 3
    temperature = trial.suggest_float('temperature',0.5,1.5)

    train_df, valid_df, test_df = get_dfs()
    train_loader = create_loaders(dataframe = train_df, train=True , batch_size = batch_size)
    valid_loader = create_loaders(dataframe = valid_df, train=False , batch_size = batch_size)
    
    model = CLIP(temperature).to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": text_encoder_lr},
        {"params": itertools.chain(
            model.img_proj.parameters(), model.text_proj.parameters()
        ), "lr": proj_lr , "weight_decay": proj_wd}
    ]
    optimizer = torch.optim.AdamW(
        params, weight_decay=0.
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience, factor=scheduler_factor
    )
    step = "epoch"
    best_eval_score = 0
    best_ndcg = 0.0
    best_mean_reciprocal_rank = 0
    counter = 0
    my_patience = 7
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_model(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            eval_ndcg,eval_mean_reciprocal_rank,eval_score = run_evaluation(valid_loader,model)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                print(f"best_eval_score {best_eval_score}")

            if eval_ndcg > best_ndcg:
                best_ndcg = float(eval_ndcg)
                counter = 0
                print(f"best_ndcg {best_ndcg}")
            else:
                counter += 1
                if counter > my_patience:
                    print("prune")
                    break

            if eval_mean_reciprocal_rank > best_mean_reciprocal_rank:
                best_mean_reciprocal_rank = eval_mean_reciprocal_rank
                print(f"best_mean_reciprocal_rank {best_mean_reciprocal_rank}")

    return best_ndcg , best_eval_score , best_mean_reciprocal_rank


def get_img_embeds(valid_df,model):
    valid_loader = create_loaders( dataframe =  valid_df, train=False , batch_size = batch_size)
    valid_img_embeds = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(device))
            img_embeds = model.img_proj(image_features)
            valid_img_embeds.append(img_embeds)
    return model, torch.cat(valid_img_embeds)

class LimitedDict:
    def __init__(self, n):
        self.n = n
        self.heap = []

    def add_item(self, images, score, caption, label_map):
        if len(self.heap) < self.n:
            # Convert tensors and numpy arrays to regular Python types
            score = score.item()
            label_map = label_map.tolist() if isinstance(label_map, torch.Tensor) else label_map
            caption = caption.tolist() if isinstance(caption, torch.Tensor) else caption
            heapq.heappush(self.heap, (score, images, caption, label_map))
        else:
            min_score, min_name, min_caption, min_label_map = self.heap[0]
            if score > min_score:
                heapq.heappop(self.heap)
                # Convert tensors and numpy arrays to regular Python types
                label_map = label_map.tolist() if isinstance(label_map, torch.Tensor) else label_map
                caption = caption.tolist() if isinstance(caption, torch.Tensor) else caption
                heapq.heappush(self.heap, (score, images, caption, label_map))

    def get_items(self):
        return self.heap

def get_local_data(model):
    dataframe, _ , _ = get_dfs(split_ratio = 0)
    desired_columns = dataframe[['Image Index', 'captions']]
    return desired_columns

def get_images(model, query, max_res=1, img_embeds=None,image_filenames=None,image_captions=None,image_label_map=None):
    dataframe, _ , _ = get_dfs(split_ratio = 0)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    local = img_embeds is None
    limited_dict = LimitedDict(max_res)
    encoded_query = tokenizer([query])
    query_token = {
        key: torch.tensor(values).to(device) for key, values in encoded_query.items()
    }

    query_features = model.text_encoder(
        input_ids=query_token["input_ids"], attention_mask=query_token["attention_mask"]
    )
    query_embeddings = model.text_proj(query_features)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    if local:
        loader = create_loaders(dataframe = dataframe, train=False , batch_size = batch_size)
        tqdm_object = tqdm(loader, total=len(loader))
        with torch.no_grad():
            for batch in tqdm_object:
                images = batch['image'].to(device)
                caption_input_ids = batch['input_ids'].to(device)
                caption_attention_mask = batch['attention_mask'].to(device)
                label_map = batch['label_map'].to(device)
                image_filenames = batch['image_filenames']
                image_captions = batch["caption"]
                
                caption_embeddings = model.text_proj(model.text_encoder(input_ids=caption_input_ids, attention_mask=caption_attention_mask))
                img_embeds = model.img_proj(model.image_encoder(images))

                img_embeds = F.normalize(img_embeds, p=2, dim=-1)
                caption_embeddings = F.normalize(caption_embeddings, p=2, dim=-1)
    
                text_similarity = torch.matmul(query_embeddings , caption_embeddings.transpose(0, 1))
                image_similarity = torch.matmul(query_embeddings , img_embeds.transpose(0, 1))

                text_weight = 0.5
                dot_similarity = ((1 - text_weight) * image_similarity) + (text_weight * text_similarity)
                
                #dot_similarity = torch.matmul(query_embeddings  img_embeds.transpose(0, 1))

                for i in range(batch_size - 1):
                    limited_dict.add_item(image_filenames[i], dot_similarity[0][i],image_captions[i],label_map[i])
    else:
        img_embeds = F.normalize(img_embeds, p=2, dim=-1)
        dot_similarity = query_embeddings @ img_embeds.transpose(0, 1)
        for i in range(len(image_filenames) - 1):
            limited_dict.add_item(image_filenames[i], dot_similarity[0][i],image_captions[i],None)

    
    response = limited_dict.get_items()
    response = sorted(response, key=lambda x: x[0], reverse=True)
    response = [[item[1], item[2]] for item in response]



    histogram_data = []
    for i,value in enumerate(response):
        value = value[1]
        if query in value:
            histogram_data.append(i)
    
    

    plt.hist(histogram_data, bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.title(f'{query}')
    plt.show()

    return response

   



import joblib
import math
if __name__ == "__main__": 
    
    from datahandler import get_dfs,create_loaders

    
    main(batch_size)

   
    directions=['maximize', 'maximize', 'maximize']

    #study = optuna.create_study(directions=directions)

    study = joblib.load(f"study.pkl")
    #joblib.dump(study, f"study.pkl")
    #study.optimize(objective, n_trials=30)
   
    params = ['projection_lr','image_encoder_lr','text_encoder_lr', 'temperature','factor']

    fig = optuna.visualization.plot_parallel_coordinate(study,target=lambda t: t.values[0],target_name = 'best eval ndcg')
    fig.data[0].line.reversescale = not fig.data[0].line.reversescale
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study,target=lambda t: t.values[1],target_name = 'best eval score')
    fig.data[0].line.reversescale = not fig.data[0].line.reversescale
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study,target=lambda t: t.values[2],target_name = 'best eval mean reciprocal rank')
    fig.data[0].line.reversescale = not fig.data[0].line.reversescale
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study,target=lambda t: t.values[3],target_name = 'best eval loss')
    fig.data[0].line.reversescale = not fig.data[0].line.reversescale
    fig.show()

    #fig = optuna.visualization.plot_optimization_history(study,target=lambda t: t.values[0],target_name = 'best eval ndcg')
    #fig.show()

    fig = optuna.visualization.plot_slice(study,params=params,target=lambda t: t.values[0],target_name = 'best eval ndcg')
    fig.show()

    fig = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[0],target_name = 'best eval ndcg')
    fig.show()
    fig = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[1],target_name = 'best eval score')
    fig.show()
    fig = optuna.visualization.plot_param_importances(study,target=lambda t: t.values[2],target_name = 'best eval mean reciprocal rank')
    fig.show()

    #best_params = study.best_params
    #print(best_params)

    #model = get_trained_clip()
    #find_matches(model,'A patient with effusion')

    