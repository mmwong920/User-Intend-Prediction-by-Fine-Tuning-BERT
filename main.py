import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
# import model
from torch import nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer):
    lr_changes = []
    criterion = nn.CrossEntropyLoss().to(device)  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,eps=args.adam_epsilon)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = prepare_inputs(batch,model) # input, labels already shipped to cuda in this step

            # zero the parameter gradients
            optimizer.zero_grad()
            # model.optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule

            # forward + backward + optimize
            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses += loss.item()
        scheduler.step()
        run_eval(args,model,datasets,tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses/step+1)

def roberta_base_AdamW_LLRD(model,init_lr,eps):
    
    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 
        
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = init_lr
    w_decay = 0.4
    lr_decay = 0.95
    
    # === Pooler and classify ======================================================  
    
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "classify" in n)
                and any(nd in n for nd in no_decay)]
    
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "classify" in n)
                and not any(nd in n for nd in no_decay)]
    
    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}    
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": lr, "weight_decay": w_decay}    
    opt_parameters.append(head_params)

    lr *= lr_decay
                
    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": w_decay}
        opt_parameters.append(layer_params)       
        
        lr *= lr_decay
        
    # === Embeddings layer ==========================================================
    
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    opt_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": w_decay} 
    opt_parameters.append(embed_params)        

    return AdamW(opt_parameters, lr=init_lr,eps=eps)
  
def LLRD_train(args, model, datasets, tokenizer):
    lr_changes = []
    criterion = nn.CrossEntropyLoss().to(device)  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    optimizer = roberta_base_AdamW_LLRD(model,init_lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = prepare_inputs(batch,model) # input, labels already shipped to cuda in this step

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_changes.append([layer['lr'] for layer in optimizer.param_groups])
            losses += loss.item()
        scheduler.step()
        run_eval(args,model,datasets,tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses/step+1)
    plt.plot(lr_changes)
    plt.show()

def Warm_up_train(args, model, datasets, tokenizer):
    gamma = 0.8
    lr_changes = []
    criterion = nn.CrossEntropyLoss().to(device)  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        optimizer = AdamW(model.parameters(), lr=args.learning_rate*(gamma**epoch_count),eps=args.adam_epsilon)
        scheduler = transformers.get_scheduler(
        "linear",    
        optimizer = optimizer,
        num_warmup_steps = 50,
        num_training_steps = len(train_dataloader)
        )

        # optimizer = roberta_base_AdamW_LLRD(model,init_lr=args.learning_rate,eps=args.adam_epsilon)
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = prepare_inputs(batch,model) # input, labels already shipped to cuda in this step

            # zero the parameter gradients
            optimizer.zero_grad()
            # model.optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule

            # forward + backward + optimize
            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_changes.append([layer['lr'] for layer in optimizer.param_groups])
            scheduler.step()

            losses += loss.item()
        run_eval(args,model,datasets,tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses/step+1)
    plt.plot(lr_changes)
    plt.show()

def Advanced_train(args, model, datasets, tokenizer):
    gamma = 0.8
    lr_changes = []
    criterion = nn.CrossEntropyLoss().to(device)  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()
        optimizer = roberta_base_AdamW_LLRD(model,init_lr=args.learning_rate*(gamma**epoch_count),eps=args.adam_epsilon)
        scheduler = transformers.get_scheduler(
        "linear",    
        optimizer = optimizer,
        num_warmup_steps = 50,
        num_training_steps = len(train_dataloader)
        )
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = prepare_inputs(batch,model) # input, labels already shipped to cuda in this step

            # zero the parameter gradients
            optimizer.zero_grad()
            # model.optimizer.step()  # backprop to update the weights
            # model.scheduler.step()  # Update learning rate schedule

            # forward + backward + optimize
            logits = model(inputs,labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_changes.append([layer['lr'] for layer in optimizer.param_groups])
            scheduler.step()

            losses += loss.item()
        run_eval(args,model,datasets,tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses/step+1)
    plt.plot(lr_changes)
    plt.show()

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        # if step %100 == 0:
        #     print("pred:",logits.argmax(1),"actual:",labels)
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature).to(device)

    # task1: load training split of the dataset
    
    # task2: setup optimizer_scheduler in your model

    # task3: write a training loop for SupConLoss function 

    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,eps=args.adam_epsilon)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.9)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = prepare_inputs(batch,model) # input, labels already shipped to cuda in this step

            # zero the parameter gradients
            optimizer.zero_grad()
            f1 = model(inputs,labels)
            f2 = model(inputs,labels)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if args.task == 'supcon':
                loss = criterion(features,labels)
            elif args.task == 'simclr':
                loss = criterion(features,labels)
            loss.backward()
            optimizer.step()

            losses += loss.item()
        scheduler.step()
        print('epoch', epoch_count, '| losses:', losses/step+1)


if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    args = check_directories(args)
    set_seed(args)

    args.n_epochs = 10
    args.learning_rate = 5e-4

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))

    cache_results, already_exist = check_cache(args)
    tokenizer = load_tokenizer(args)

    if already_exist:
        features = cache_results
    else:
        data = load_data()
        features = prepare_features(args, data, tokenizer, cache_results)
    datasets = process_data(args, features, tokenizer)
    for k,v in datasets.items():
        print(k, len(v))

    #Training model without fine tuning (Transfer Learning)
    args.learning_rate = 5e-5
    model = IntentModel(args, tokenizer, target_size=60).to(device)

    for layer in model.encoder.parameters():
        layer.requires_grad = False

    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

    # Training baseline model....
    args.learning_rate = 5e-5
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

    #Training model with LLRD...
    args.learning_rate = 5e-5
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    LLRD_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
    
    #Training model with Warm Up learning rate...

    args.learning_rate = 1e-4
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    Warm_up_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

    #Training model with both LLRD and warmup...

    args.learning_rate = 1e-4
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    Warm_up_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')

    #Training model Sup C...

    args.task = 'supcon'
    args.learning_rate = 5e-5

    sup_con_model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, sup_con_model, datasets, tokenizer)

    import umap.umap_ as umap
    args.batch_size = 1000
    dataloader = get_dataloader(args, datasets['test'], 'test')
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, sup_con_model)
        embedding = sup_con_model.encoder(**inputs)
        break
    args.batch_size = 32
    indicies = [torch.cat([(labels.cpu() == i).nonzero() for i in range(0,60,6)]).view([-1]).detach().numpy().tolist()]
    reducer = umap.UMAP()
    new_embed = reducer.fit_transform(embedding[0][:,0,:][indicies].cpu().detach().numpy())
    plt.scatter(new_embed[:, 0], new_embed[:, 1], c=labels[indicies].cpu().detach().numpy(), cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24);

    #Training model with SimCLR Loss...
    args.task = 'simclr'
    args.learning_rate = 5e-5

    sim_clr_model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, sim_clr_model, datasets, tokenizer)
    args.batch_size = 1000
    dataloader = get_dataloader(args, datasets['test'], 'test')
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, sim_clr_model)
        embedding = sim_clr_model.encoder(**inputs)
        break
    args.batch_size = 32
    indicies = [torch.cat([(labels.cpu() == i).nonzero() for i in range(0,60,6)]).view([-1]).detach().numpy().tolist()]
    import umap.umap_ as umap
    reducer = umap.UMAP()
    new_embed = reducer.fit_transform(embedding[0][:,0,:][indicies].cpu().detach().numpy())
    plt.scatter(new_embed[:, 0], new_embed[:, 1], c=labels[indicies].cpu().detach().numpy(), cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24);
    args.batch_size = 32
