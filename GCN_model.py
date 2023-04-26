# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from reference_code import *
from transformer import TransformerEncoder
from functions import r2_rmse_mae, device, random_seed, tensor_to_numpy, make_batch
    
class SimpleDNN(nn.Module):

    def __init__(self, 
                args:dict, 
                layers:list):

        super().__init__()
        modules  = [nn.Linear(layers[0], layers[1], bias=args["bias"])]

        for l_num in range(len(layers)-2):

            modules.extend([nn.GELU(), nn.Dropout(p=args["dropout_ratio"]), nn.Linear(layers[l_num+1], layers[l_num+2], bias=False)])
            
        self.weight  =  nn.Sequential(*modules)

    def forward(self, 
                input:torch.Tensor):

        return self.weight(input)
    
    
class BuchwaldHartwigModel(nn.Module):
    
    def __init__(self, 
                mpnn_args:dict, 
                tr_args:dict, 
                dnn_args:dict, 
                args:dict, 
                layer:dict, 
                sample_num:int):
        
        super(BuchwaldHartwigModel, self).__init__()
        random_seed(args["seed"])
        self.args = args
        self.tr_args = tr_args
        self.graphs  = dict()
        self.device = device()
        self.mpnn = MPN(mpnn_args).to(self.device)
        self.transformer = TransformerEncoder(tr_args).to(self.device)
        self.pre_yield = SimpleDNN(dnn_args, layer).to(self.device)
        self.embedding = nn.Embedding(8, args["hidden_size"]).to(self.device)
        
        if sample_num:
            path = f"DataFrame/pretrain/MolCLR/sample_num{sample_num}/mpnn"
            self.mpnn.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
            
    def _set_train_eval(self, 
                        sets:str):
        
        if sets == "train":
            self.mpnn.train()
            self.transformer.train()
            self.pre_yield.train()
            self.embedding.train()
            
        if sets == "eval":
            self.mpnn.eval()
            self.transformer.eval()
            self.pre_yield.eval()
            self.embedding.eval()
            
    def _save_load(self, 
                    path:str, 
                    mode:str=None, 
                    device:str=None, 
                    strict:bool=True):
        
        dnn_names  = [(self.mpnn, "mpnn"), (self.transformer, "transformer"), (self.pre_yield, "pre_yield"), (self.embedding, "embedding")]
        
        for dnn, name in dnn_names:
            
            if mode == "save":
                    torch.save(dnn.state_dict(), path+"/"+name)

            else:
                dnn.load_state_dict(torch.load(path+"/"+name, map_location=device), strict=strict)
                
    def _make_graph(self, 
                    data:pd.DataFrame):
        
        for name in self.args["target"]:
            
            smiles_list = list(set([cpds for cpds in data[name]]))
            
            for smiles in smiles_list:
                
                self.graphs[smiles] = MolGraph(smiles)
                
    def _insert_graph(self, graph):
        self.graphs = graph
                
    def forward(self, 
                data:pd.DataFrame):
        
        data = data.reset_index(drop=True)
        max_atoms_num   = self.args["max_length"]
        
        add_tensor,  add_scope  = self.mpnn([self.graphs[smi] for smi in data["additive_smiles"]])
        ani_tensor,  ani_scope  = self.mpnn([self.graphs[smi] for smi in data["aniline_smiles"]])
        lig_tensor,  lig_scope  = self.mpnn([self.graphs[smi] for smi in data["ligand_smiles"]])
        pro_tensor,  pro_scope  = self.mpnn([self.graphs[smi] for smi in data["product_smiles"]])  
        aryl_tensor, aryl_scope = self.mpnn([self.graphs[smi] for smi in data["aryl_halide_smiles"]])
        base_tensor, base_scope = self.mpnn([self.graphs[smi] for smi in data["base_smiles"]])
        
        add_tensor  += self.embedding(torch.LongTensor([0]*add_tensor.shape[0]).to(self.device))
        ani_tensor  += self.embedding(torch.LongTensor([1]*ani_tensor.shape[0]).to(self.device))
        lig_tensor  += self.embedding(torch.LongTensor([2]*lig_tensor.shape[0]).to(self.device))
        pro_tensor  += self.embedding(torch.LongTensor([3]*pro_tensor.shape[0]).to(self.device))
        aryl_tensor += self.embedding(torch.LongTensor([4]*aryl_tensor.shape[0]).to(self.device))
        base_tensor += self.embedding(torch.LongTensor([5]*base_tensor.shape[0]).to(self.device))
        reactions, masks, self.reaction_scope = list(), list(), list()
        
        for num in range(data.shape[0]):
            
            add_hiddens  = add_tensor.narrow(0,  add_scope[num][0],  add_scope[num][1])
            ani_hiddens  = ani_tensor.narrow(0,  ani_scope[num][0],  ani_scope[num][1])
            lig_hiddens  = lig_tensor.narrow(0,  lig_scope[num][0],  lig_scope[num][1])
            pro_hiddens  = pro_tensor.narrow(0,  pro_scope[num][0],  pro_scope[num][1])
            aryl_hiddens = aryl_tensor.narrow(0, aryl_scope[num][0], aryl_scope[num][1])
            base_hiddens = base_tensor.narrow(0, base_scope[num][0], base_scope[num][1])
            
            atoms_num = add_hiddens.shape[0] + ani_hiddens.shape[0] + lig_hiddens.shape[0] + pro_hiddens.shape[0] + aryl_hiddens.shape[0] + base_hiddens.shape[0]
            zero_padding = torch.zeros(max_atoms_num - atoms_num, self.args["hidden_size"]).to(self.device)
            reaction = torch.cat([ani_hiddens, aryl_hiddens, add_hiddens, lig_hiddens, base_hiddens, pro_hiddens, zero_padding]).unsqueeze(dim=0)
            mask = torch.zeros(1, 1, max_atoms_num, max_atoms_num).to(self.device)
            mask[:, :, :, :atoms_num] = 1
            self.reaction_scope.append(atoms_num)
            reactions.append(reaction)
            masks.append(mask)
        
        output   = torch.cat(reactions, dim=0)
        att_mask = torch.cat(masks, dim=0)
        output   = self.transformer(output, att_mask)
        outputs = list()
        
        for num, scope in enumerate(self.reaction_scope):
            outputs.append(torch.sum(output[num][:scope, :], axis=0, keepdim=True))
            
        yields = torch.cat(outputs, axis=0)
        yields = self.pre_yield(yields)
        
        return yields
    
class SuzukiMiyauraModel(BuchwaldHartwigModel):
    
    def forward(self, 
                data:pd.DataFrame
                ):
        
        data = data.reset_index(drop=True)
        max_atoms_num   = self.args["max_length"]
            
        react1_tensor, react1_scope =  self.mpnn([self.graphs[smi] for smi in data["Reactant_1_SMILES"]])
        react2_tensor, react2_scope =  self.mpnn([self.graphs[smi] for smi in data["Reactant_2_SMILES"]])
        lig_tensor,    lig_scope    =  self.mpnn([self.graphs[smi] for smi in data["Ligand_SMILES"]])
        reag_tensor,   reag_scope   =  self.mpnn([self.graphs[smi] for smi in data["Reagent_SMILES"]]) 
        solv_tensor,   solv_scope   =  self.mpnn([self.graphs[smi] for smi in data["Solvent_SMILES"]])
        pro_tensor,    pro_scope    =  self.mpnn([self.graphs[smi] for smi in data["Product_SMILES"]])
        
        react1_tensor += self.embedding(torch.LongTensor([0]*react1_tensor.shape[0]).to(self.device))
        react2_tensor += self.embedding(torch.LongTensor([1]*react2_tensor.shape[0]).to(self.device))
        lig_tensor    += self.embedding(torch.LongTensor([2]*lig_tensor.shape[0]).to(self.device))
        pro_tensor    += self.embedding(torch.LongTensor([3]*pro_tensor.shape[0]).to(self.device))
        solv_tensor   += self.embedding(torch.LongTensor([4]*solv_tensor.shape[0]).to(self.device))
        reag_tensor   += self.embedding(torch.LongTensor([5]*reag_tensor.shape[0]).to(self.device))
        reactions, masks, self.reaction_scope, self.reaction_center = list(), list(), list(), list()
        
        for num in range(data.shape[0]):
            
            react1_hiddens = react1_tensor.narrow(0, react1_scope[num][0], react1_scope[num][1])
            react2_hiddens = react2_tensor.narrow(0, react2_scope[num][0], react2_scope[num][1])
            lig_hiddens    = lig_tensor.narrow(0, lig_scope[num][0], lig_scope[num][1])
            pro_hiddens    = pro_tensor.narrow(0, pro_scope[num][0], pro_scope[num][1])
            solv_hiddens   = solv_tensor.narrow(0, solv_scope[num][0], solv_scope[num][1])
            reag_hiddens   = reag_tensor.narrow(0, reag_scope[num][0], reag_scope[num][1])
            
            atoms_num = react1_scope[num][1] + react2_scope[num][1] + lig_scope[num][1] + pro_scope[num][1] + solv_scope[num][1] + reag_scope[num][1]
            zero_padding = torch.zeros(max_atoms_num - atoms_num, self.args["hidden_size"]).to(self.device)
            reaction = torch.cat([react1_hiddens, react2_hiddens, lig_hiddens, reag_hiddens, solv_hiddens, pro_hiddens, zero_padding]).unsqueeze(dim=0)
            mask = torch.zeros(1, 1, max_atoms_num, max_atoms_num).to(self.device)
            mask[:, :, :, :atoms_num] = 1
            self.reaction_scope.append(atoms_num)
            reactions.append(reaction)
            masks.append(mask)
        
        output   = torch.cat(reactions, dim=0)
        att_mask = torch.cat(masks, dim=0)
        output   = self.transformer(output, att_mask)
        
        outputs = list()
        for num, scope in enumerate(self.reaction_scope):
            outputs.append(torch.sum(output[num][:scope, :], axis=0, keepdim=True))
            
        yields = torch.cat(outputs, axis=0)
        yields = self.pre_yield(yields)
        return yields
        
    
class TransformerModel:
    
    def __init__(self, 
                mpnn_args:dict, 
                tr_args:dict, 
                dnn_args:dict, 
                args:dict, 
                layer:list, 
                type:str, 
                sample_num:int):
        
        self.args = args
        self.device = device()
        self.model  = self._model_type(mpnn_args, tr_args, dnn_args, args, layer, type, sample_num)
        self.criterion = nn.MSELoss().to(self.device)
        
        if sample_num:
            self.optimizer = torch.optim.Adam([{"params":list(self.model.mpnn.parameters()), "lr":1e-5}, {"params":  list(self.model.pre_yield.parameters()) + list(self.model.transformer.parameters()) + list(self.model.embedding.parameters()), "lr":1e-4}])
        else:
            self.optimizer = torch.optim.Adam(params=list(self.model.mpnn.parameters()) + list(self.model.transformer.parameters()) + list(self.model.embedding.parameters()) + list(self.model.pre_yield.parameters()),  lr=1e-4)
            
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args["step_size"], gamma=args["gamma"])
        self.scaler = torch.cuda.amp.GradScaler()
        
    def _model_type(self, mpnn_args, tr_args, dnn_args, args, layer, type, sample_num):
        
        if type == "buchwald-hartwig":
            return BuchwaldHartwigModel(mpnn_args, tr_args, dnn_args, args, layer, sample_num)
        
        elif type == "suzuki-miyaura":
            return SuzukiMiyauraModel(mpnn_args, tr_args, dnn_args, args, layer, sample_num)
        
        else:
            raise ValueError(f"{type} model doesn't exist")
            
    def save_load_model(self, 
                        path:str, 
                        mode:str=None, 
                        device:str=None, 
                        strict:bool=True):
        
        self.model._save_load(path, mode, device, strict)
    
    def make_graph(self, datas):
        
        for data in datas:
            
            self.model._make_graph(data)
            
    def batch_train(self, 
                    input:pd.DataFrame):

        self.optimizer.zero_grad()
        self.model._set_train_eval("train")
        with torch.cuda.amp.autocast():
            observe = torch.FloatTensor(np.array(round(input["yield"]/100, 3)).reshape(-1, 1)).to(self.device)
            predict = self.model(input)
            if self.args["loss"] == "weight":
                weight  = torch.FloatTensor(np.array(input["weight"])).reshape(-1, 1).to(self.device)
                loss = self._weighted_mse(predict, observe, weight)
            elif self.args["loss"] == "normal":
                loss = self.criterion(predict, observe)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
    def _weighted_mse(self, predict, observe, weight):
        return torch.mean(torch.square(observe - predict)*weight)
        
    def test(self, 
            data:pd.DataFrame):
        
        self.model._set_train_eval(sets="eval")
        predict = self.model(data)
        predict = tensor_to_numpy(torch.clamp(predict, min=0.0, max=1))
        observe = (data["yield"]/100).values.reshape(-1, 1)
        self.model._set_train_eval(sets="train")
        
        return predict, observe
    
    def _large_test(self, 
                    datas:list[pd.DataFrame]):
        
        pre_li, obs_li = list(), list()
        for data in datas:
            pre, obs = self.test(data)
            pre_li.append(pre)
            obs_li.append(obs)
            
        pre_np, obs_np = np.vstack(pre_li), np.vstack(obs_li)
        
        return pre_np*100, obs_np*100
            
        
    def learning_process(self, 
                        train:pd.DataFrame, 
                        test:pd.DataFrame):
        
        self.make_graph([train, test])
        result_all_list = list()
        pre_tr_list = list()
        pre_te_list = list()
        epoch_list  = list()
        self.model._set_train_eval(sets="train")
        
        for epoch in range(self.args["epoch_num"]):
            self.epoch = epoch
            batches = make_batch(self.args["batch_size"], train, epoch)

            for batch in batches:
    
                self.batch_train(batch)
            
            if self.args["epoch_num"] - epoch <= self.args["epoch_num"]:
                
                #Since train and test data are too large to be put in memory all at once, there are divided into some parts.
                tr_data = [train.loc[i:i+int(train.shape[0]/50)-1, :] for i in range(0, train.shape[0], int(train.shape[0]/50))] 
                te_data = [test.loc[i:i+int(test.shape[0]/20)-1, :] for i in range(0, test.shape[0], int(test.shape[0]/20))]
                
                pre_tr, obs_tr = self._large_test(tr_data)
                pre_te, obs_te = self._large_test(te_data)
                
                pre_tr_list.append(pre_tr)
                pre_te_list.append(pre_te)
                epoch_list.append(epoch+1)
                
                tr_R2, tr_RMSE, tr_MAE = r2_rmse_mae(pre_tr, obs_tr)
                te_R2, te_RMSE, te_MAE = r2_rmse_mae(pre_te, obs_te)
                result_all_list.append([tr_R2, tr_RMSE, tr_MAE, te_R2, te_RMSE, te_MAE, epoch+1])
    
            self.scheduler.step()
            
        pre_tr, pre_te = np.hstack(pre_tr_list), np.hstack(pre_te_list)
        pre_tr, pre_te = pd.DataFrame(pre_tr, columns=["predict_epoch%s"%(num) for num in epoch_list]), pd.DataFrame(pre_te, columns=["predict_epoch%s"%(num) for num in epoch_list])
        return pd.DataFrame(result_all_list, columns=["train_R2", "train_RMSE", "train_MAE", "test_R2", "test_RMSE", "test_MAE", "train_num"]), pre_tr, pre_te
        
    