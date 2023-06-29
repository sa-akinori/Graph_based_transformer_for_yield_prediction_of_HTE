# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import device

class BatchMultiHeadAttentionUnit(nn.Module):
    
    def __init__(self, args:dict):
        
        super(BatchMultiHeadAttentionUnit,self).__init__()
        
        self.args     = args
        self.head_num = args["multi_head_num"]
        self.act_func = nn.GELU()

        self.W_q = nn.Linear(args["hidden_size"], args["hidden_size"], bias=False)
        self.W_k = nn.Linear(args["hidden_size"], args["hidden_size"], bias=False)
        self.W_v = nn.Linear(args["hidden_size"], args["hidden_size"], bias=False)
        self.output   = nn.Linear(args["hidden_size"], args["hidden_size"])
        self.att_drop = nn.Dropout(p=args["dropout_ratio"])
        self.out_drop = nn.Dropout(p=args["dropout_ratio"])
        
    def forward(self, value, key, query, mask):
            
        query, key, value = self._split_head(self.W_q(query)), self._split_head(self.W_k(key)), self._split_head(self.W_v(value))
        mol_matrix, att_w = self._multi_head_attention(query, key, value, mask)
        mol_matrix = self._concat_head(mol_matrix)
        mol_matrix = self.output(mol_matrix)
        mol_matrix = self.act_func(mol_matrix)
        mol_matrix = self.out_drop(mol_matrix)
        self.att_weight = att_w #to get attention weight
        
        return mol_matrix
        
    def _split_head(self, x):
        
        x_batch, x_index, x_columns = x.size()
        new_x_columns = int(x_columns/self.head_num)
        new_x = torch.reshape(x, (x_batch, x_index, self.head_num, new_x_columns))

        return torch.transpose(new_x, 1, 2)

    def _concat_head(self, x):

        x_batch, x_head, x_index, x_columns = x.size()
        new_x_columns = int(x_columns*x_head)
        x     = torch.transpose(x, 1, 2)
        new_x = torch.reshape(x, (x_batch, x_index, new_x_columns))

        return new_x

    def _multi_head_attention(self, query, key, value, mask):
        
        depth  =  list(query.shape)[-1]
        query *=  depth**-0.5
        att_w  =  torch.matmul(query, torch.transpose(key, -2, -1))
        att_w  =  att_w.masked_fill(mask==0, float("-inf"))
        att_w  =  F.softmax(att_w, dim=-1)
        att_w  =  self.att_drop(att_w)
        att_hiddens = torch.matmul(att_w, value)

        return att_hiddens, att_w


class TransformerEncoderUnit(nn.Module):
    
    def __init__(self, args:dict):
        
        super(TransformerEncoderUnit, self).__init__()
        
        self.args = args
        self.device = device()
        self.self_attention = BatchMultiHeadAttentionUnit(args)
        self.ff_network     = nn.Sequential(nn.Linear(args["hidden_size"], 2048), nn.GELU(), nn.Dropout(p=args["dropout_ratio"]), nn.Linear(2048, args["hidden_size"]))
        
        if args["tf_norm"] == "BatchNorm":
            self.norm_sa = nn.BatchNorm1d(args["hidden_size"])
            self.norm_ff = nn.BatchNorm1d(args["hidden_size"])
            
        elif args["tf_norm"] == "LayerNorm":
            self.norm_sa = nn.LayerNorm(args["hidden_size"])
            self.norm_ff = nn.LayerNorm(args["hidden_size"])
        
    def _self_attention_part(self, input, mask):
        
        residue = input.clone()
        
        if self.args["tf_norm"] == "BatchNorm":
            input = torch.transpose(input, -2, -1)
            input = self.norm_sa(input)
            input = torch.transpose(input, -2, -1)
            
        else:
            input  = self.norm_sa(input)
        
        input  = self.self_attention(input.clone(), input.clone(), input.clone(), mask)
        input += residue
        
        return input
        
    def _feedfoward(self, input):
        
        residue = input.clone()
        if self.args["tf_norm"] == "BatchNorm":
            input = torch.transpose(input, -2, -1)
            input  = self.norm_ff(input)
            input = torch.transpose(input, -2, -1)
            
        else:
            input  = self.norm_ff(input)
        
        input  = self.ff_network(input)
        input += residue
        
        return input
        
    def forward(self, reaction, mask):
        
        reaction = self._self_attention_part(reaction, mask)
        reaction = self._feedfoward(reaction)
        
        return reaction
    
    def get_attention_weight(self):
        
        return self.self_attention.att_weight.to('cpu').detach().numpy().copy()


class TransformerEncoder(nn.Module):
    
    def __init__(self, args:dict):
        
        super(TransformerEncoder, self).__init__()
        self.args   = args
        self.device = device()
        self.units  = nn.ModuleList([TransformerEncoderUnit(args) for _ in range(args["tf_repeat"])])
        
        if args["tf_norm"] == "BatchNorm":
            self.last_norm = nn.BatchNorm1d(args["hidden_size"])
            
        elif args["tf_norm"] == "LayerNorm":
            self.last_norm = nn.LayerNorm(args["hidden_size"])
            
    def forward(self, output, mask):
        
        for i in range(len(self.units)):

            output = self.units[i](output, mask)
            
        if self.args["tf_norm"] == "BatchNorm":
            output = torch.transpose(output, -2, -1)
            output = self.last_norm(output)
            output = torch.transpose(output, -2, -1)
            
        else:
            output = self.last_norm(output)
        
        return output
    