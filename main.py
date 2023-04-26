# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd
from model import *
from graphconvolution import *
from functions import SmilesToOEGraphMol

def NumAtoms(
    data:pd.DataFrame, 
    targets:list[str]
    ):
    """
    各反応データの原子数を計算(固定長を決定するため)
    """
    atoms_list = [0]* data.shape[0]
    for target in targets:
        atoms_list = [SmilesToOEGraphMol(smi).NumAtoms() + num if smi != "None" else num for num, smi in zip(atoms_list, data[target])]
        
    data["atoms_num"] = atoms_list
    return data

def necessary_member(
    args:dict
    ):
    
    if args.target == "buchwald-hartwig":
        columns = ["aniline_smiles", "additive_smiles", "aryl_halide_smiles", "ligand_smiles", "base_smiles", "product_smiles"]
        
        if eval(parse_args.extrapolation):
            names = [f"Test{num}" for num in range(1, 5)]
            
        else:
            names = ["FullCV_01", "FullCV_02", "FullCV_03", "FullCV_04", "FullCV_05", "FullCV_06", "FullCV_07", "FullCV_08", "FullCV_09", "FullCV_10"]
        
    elif args.target == "suzuki-miyaura":
        columns = ["Organoboron_SMILES", "Organic_Halide_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES", "Product_SMILES"]
        
        if eval(parse_args.extrapolation):
            names = [f"Test{num}" for num in range(1, 13)]
            
        else:
            names = ["FullCV_01", "FullCV_02", "FullCV_03", "FullCV_04", "FullCV_05", "FullCV_06", "FullCV_07", "FullCV_08", "FullCV_09", "FullCV_10"]
            
    else:
        raise ValueError(f"{args.target} is invalid")
    
    columns = [f"washed_{col}" for col in columns]
    return columns, names

def output_networks(
    args:dict
    ):
    
    if args.target == "buchwald-hartwig" and args.sample_num==3000000:
        return [400, 1024, 1]
    
    else:
        return [400, 1024, 512, 256, 1]
    


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=200, type=int, help="number of learning times")
parser.add_argument("--step_size", default=150, type=int, help="a parameter(step_size) in optim.lr_scheduler.StepLR")
parser.add_argument("--gamma", default=0.1, type=float, help="a parameter(gamma) in optim.lr_scheduler.StepLR")
parser.add_argument("--hidden_size", default=400, type=int, help="")
parser.add_argument("--sample_num", default=0, type=int, choices=[0, 3000000], help="this number represents number of compounds were used to train Contrastive Learning.")
parser.add_argument("--batch_size", default=16, type=int, help="batch size used in training data")
parser.add_argument("--seed", default=0, choices=[0, 5, 33, 42, 49, 51, 53, 62, 65, 97], type=int, help="initializing weights of a model")
parser.add_argument("--target", default="buchwald-hartwig", type=str, choices=["buchwald-hartwig", "suzuki-miyaura"], help="")
parser.add_argument("--extrapolation", default="True", type=str, help="")
parser.add_argument("--loss", default="normal", type=str, help="loss function used in training")
parser.add_argument("--train_size", default="Ratio70", type=str, choices=["Ratio70", "Ratio50", "Ratio30", "Ratio20", "Ratio10", "Ratio5.0", "Ratio2.5"], help="training data size in random-split")
parser.add_argument("--tf_layer_num", default=3, type=int, help="number of iterations oftransformer encoder")
parser.add_argument("--model_name", default="Contrastive", type=str, help="this is used to save some files")
parse_args = parser.parse_args()

if __name__=="__main__":  
    
    if sys.argv:
        del sys.argv[1:]
    
    columns, names = necessary_member(parse_args)
    network = output_networks(parse_args)
    
    mpnn_args = {"hidden_size":400, "bias":True, "dropout_ratio":0.0, "depth":2, "agn_num":4, "target":parse_args.target}
    tf_args   = {"multi_head_num":10, "hidden_size":400, "dropout_ratio":0.1, "tf_norm":"LayerNorm", "tf_repeat":parse_args.tf_layer_num}
    dnn_args  = {"bias":True, "dropout_ratio":0.1}
    
    if eval(parse_args.extrapolation):
        
        for name in names:
                
            train, test = pd.read_csv(f"DataFrame/{parse_args.target}/train/{name}.csv", index_col=0), pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            train, test = NumAtoms(train, columns), NumAtoms(test, columns)
            train["reaction_id"], test["reaction_id"] = train.index, test.index
            args = {"seed":parse_args.seed, "epoch_num":parse_args.epoch, "step_size":parse_args.step_size, "gamma":parse_args.gamma, "hidden_size":parse_args.hidden_size, 
                    "batch_size":parse_args.batch_size, "columns":columns, "loss":parse_args.loss, "max_length":max(max(train["atoms_num"]), max(test["atoms_num"]))}
            model = TransformerModel(mpnn_args, tf_args, dnn_args, args, network, type=parse_args.target, sample_num=parse_args.sample_num)
            process, pre_tr, pre_te = model.learning_process(train, test)
            
            os.makedirs(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            process.to_csv(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}.csv")
            os.makedirs(f"DataFrame/{parse_args.target}/model/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}", exist_ok=True)
            model.save_load_model(f"DataFrame/{parse_args.target}/model/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}/", mode="save")
            os.makedirs(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            pre_tr.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_train.csv")
            pre_te.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_test.csv")
            
    
    elif not eval(parse_args.extrapolation):
        
        for name in names:
            
            train, test = pd.read_csv(f"DataFrame/{parse_args.target}/train/{parse_args.train_size}/{name}.csv"), pd.read_csv(f"DataFrame/{parse_args.target}/test/{parse_args.train_size}/{name}.csv")
            train, test = NumAtoms(train, columns), NumAtoms(test, columns)
            train["reaction_id"], test["reaction_id"] = train.index, test.set_index
            args = {"seed":parse_args.seed, "epoch_num":parse_args.epoch, "step_size":parse_args.step_size, "gamma":parse_args.gamma, "hidden_size":parse_args.hidden_size, 
                    "batch_size":parse_args.batch_size, "target":columns, "loss":parse_args.loss, "max_length":max(max(train["atoms_num"]), max(test["atoms_num"]))}
            model = TransformerModel(mpnn_args, tf_args, dnn_args, args, network, type=parse_args.target, sample_num=parse_args.sample_num)
            process, pre_tr, pre_te = model.learning_process(train, test)
            
            os.makedirs(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/", exist_ok=True)
            process.to_csv(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/{name}.csv")
            os.makedirs(f"DataFrame/{parse_args.target}/model/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/{name}", exist_ok=True)
            model.save_load_model(f"DataFrame/{parse_args.target}/model/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/{name}/", mode="save")
            os.makedirs(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/", exist_ok=True)
            pre_tr.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/{name}_train.csv")
            pre_te.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/{parse_args.train_size}/seed{parse_args.seed}/{name}_test.csv")
    
