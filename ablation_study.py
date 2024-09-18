import os
import sys
import argparse
import pandas as pd
from model import *
from graphconvolution import *
from functions import SmilesToOEGraphMol
from typing import List

def NumAtoms(
    data:pd.DataFrame, 
    targets:List[str],
    )->pd.DataFrame:
    """
    This function is to calculate the heavy atoms number of molecules in a chemical reaction to decide the fixed length of input.
    """
    atoms_list = [0]* data.shape[0]
    for target in targets:
        atoms_list = [SmilesToOEGraphMol(smi).NumAtoms() + num if smi != "None" else num for num, smi in zip(atoms_list, data[target])]
        
    data["atoms_num"] = atoms_list
    return data

def necessary_member():
    
    if parse_args.target == "buchwald-hartwig":
        columns = ["aniline_smiles", "additive_smiles", "aryl_halide_smiles", "ligand_smiles", "base_smiles", "product_smiles"]
        
        if parse_args.extrapolation_role == "additive":
            names = [f"Test{num}" for num in range(1, 5)]
        
        elif parse_args.extrapolation_role == "three-roles":
            names = [f"new_Test{num}" for num in range(1, 21)]

    elif parse_args.target == "suzuki-miyaura":
        columns = ["Organoboron_SMILES", "Organic_Halide_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES", "Product_SMILES"]
        names = [f"Test{num}" for num in range(1, 13)]
            
    else:
        raise ValueError(f"{parse_args.target} is invalid")
    
    columns = [f"washed_{col}" for col in columns]
    return columns, names

def optimize_batch_size():
    
    if parse_args.target=="buchwald-hartwig":
        
        if parse_args.extrapolation_role == "additive":
            return 16
        
        elif parse_args.extrapolation_role == "three-roles":
            return 8
    
    if parse_args.target=="suzuki-miyaura":
        return 16
            
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=200, type=int, help="number of learning times")
parser.add_argument("--start_eval_epoch", default=0, type=int)
parser.add_argument("--step_size", default=150, type=int, help="a parameter(step_size) in optim.lr_scheduler.StepLR")
parser.add_argument("--gamma", default=0.1, type=float, help="a parameter(gamma) in optim.lr_scheduler.StepLR")
parser.add_argument("--hidden_size", default=400, type=int, help="")
parser.add_argument("--sample_num", default=0, type=int, choices=[0, 3000000], help="this number represents number of compounds were used to train Contrastive Learning.")
parser.add_argument("--seed", default=0, type=int, help="initializing weights of a model")
parser.add_argument("--target", default="buchwald-hartwig", type=str, choices=["buchwald-hartwig", "suzuki-miyaura"], help="")
parser.add_argument("--extrapolation_role", default="additive", choices=["additive", "three-roles"], type=str, help="")
parser.add_argument("--tf_layer_num", default=3, type=int, help="number of iterations of transformer encoder")
parser.add_argument("--ablation", choices=["MPNN", "Transformer", "Random"], type=str)
parse_args = parser.parse_args()

if __name__=="__main__":  
    
    if sys.argv:
        del sys.argv[1:]
    
    columns, names = necessary_member()
    batch_size = optimize_batch_size()
    network = [400, 1024, 512, 256, 1]#all model has the same best hyperparameter
    
    mpnn_args = {"hidden_size":400, "bias":True, "dropout_ratio":0.0, "depth":2, "agn_num":4, "target":parse_args.target}
    tf_args   = {"multi_head_num":10, "hidden_size":400, "dropout_ratio":0.1, "tf_norm":"LayerNorm", "tf_repeat":parse_args.tf_layer_num}
    dnn_args  = {"bias":True, "dropout_ratio":0.1}
        
    if parse_args.ablation in ["MPNN", "Transformer"]:
        for name in names:
                
            train, test = pd.read_csv(f"DataFrame/{parse_args.target}/train/{name}.csv", index_col=0), pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            train, test = NumAtoms(train, columns), NumAtoms(test, columns)
            train["reaction_id"], test["reaction_id"] = train.index, test.index
            args = {"seed":parse_args.seed, "epoch_num":parse_args.epoch, "start_eval_epoch":parse_args.start_eval_epoch, "step_size":parse_args.step_size, "gamma":parse_args.gamma,
                    "hidden_size":parse_args.hidden_size, "batch_size":batch_size, "columns":columns, "embedding_species":"mol2vec", "max_length":max(max(train["atoms_num"]), max(test["atoms_num"]))}
            if parse_args.ablation == "MPNN":
                model = MPNNModel(mpnn_args, dnn_args, args, network, type=parse_args.target, sample_num=parse_args.sample_num)
            
            elif parse_args.ablation == "Transformer":
                model = TransformerModel(tf_args, dnn_args, args, network, type=parse_args.target)
                
            process, pre_tr, pre_te = model.learning_process(train, test)
            
            os.makedirs(f"DataFrame/{parse_args.target}/process/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            process.to_csv(f"DataFrame/{parse_args.target}/process/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}.csv")
            os.makedirs(f"DataFrame/{parse_args.target}/model/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}", exist_ok=True)
            model.save_load_model(f"DataFrame/{parse_args.target}/model/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}/", mode="save")
            os.makedirs(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            pre_tr.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_train.csv")
            pre_te.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_test.csv")
        
    elif parse_args.ablation == "Random":
        for name in names:
                
            train, test = pd.read_csv(f"DataFrame/{parse_args.target}/train/{name}.csv", index_col=0), pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            train, test = NumAtoms(train, columns), NumAtoms(test, columns)
            train["reaction_id"], test["reaction_id"] = train.index, test.index
            args = {"seed":parse_args.seed, "epoch_num":parse_args.epoch, "start_eval_epoch":parse_args.start_eval_epoch, "step_size":parse_args.step_size, "gamma":parse_args.gamma,
                    "hidden_size":parse_args.hidden_size, "batch_size":batch_size, "columns":columns, "embedding_species":"random", "max_length":max(max(train["atoms_num"]), max(test["atoms_num"]))}
            
            model = TransformerModel(tf_args, dnn_args, args, network, type=parse_args.target)
            
            process, pre_tr, pre_te = model.learning_process(train, test)
            
            os.makedirs(f"DataFrame/{parse_args.target}/process/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            process.to_csv(f"DataFrame/{parse_args.target}/process/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}.csv")
            os.makedirs(f"DataFrame/{parse_args.target}/model/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}", exist_ok=True)
            model.save_load_model(f"DataFrame/{parse_args.target}/model/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}/", mode="save")
            os.makedirs(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/", exist_ok=True)
            pre_tr.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_train.csv")
            pre_te.to_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.ablation}/sample{parse_args.sample_num}/seed{parse_args.seed}/{name}_test.csv")