# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
from functions import r2_rmse_mae

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", default=0, choices=[0, 3000000], type=int, help="This number represents number of compounds were used to train Contrastive Learning. We can select a number in [0, 3000000]")
parser.add_argument("--target", default="buchwald-hartwig", choices=["buchwald-hartwig", "suzuki-miyaura"], type=str, help="")
parser.add_argument("--ablation", default="MPNN", choices=["MPNN", "Transformer"], type=str)
parse_args = parser.parse_args()
    
def necessary_member(
    args:dict
    ):
    
    if args.target == "buchwald-hartwig":
        columns = ["aniline_smiles", "additive_smiles", "aryl_halide_smiles", "ligand_smiles", "base_smiles", "product_smiles"]
        names = [f"Test{num}" for num in range(1, 5)]
        
    elif args.target == "suzuki-miyaura":
        columns = ["Organoboron_SMILES", "Organic_Halide_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES", "Product_SMILES"]
        names = [f"Test{num}" for num in range(1, 13)]
            
    else:
        raise ValueError(f"{args.target} is invalid")
    
    columns = [f"washed_{col}" for col in columns]
    return columns, names

if __name__=="__main__":  
    
    if sys.argv:
        del sys.argv[1:]
        
    columns, names = necessary_member(parse_args)
    
    if parse_args.ablation == "MPNN":

        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in [0, 5, 33, 42, 49, 51, 53, 62, 65, 97]:
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/MPNN/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        for name in names:
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed in [0, 5, 33, 42, 49, 51, 53, 62, 65, 97]:
                pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/MPNN/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
            pre_te = np.concatenate(pre_te_list, axis=1)
            result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
            
        result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        
        
    elif parse_args.ablation == "Transformer":

        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in [0, 5, 33, 42, 49, 51, 53, 62, 65, 97]:
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/Transformer/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        for name in names:
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed in [0, 5, 33, 42, 49, 51, 53, 62, 65, 97]:
                pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/Transformer/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
            pre_te = np.concatenate(pre_te_list, axis=1)
            result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
            
        result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        