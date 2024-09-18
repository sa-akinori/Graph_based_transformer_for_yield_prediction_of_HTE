# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", default=0, choices=[0, 3000000], type=int, help="This number represents number of compounds were used to train Contrastive Learning. We can select a number in [0, 3000000]")
parser.add_argument("--target", default="buchwald-hartwig", choices=["buchwald-hartwig", "suzuki-miyaura"], type=str, help="")
parser.add_argument("--ablation", default="MPNN", choices=["MPNN", "Transformer", "Random"], type=str)
parse_args = parser.parse_args()
    
def necessary_member(
    args:dict
    ):
    
    if args.target == "buchwald-hartwig":
        columns = ["aniline_smiles", "additive_smiles", "aryl_halide_smiles", "ligand_smiles", "base_smiles", "product_smiles"]
        names = [f"Test{num}" for num in range(1, 5)] + [f"new_Test{num}" for num in range(1, 21)]
        
    elif args.target == "suzuki-miyaura":
        columns = ["Organoboron_SMILES", "Organic_Halide_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES", "Product_SMILES"]
        names = [f"Test{num}" for num in range(1, 13)]
            
    else:
        raise ValueError(f"{args.target} is invalid")
    
    columns = [f"washed_{col}" for col in columns]
    return columns, names

seeds = [0, 2, 3, 4, 5, 7, 8, 9, 14, 17,
        19, 23, 25, 27, 28, 31, 33, 34, 42, 44,
        46, 49, 51, 52, 53, 54, 55, 56, 57, 61,
        62, 63, 64, 65, 69, 72, 74, 75, 76, 77,
        79, 81, 83, 87, 91, 93, 94, 96, 97, 100]

seeds_list  =  [[0, 5, 33, 42, 49, 51, 53, 62, 65, 97],
                [94, 31, 74, 75, 44, 14, 96, 25, 55, 76],
                [2, 64, 61, 17, 79, 81, 72, 4, 77, 3],
                [91, 27, 46, 34, 9, 8, 19, 83, 23, 56],
                [87, 28, 100, 57, 54, 7, 69, 93, 63, 52]]

if __name__=="__main__":  
    
    if sys.argv:
        del sys.argv[1:]
        
    columns, names = necessary_member(parse_args)
    
    if parse_args.ablation == "MPNN":

        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in seeds:
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/MPNN/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        each_result = list()
        for name in names:
            r2_list = list()
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed_list in seeds_list:
                for seed in seed_list:
                    pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/MPNN/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                    pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                pre_te = np.concatenate(pre_te_list, axis=1)
                r2, rmse, mae = r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))
                r2_list.append(r2)
            result.append([name, np.mean(r2_list), np.std(r2_list)])
            each_result.append([name] + r2_list)
        result = pd.DataFrame(result, columns=["name", "R2_mean", "R2_std"])
        each_result = pd.DataFrame(each_result, columns=["name", "R2_mean1", "R2_mean2", "R2_mean3", "R2_mean4", "R2_mean5"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        each_result.to_csv(f"DataFrame/{parse_args.target}/evaluation/MPNN/sample{parse_args.sample_num}/extrapolation/each_ensemble.csv")
        
        
    elif parse_args.ablation == "Transformer":

        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in seeds:
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/Transformer/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        each_result = list()
        for name in names:
            r2_list = list()
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed_list in seeds_list:
                for seed in seed_list:
                    pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/Transformer/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                    pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                pre_te = np.concatenate(pre_te_list, axis=1)
                r2, rmse, mae = r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))
                r2_list.append(r2)
            result.append([name, np.mean(r2_list), np.std(r2_list)])
            each_result.append([name] + r2_list)
        result = pd.DataFrame(result, columns=["name", "R2_mean", "R2_std"])
        each_result = pd.DataFrame(each_result, columns=["name", "R2_mean1", "R2_mean2", "R2_mean3", "R2_mean4", "R2_mean5"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        each_result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Transformer/sample{parse_args.sample_num}/extrapolation/each_ensemble.csv")
        
        
    elif parse_args.ablation == "Random":

        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in seeds:
                try:
                    predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/Random/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                    predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
                except:
                    print(seed)
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Random/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Random/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        each_result = list()
        for name in names:
            r2_list = list()
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed_list in seeds_list:
                for seed in seed_list:
                    pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/Random/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                    pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                pre_te = np.concatenate(pre_te_list, axis=1)
                r2, rmse, mae = r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))
                r2_list.append(r2)
            result.append([name, np.mean(r2_list), np.std(r2_list)])
            each_result.append([name] + r2_list)
        result = pd.DataFrame(result, columns=["name", "R2_mean", "R2_std"])
        each_result = pd.DataFrame(each_result, columns=["name", "R2_mean1", "R2_mean2", "R2_mean3", "R2_mean4", "R2_mean5"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/Random/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Random/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        each_result.to_csv(f"DataFrame/{parse_args.target}/evaluation/Random/sample{parse_args.sample_num}/extrapolation/each_ensemble.csv")
        