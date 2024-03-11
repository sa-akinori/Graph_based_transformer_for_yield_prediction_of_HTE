# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
import itertools
from main import necessary_member
from functions import r2_rmse_mae

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", default=0, type=int, help="This number represents number of compounds were used to train Contrastive Learning. We can select a number in [0, 3000000]")
parser.add_argument("--target", default="buchwald-hartwig", type=str, help="")
parser.add_argument("--extrapolation", default="True", type=str, help="")
parser.add_argument("--model_name", default="Contrastive", type=str, help="")
parse_args = parser.parse_args()
    
if __name__=="__main__": 
    
    seeds_list  =  [[0, 5, 33, 42, 49, 51, 53, 62, 65, 97],
                    [94, 31, 74, 75, 44, 14, 96, 25, 55, 76],
                    [2, 64, 61, 17, 79, 81, 72, 4, 77, 3],
                    [91, 27, 46, 34, 9, 8, 19, 83, 23, 56],
                    [87, 28, 100, 57, 54, 7, 69, 93, 63, 52]]
    
    if sys.argv:
        del sys.argv[1:]
        
    columns, names = necessary_member(parse_args)
    
    if eval(parse_args.extrapolation):
        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in list(itertools.chain.from_iterable(seeds_list)):
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        for i, seeds in enumerate(seeds_list):
            result = list()
            for name in names:
                pre_te_list   = list()
                test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
                for seed in seeds:
                    pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                    pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                pre_te = np.concatenate(pre_te_list, axis=1)
                result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
                
            result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
            os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
            result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/ensemble{i}.csv")
        
        #ensemble_accuracy(calculate mean and std of ensemble)
        mean_std = list()
        for name in names:
            results_list = list()
            for i in range(len(seeds_list)):
                result = pd.read_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/ensemble{i}.csv", index_col=0)
                results_list.append(result.query("name == @name"))
            results = np.array(pd.concat(results_list).loc[:, "R2"])
            results_mean = np.mean(results)
            results_std  = np.std(results)
            mean_std.append([name, results_mean, results_std])
            
        final_result = pd.DataFrame(mean_std, columns=["name", "R2_mean", "R2_std"])
        final_result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        
        #big ensemble_accuracy
        result = list()
        for name in names:
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed in list(itertools.chain.from_iterable(seeds_list)):
                pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
            pre_te = np.concatenate(pre_te_list, axis=1)
            result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
            
        result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/big_ensemble.csv")
        
    else:
        #mean_accuracy
        for i, seeds in enumerate(seeds_list):
            for ratio in parse_args.train_size:
                result = list()
                for name in names:
                    predicts = list()
                    for seed in seeds:
                        predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/Ratio{ratio}/seed{seed}/{name}.csv", index_col=0)
                        predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
                    predicts = np.concatenate(predicts)
                    result.append([name, np.mean(predicts), np.std(predicts)])
                result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
                os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/", exist_ok=True)
                result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/mean{i}.csv")
        
        #ensemble_accuracy
        for i, seeds in enumerate(seeds_list):
            for ratio in parse_args.train_size:
                result = list()
                for name in names:
                    pre_te_list   = list()
                    test = pd.read_csv(f"DataFrame/{parse_args.target}/test/Ratio{ratio}/{name}.csv", index_col=0)
                    for seed in seeds:
                        pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/Ratio{ratio}/seed{seed}/{name}_test.csv", index_col=0)
                        pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                    pre_te = np.concatenate(pre_te_list, axis=1)
                    result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
                    
                result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
                os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/", exist_ok=True)
                result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/ensemble{i}.csv")
        
        
# summation = list()
# for name in ["one_hot", "random"]:
#     sums = list()
#     a = pd.read_csv(f"/home/sato_akinori/For_Github/BuchwaldHartwig/DataFrame/buchwald-hartwig/evaluation/{name}/ensemble.csv", index_col=0)
#     for i, test in enumerate([f"Test{i}" for i in range(1, 5)]):
#         sums.extend([a.loc[i, "te_R2_mean"], a.loc[i, "te_R2_std"]])
#     summation.append([name] + sums)
# for name in["Random", "MPNN", "Transformer", "Contrastive"]:
#     sums = list()
#     a = pd.read_csv(f"/home/sato_akinori/For_Github/BuchwaldHartwig/DataFrame/buchwald-hartwig/evaluation/{name}/sample0/extrapolation/ensemble.csv", index_col=0)
#     for i, test in enumerate([f"Test{i}" for i in range(1, 5)]):
#         sums.extend([a.loc[i, "R2_mean"], a.loc[i, "R2_std"]])
#     summation.append([f"{name}_sample0"] + sums)
# for name in ["MPNN", "Contrastive"]:
#     sums = list()
#     a = pd.read_csv(f"/home/sato_akinori/For_Github/BuchwaldHartwig/DataFrame/buchwald-hartwig/evaluation/{name}/sample3000000/extrapolation/ensemble.csv", index_col=0)
#     for i, test in enumerate([f"Test{i}" for i in range(1, 5)]):
#         sums.extend([a.loc[i, "R2_mean"], a.loc[i, "R2_std"]])
#     summation.append([f"{name}_sample3000000"] + sums)
# b = pd.DataFrame(summation, columns = ["name"] + list(itertools.chain.from_iterable([[f"Test{i}_R2_mean", f"Test{i}_R2_std"] for i in range(1, 5)])))
# b.to_csv("/home/sato_akinori/For_Github/BuchwaldHartwig/DataFrame/buchwald-hartwig/evaluation/ensemble_summation.csv")