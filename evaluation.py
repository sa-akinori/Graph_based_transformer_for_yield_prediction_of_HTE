# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
from main import necessary_member
from functions import r2_rmse_mae

parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", default=0, type=int, help="This number represents number of compounds were used to train Contrastive Learning. We can select a number in [0, 3000000]")
parser.add_argument("--seed", default=[0, 5, 33, 42, 49, 51, 53, 62, 65, 97], type=str, help="")
parser.add_argument("--target", default="buchwald-hartwig", type=str, help="")
parser.add_argument("--extrapolation", default="True", type=str, help="")
parser.add_argument("--model_name", default="Contrastive", type=str, help="")
parser.add_argument("--train_size", default=[70, 50, 30, 20, 10, 5.0, 2.5, 1.0], type=list, help="")
parse_args = parser.parse_args()
    
if __name__=="__main__":  
    
    if sys.argv:
        del sys.argv[1:]
        
    columns, names = necessary_member(parse_args)
    
    if eval(parse_args.extrapolation):
        #mean_accuracy
        result = list()
        for name in names:
            predicts = list()
            for seed in parse_args.seed:
                predict = pd.read_csv(f"DataFrame/{parse_args.target}/process2/{parse_args.model_name}/sample{parse_args.sample_num}/seed{seed}/{name}.csv", index_col=0)
                predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
            predicts = np.concatenate(predicts)
            result.append([name, np.mean(predicts), np.std(predicts)])
        result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation2/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation2/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/mean.csv")
        
        #ensemble_accuracy
        result = list()
        for name in names:
            pre_te_list   = list()
            test = pd.read_csv(f"DataFrame/{parse_args.target}/test/{name}.csv", index_col=0)
            for seed in parse_args.seed:
                pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict2/{parse_args.model_name}/sample{parse_args.sample_num}/seed{seed}/{name}_test.csv", index_col=0)
                pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
            pre_te = np.concatenate(pre_te_list, axis=1)
            result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
            
        result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
        os.makedirs(f"DataFrame/{parse_args.target}/evaluation2/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/", exist_ok=True)
        result.to_csv(f"DataFrame/{parse_args.target}/evaluation2/{parse_args.model_name}/sample{parse_args.sample_num}/extrapolation/ensemble.csv")
        
        
    else:
        #mean_accuracy
        for ratio in parse_args.train_size:
            result = list()
            for name in names:
                predicts = list()
                for seed in parse_args.seed:
                    predict = pd.read_csv(f"DataFrame/{parse_args.target}/process/{parse_args.model_name}/sample{parse_args.sample_num}/Ratio{ratio}/seed{seed}/{name}.csv", index_col=0)
                    predicts.append(np.mean(np.array(predict.loc[predict.shape[0]-1,"test_R2"]).reshape(-1, 1), axis=1))
                predicts = np.concatenate(predicts)
                result.append([name, np.mean(predicts), np.std(predicts)])
            result = pd.DataFrame(result, columns=["name", "mean_R2", "std_R2"])
            os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/", exist_ok=True)
            result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/mean.csv")
        
        #ensemble_accuracy
        for ratio in parse_args.train_size:
            result = list()
            for name in names:
                pre_te_list   = list()
                test = pd.read_csv(f"DataFrame/{parse_args.target}/test/Ratio{ratio}/{name}.csv", index_col=0)
                for seed in parse_args.seed:
                    pre_te   = pd.read_csv(f"DataFrame/{parse_args.target}/predict/{parse_args.model_name}/sample{parse_args.sample_num}/Ratio{ratio}/seed{seed}/{name}_test.csv", index_col=0)
                    pre_te_list.append(np.mean(np.array(pre_te.loc[:, ["predict_epoch200"]]), axis=1, keepdims=True))
                pre_te = np.concatenate(pre_te_list, axis=1)
                result.append([name] + list(r2_rmse_mae(np.mean(pre_te, axis=1), (test["yield"]).values.reshape(-1, 1))))
                
            result = pd.DataFrame(result, columns=["name", "R2", "RMSE", "MAE"])
            os.makedirs(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/", exist_ok=True)
            result.to_csv(f"DataFrame/{parse_args.target}/evaluation/{parse_args.model_name}/sample{parse_args.sample_num}/random/Ratio{ratio}/ensemble.csv")
        
        