# -*- coding: utf-8 -*-
import os
import random
import torch
import numpy as np
import pandas as pd
import pickle
from openeye.oechem import *

BASEPATH = os.path.dirname(__file__)

def reshape(obj, size):
    """
    Wrapper function of reshape compatible with both pandas dataframe and numpy array
    """
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        n_obj = obj.values.reshape(size)
    else:
        n_obj = obj.reshape(size)
    
    return n_obj

def r2_rmse_mae(yp, yobs, verbose=False):
    """
    R2, RMSE, MAE calculation
    :param yp: predicted y values (array)
    :param yobs: observed y values (array)
    :return: R2, RMSE, MAE
    """
    yp   = reshape(yp, (-1,1))
    yobs = reshape(yobs, (-1,1))
    
    R2 = 1 - np.sum((yp - yobs)**2)/(np.var(yobs)*(yobs.shape[0]))
    RMSE = np.sqrt(np.mean( (yp - yobs)**2))
    MAE = np.mean(np.absolute(yp-yobs))

    if verbose:
        print("R2: %f \n RMSE: %f \n MAE %f \n" %(R2, RMSE, MAE))

    return R2, RMSE, MAE

def random_seed(
    seed:int
    ):
    """
    To ensure reproducibility, I introduced 'torch.use_deterministic_algorithms'.
    The system requires os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' when we introduce 'torch.use_deterministic_algorithms'.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def tensor_to_numpy(
    x:torch.Tensor
    ):
    """
    Change tensor type to numpy type
    """
    return x.to("cpu").detach().numpy().copy()

def make_batch(
    batch_size:int, 
    data:pd.DataFrame, 
    seed:int):
    
    ids = [id for id in data["reaction_id"]]
    batch_data = list()
    random.seed(seed)
    random.shuffle(ids)

    for num in range(0, len(ids), batch_size):

        batch_ids = ids[num : num + batch_size]
        batch_data.append(data.query("reaction_id in @batch_ids"))

    return batch_data

def device():

    if torch.cuda.is_available():
        return "cuda"

    else:
        return "cpu"
    
def SmilesToOEGraphMol(smiles, strict=False):
    """
    wrapper function for making a new molecule from similes
    Add None if the molecule is not correct..
    """
    if smiles is None:
        return None

    mol = OEGraphMol()
    if strict:
        mol2 = OEGraphMol()
        if not OEParseSmiles(mol2, smiles, False, True):
            return None

    output = OESmilesToMol(mol, smiles)
    if not output:
        return None
    else:
        return mol
    
def pickle_load(
    path : str
    ):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return data
    
def pickle_save(
    path : str, 
    data,
    protocol=5
):
    with open(f"{path}.pkl", 'wb') as f:
        pickle.dump(data, f, protocol=protocol)
        
def necessary_member(args):
    
    if args.target == "buchwald-hartwig":
        columns = ["aniline_smiles", "additive_smiles", "aryl_halide_smiles", "ligand_smiles", "base_smiles", "product_smiles"]
        
        if eval(args.extrapolation):
            
            if args.extrapolation_role == "additive":
                names = [f"Test{num}" for num in range(1, 5)]
            
            elif args.extrapolation_role == "three-roles":
                names = [f"sTest{num}" for num in range(1, 21)]
            
        else:
            names = ["FullCV_01", "FullCV_02", "FullCV_03", "FullCV_04", "FullCV_05", "FullCV_06", "FullCV_07", "FullCV_08", "FullCV_09", "FullCV_10"]
        
    elif args.target == "suzuki-miyaura":
        columns = ["Organoboron_SMILES", "Organic_Halide_SMILES", "Solvent_SMILES", "Reagent_SMILES", "Ligand_SMILES", "Product_SMILES"]
        
        if eval(args.extrapolation):
            names = [f"Test{num}" for num in range(1, 13)]
            
        else:
            names = ["FullCV_01", "FullCV_02", "FullCV_03", "FullCV_04", "FullCV_05", "FullCV_06", "FullCV_07", "FullCV_08", "FullCV_09", "FullCV_10"]
    else:
        raise ValueError(f"{args.target} is invalid")
    
    columns = [f"washed_{col}" for col in columns]
    return columns, names