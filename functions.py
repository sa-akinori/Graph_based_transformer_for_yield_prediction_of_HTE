import random
import torch
import numpy as np
import pandas as pd 
from openeye.oechem import *

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

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    
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