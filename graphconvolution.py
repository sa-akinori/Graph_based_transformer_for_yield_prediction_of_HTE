# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from openeye import oechem
from typing import List, Tuple
from functions import device, SmilesToOEGraphMol

ATOM_FDIM, BOND_FDIM = 400, 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def mol2vec_genenrate(smiles:str,
                    atom_idx:int, 
                    mol2vec_pd):
    mol2vec = mol2vec_pd.query("input_washed_smiles == @smiles").reset_index(drop=True)
    return np.hstack(mol2vec.loc[0, atom_idx]).tolist()

def random_embedding(
    atomic_num:int,
    random_embed
    ):
    return random_embed[atomic_num]

def bond_generate(bond):
    if bond.IsAromatic():
        bt = 0
    else:
        bt = bond.GetOrder()
    fbond = onek_encoding_unk(bt, [0, 1, 2, 3, 4])
    fbond = fbond + [bond.IsInRing()]
    return fbond

def index_select_ND(source, index):

    source = source.float()
    index = index.long()
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source[index.view(-1), :]
    target = target.view(final_size)

    return target

class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2a: A mapping from an atom index to the index of the neighbor atoms.
    - a2b: A mapping from an atom index to a list of bond indices.
    """

    def __init__(self, 
                smiles:str, 
                mol2vec:pd.DataFrame):

        if smiles == "None":
            self.n_atoms, self.n_bonds = 0, 0
            self.f_atoms, self.f_bonds = [], []
            self.a2a, self.a2b = [], []

        else:
            smiles = oechem.OECreateCanSmiString(SmilesToOEGraphMol(smiles, True))
            mol = SmilesToOEGraphMol(smiles, True)
            self.n_atoms, self.n_bonds = mol.NumAtoms(), mol.NumBonds()
            self.f_atoms, self.f_bonds = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_bonds)]
            self.a2a, self.a2b = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_atoms)]
            
            mid_idx = 0
            sep_smiles = smiles.split(".")
            for s_smile in sep_smiles:
                for s_atom in SmilesToOEGraphMol(s_smile).GetAtoms():
                    idx = s_atom.GetIdx()
                    f_atoms = mol2vec_genenrate(s_smile, idx, mol2vec)
                    self.f_atoms[idx + mid_idx] = f_atoms

                mid_idx += idx+1
                
            for atom1 in mol.GetAtoms():
                
                neigh = [a for a in atom1.GetAtoms()]
                self.a2a[atom1.GetIdx()] = [a.GetIdx() for a in neigh]
                
                bonds = [mol.GetBond(atom1, atom2) for atom2 in neigh]
                bond_idx = [bond.GetIdx() for bond in bonds]
                self.a2b[atom1.GetIdx()] = bond_idx

                for b_id, bond in zip(bond_idx, bonds):
                    
                    f_bond = bond_generate(bond)
                    self.f_bonds[b_id] = f_bond
                    

class RandomMolGraph:

    def __init__(self, 
                smiles:str, 
                random_embed):

        if smiles == "None":
            self.n_atoms, self.n_bonds = 0, 0
            self.f_atoms, self.f_bonds = [], []
            self.a2a, self.a2b = [], []

        else:
            smiles = oechem.OECreateCanSmiString(SmilesToOEGraphMol(smiles, True))
            mol = SmilesToOEGraphMol(smiles, True)
            self.n_atoms, self.n_bonds = mol.NumAtoms(), mol.NumBonds()
            self.f_atoms, self.f_bonds = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_bonds)]
            self.a2a, self.a2b = [[] for _ in range(self.n_atoms)], [[] for _ in range(self.n_atoms)]
            
            mid_idx = 0
            sep_smiles = smiles.split(".")
            for s_smile in sep_smiles:
                for s_atom in SmilesToOEGraphMol(s_smile).GetAtoms():
                    idx = s_atom.GetIdx()
                    f_atoms = random_embedding(s_atom.GetAtomicNum(), random_embed)
                    self.f_atoms[idx + mid_idx] = f_atoms

                mid_idx += idx+1
                
            for atom1 in mol.GetAtoms():
                
                neigh = [a for a in atom1.GetAtoms()]
                self.a2a[atom1.GetIdx()] = [a.GetIdx() for a in neigh]
                
                bonds = [mol.GetBond(atom1, atom2) for atom2 in neigh]
                bond_idx = [bond.GetIdx() for bond in bonds]
                self.a2b[atom1.GetIdx()] = bond_idx

                for b_id, bond in zip(bond_idx, bonds):
                    
                    f_bond = bond_generate(bond)
                    self.f_bonds[b_id] = f_bond
                    

class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs):

        self.n_atoms = 1
        self.n_bonds = 1
        f_atoms = [[0] * ATOM_FDIM]
        f_bonds = [[0] * BOND_FDIM]
        self.a_scope = []

        a2a, a2b = [[]], [[]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            a2a.extend([[b2 + self.n_atoms for b2 in b1] for b1 in mol_graph.a2a])
            a2b.extend([[b2 + self.n_bonds for b2 in b1] for b1 in mol_graph.a2b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        max_num_bonds = max(len(in_bonds) for in_bonds in a2b)
        max_num_atoms = max(len(nei_atoms) for nei_atoms in a2a)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.a2a = torch.LongTensor([a2a[a] + [0] * (max_num_atoms - len(a2a[a])) for a in range(self.n_atoms)])

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                    torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                    List[Tuple[int, int]], List[Tuple[int, int]]]:

        return self.f_atoms, self.f_bonds, self.a2a, self.a2b, self.a_scope


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: dict, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.
        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        
        super(MPNEncoder,self).__init__()

        self.args   = args
        self.device = device()
        self.zeros  = nn.Parameter(torch.zeros(args["hidden_size"]), requires_grad=False)
        self.W_i    = nn.Linear(atom_fdim, args["hidden_size"], bias=args["bias"]).to(self.device)
        self.W_o    = nn.Sequential(*[nn.Linear(args["hidden_size"]*2, args["hidden_size"], bias=args["bias"]), nn.GELU()]).to(self.device)
        self.gelu   = nn.GELU().to(self.device)
        self.W_ah   = self._make_module_list(atom_fdim, args["depth"])
        
        for i in range(args["depth"]):
            exec(f"self.W_h{i} = self._make_module_list(self.args['hidden_size'] + bond_fdim, self.args['agn_num'])")
            
            
    def _make_module_list(self, input_size, iteration):
        
        modulelist = [nn.Linear(input_size, self.args["hidden_size"], bias=self.args["bias"])]
            
        for _ in range(iteration-1):
            
            modulelist.extend([nn.GELU(), nn.Dropout(p=self.args["dropout_ratio"]), nn.Linear(self.args["hidden_size"], self.args["hidden_size"], bias=self.args["bias"])])
            
        return nn.Sequential(*modulelist).to(self.device)

    def forward(self,
                mol_graph: BatchMolGraph
                ) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.
        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        f_atoms, f_bonds, a2a, a2b, a_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, a2a  = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device), a2a.to(self.device)
        
        input = self.W_i(f_atoms)
        self_message, message = input, input
        self_message[0, :], message[0, :] = 0, 0
        
        for depth in range(self.args["depth"]):
        
            nei_a_message = index_select_ND(message, a2a)
            nei_f_bonds   = index_select_ND(f_bonds, a2b)
            n_message = torch.cat([nei_a_message, nei_f_bonds], dim=2)
            message   = n_message.sum(dim=1)
            message = eval(f"self.W_h{depth}(message)")
            self_message = self_message + message
            message = self_message
            message[0 , :] = 0
            
        nei_a_message = index_select_ND(message, a2a)
        a_message  = self.gelu(nei_a_message.sum(dim=1))
        cc_message = self.W_ah(f_atoms)
        a_input = torch.cat([cc_message, a_message], dim=1)
        atom_hiddens = self.W_o(a_input)
            
        return atom_hiddens, a_scope

class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                args: dict
                ):

        super(MPN, self).__init__()
        self.args    = args
        self.encoder = MPNEncoder(self.args, ATOM_FDIM, BOND_FDIM)
        
    def forward(self,
                batch) -> torch.FloatTensor:
            
        batch  = BatchMolGraph(batch)
        output = self.encoder.forward(batch)

        return output

