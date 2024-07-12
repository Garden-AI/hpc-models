from __future__ import print_function, division
import abc, sys
from torch_geometric.data import Data, Dataset
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
import csv
import functools
import json
import os, io
import random
import warnings
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter
from pymatgen.core.lattice import Lattice
from pymatgen.core import Element, Composition
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!
import torch.distributed as dist 
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, WandbLogger
from torch.utils.data import DistributedSampler
from typing import *
from crystals.cgcnn_data_utils import * #get_dataloader func, _get_split_sizes etc.
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import multiprocessing as mp
import torch.distributed as dist
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.symmetry.groups import SpaceGroup
from p_tqdm import p_umap
import wandb
from pymatgen.optimization.neighbors import find_points_in_spheres
from crystals.dataloader import *
from itertools import combinations, product
import pandas as pd
from crystals import atomic_properties
from crystals.atomic_properties import raw_features, en_pauling
from pymatgen.analysis.local_env import IsayevNN
from torch_geometric.data import Batch
# from poremake.ruijie-code.2_parse_SBU_from_existing_MOF import 
from pathlib import Path
import ase
import ase.io
import ase.neighborlist
try:
    from ase.utils import natural_cutoffs
except Exception as e:
    from ase.neighborlist import natural_cutoffs
import pymatgen.core as mg
 
LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']
short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
# sp2id = {'Fm-3m':0,'F-43m':1,'Pm-3m':2} #WIP: USE PGCGM extension!
topology = {"pcu":0} 
nodes = {"[O][Zr]123([O])[O]4[Zr]56[O]3[Zr]37([O]2[Zr]28[O]1[Zr]14([O]6[Zr]([O]53)([O]21)([O]78)([O])[O])([O])[O])([O])[O]"}
linkers = {"[O-]C(=O)c1ccc(cc1)c1cc(c2ccc(cc2)C(=O)[O-])c2c3c1ccc1c3c(cc2)c(cc1c1ccc(cc1)C(=O)[O-])c1ccc(cc1)C(=O)[O-]"}

def read_cgd(filename, node_symbol="C", edge_center_symbol="O"):
    """
    Read cgd format and return topology as ase.Atoms object.
    """
    # try:
    with open(filename, "r") as f:
        # Neglect "CRYSTAL" and "END"
        lines = f.readlines()[1:-1]
    lines = [line for line in lines if not line.startswith("#")]

    # Get topology name.
    name = lines[0].split()[1]
    # Get spacegroup.
    spacegroup = lines[1].split()[1]

    # Get cell paremeters and expand cell lengths by 10.
    print(filename)
    cellpar = np.array(lines[2].split()[1:], dtype=np.float32)

    # Parse node information.
    node_positions = []
    coordination_numbers = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "NODE":
            continue

        coordination_number = int(tokens[2])
        pos = [float(r) for r in tokens[3:]]
        node_positions.append(pos)
        coordination_numbers.append(coordination_number)

    node_positions = np.array(node_positions)
    #coordination_numbers = np.array(coordination_numbers)

    # Parse edge information.
    edge_center_positions = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE":
            continue

        pos_i = np.array([float(r) for r in tokens[1:4]])
        pos_j = np.array([float(r) for r in tokens[4:]])

        edge_center_pos = 0.5 * (pos_i+pos_j)
        edge_center_positions.append(edge_center_pos)

    # New feature. Read EDGE_CENTER.
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE_CENTER":
            continue

        edge_center_pos = np.array([float(r) for r in tokens[1:]])
        edge_center_positions.append(edge_center_pos)

    edge_center_positions = np.array(edge_center_positions)

    # Carbon for nodes, oxygen for edges.
    n_nodes = node_positions.shape[0]
    n_edges = edge_center_positions.shape[0]
    species = np.concatenate([
        np.full(shape=n_nodes, fill_value=node_symbol),
        np.full(shape=n_edges, fill_value=edge_center_symbol),
    ])

    coords = np.concatenate([node_positions, edge_center_positions], axis=0)

    # Pymatget can handle : indicator in spacegroup.
    # Mark symmetrically equivalent sites.
    node_types = [i for i, _ in enumerate(node_positions)]
    edge_types = [-(i+1) for i, _ in enumerate(edge_center_positions)]
    site_properties = {
        "type": node_types + edge_types,
        "cn": coordination_numbers + [2 for _ in edge_center_positions],
    }

    # I don't know why pymatgen can't parse this spacegroup.
    if spacegroup == "Cmca":
        spacegroup = "Cmce"

    structure = mg.Structure.from_spacegroup(
                    sg=spacegroup,
                    lattice=mg.Lattice.from_parameters(*cellpar),
                    species=species,
                    coords=coords,
                    site_properties=site_properties,
                ).get_primitive_structure()

    # Add information.
    info = {
        "spacegroup": spacegroup,
        "name": name,
        "cn": structure.site_properties["cn"],
    }

    # Cast mg.Structure to ase.Atoms
    atoms = ase.Atoms(
        symbols=[s.name for s in structure.species],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        tags=structure.site_properties["type"],
        pbc=True,
        info=info,
    )

    # Remove overlap
    I, J, D = ase.neighborlist.neighbor_list("ijd", atoms, cutoff=0.1)
    # Remove higher index.
    J = J[J > I]
    if len(J) > 0:
        # Save original size of atoms.
        n = len(atoms)
        removed_indices = set(J)

        del atoms[list(removed_indices)]

        cn = atoms.info["cn"]
        # Remove old cn info.
        cn = [cn[i] for i in range(n) if i not in removed_indices]

        atoms.info["cn"] = cn

    return atoms
        
    # except Exception as e:
    #     print(e)
    #     pass
        # return None

def atom_embedding(d_elements: Dict[str, int]):
#     d_elements #Structure.sites instance
    features = np.zeros((len(d_elements), 23))
    for k in d_elements:
        # i: index; k: Structure.site
#         e = k.specie
#         print(k)
        i = d_elements[k]
        e = Element(k)
        features[i][0] = e.Z
        features[i][1] = e.X
        features[i][2] = e.row
        features[i][3] = e.group
        features[i][4] = e.atomic_mass
        features[i][5] = float(e.atomic_radius)
        features[i][6] = e.mendeleev_no
        # features[i][7] = sum(e.atomic_orbitals.values())
        features[i][7] = float(e.average_ionic_radius)
        features[i][8] = float(e.average_cationic_radius)
        features[i][9] = float(e.average_anionic_radius)
        features[i][10] = sum(e.ionic_radii.values())
        features[i][11] = e.max_oxidation_state
        features[i][12] = e.min_oxidation_state
#         features[i][13] = np.nan_to_num(sum(e.oxidation_states)/len(e.oxidation_states), nan=0.0)
#         features[i][14] = np.nan_to_num(sum(e.common_oxidation_states)/len(e.common_oxidation_states), nan=0.0)
        features[i][13] = np.divide(sum(e.oxidation_states), len(e.oxidation_states), out=np.zeros(1), where=len(e.oxidation_states)!=0)
        features[i][14] = np.divide(sum(e.common_oxidation_states), len(e.common_oxidation_states), out=np.zeros(1), where=len(e.common_oxidation_states)!=0)
        features[i][15] = float(e.is_noble_gas)
        features[i][16] = float(e.is_transition_metal)
        features[i][17] = float(e.is_post_transition_metal)
        features[i][18] = float(e.is_metalloid)
        features[i][19] = float(e.is_alkali)
        features[i][20] = float(e.is_alkaline)
        features[i][21] = float(e.is_halogen)
        features[i][22] = float(e.molar_volume)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    return features

def extract_function(fname):
    cif_file = CifParser(fname)
    struct = cif_file.get_structures()
    P1_fname = fname.replace(".cif", "") + "_P1.cif"
    CifWriter(struct[0]).write_file(P1_fname)
    cif = CifFile.from_file(P1_fname)
    data = cif.data
    formula = list(data.keys())[0]
    block = list(data.values())[0]
    bases = block['_atom_site_type_symbol']

    occu = np.array(block['_atom_site_occupancy']).astype(float)

    xs = np.array([x.replace("(","").replace(")","") for x in block['_atom_site_fract_x']]).reshape((-1,1))
    ys = np.array([x.replace("(","").replace(")","") for x in block['_atom_site_fract_y']]).reshape((-1,1))
    zs = np.array([x.replace("(","").replace(")","") for x in block['_atom_site_fract_z']]).reshape((-1,1))

    coords = np.hstack([xs,ys,zs]).astype(float)
    lengths = np.array([block['_cell_length_a'].replace("(","").replace(")",""),
                        block['_cell_length_b'].replace("(","").replace(")",""),
                        block['_cell_length_c'].replace("(","").replace(")","")]).astype(float)
    angles = np.array([block['_cell_angle_alpha'].replace("(","").replace(")",""),
                       block['_cell_angle_beta'].replace("(","").replace(")",""),
                       block['_cell_angle_gamma'].replace("(","").replace(")","")]).astype(float)
    lattice = Lattice.from_parameters(
            lengths[0],
            lengths[1],
            lengths[2],
            angles[0],
            angles[1],
            angles[2],
        )
    matrix = lattice.matrix

    a = [
        formula,
        bases,
        coords,
        matrix,
        lengths,
        angles
    ]

    b = [
        P1_fname.replace('.cif',''),
        formula,
        ]
    return a,b

def node_embedding():
    pass

def topology_embedding(file):
    struct = Structure.from_file(file)
    angles = struct.lattice.angles
    coords = 3

def get_radial_features(cif_struct, symbols, labels):
    n_atoms = len(cif_struct)
    property_en = [en_pauling[symbol] for symbol in symbols]

    max_frac = cif_struct.lattice.get_fractional_coords([8.0, 8.0, 8.0])
    super_scale = np.ceil(max_frac).astype(int)
    list_options = list(product(*[range(2 * i + 1) for i in super_scale]))
    super_options = np.array(list_options, dtype=float) - super_scale

    frac_xyz_og = np.array([atom.frac_coords for atom in cif_struct])
    frac_xyz_diff = frac_xyz_og.repeat(n_atoms, axis=0)
    frac_xyz_diff -= np.tile(frac_xyz_og, (n_atoms, 1))

    frac2cart = cif_struct.lattice.get_cartesian_coords

    cart_headers = ["cart_x", "cart_y", "cart_z"]

    all_img_list = []
    for i_opt, option in enumerate(super_options):
        df_tmp1 = pd.DataFrame(
            {"img_num": i_opt, "atom_label": labels, "property_en": property_en}
        )
        cart_xyz_mirror = frac2cart(frac_xyz_og + option)
        df_tmp2 = pd.DataFrame(data=cart_xyz_mirror, columns=cart_headers)
        cart_xyz_opt_diff = frac2cart(frac_xyz_diff + option)
        distance_full = np.linalg.norm(cart_xyz_opt_diff, axis=1).reshape(
            n_atoms, n_atoms
        )
        df_tmp3 = pd.DataFrame(data=distance_full, columns=labels)
        all_img_list.append(pd.concat([df_tmp1, df_tmp2, df_tmp3], axis=1))
    all_img_df = pd.concat(all_img_list)

    # parameters
    r_0 = 1.0
    rad_cut = 8.0
    ang_cut = 6.0
    rdf_bin_size = 0.25
    rdf_alpha = 60.0
    acsf_rad_bin_size = 0.5
    acsf_rad_eta = 6.0
    adf_bin_size_degree = 10
    adf_beta = 60.0
    acsf_ang_n_eta = 8

    # rdf
    rdf_n_bins = int((rad_cut - r_0) / rdf_bin_size)
    rdf_edges = np.linspace(r_0 + rdf_bin_size, rad_cut, num=rdf_n_bins)

    # radial ACSF
    acsf_rad_n_bins = int((rad_cut - r_0) / acsf_rad_bin_size)
    acsf_rad_edges = np.linspace(r_0 + acsf_rad_bin_size, rad_cut, num=acsf_rad_n_bins)

    # adf
    adf_bin_size = np.deg2rad(adf_bin_size_degree)
    adf_n_bins = int(np.pi / adf_bin_size)
    adf_edges = np.linspace(adf_bin_size, np.pi, num=adf_n_bins)

    # angular ACSF
    acsf_ang_edges = np.linspace(r_0, ang_cut, num=acsf_ang_n_eta)
    acsf_ang_eta = 1 / (2 * (acsf_ang_edges**2))
    acsf_ang_lambda = [-1, 1]
    acsf_ang_n_bins = acsf_ang_n_eta * 2
    acsf_ang_lambda_bins, acsf_ang_eta_bins = np.array(
        list(product(acsf_ang_lambda, acsf_ang_eta))
    ).T

    features_list = []
    for atom_label in labels:
        # radial environment
        rad_range_idx = all_img_df[(all_img_df[atom_label]>=r_0)&(all_img_df[atom_label]<=rad_cut)].index
        rad_prop = all_img_df.loc[rad_range_idx, "property_en"].to_numpy()
        rad_sphere = all_img_df.loc[rad_range_idx, atom_label].to_numpy()

        # rdf
        rdf_diff_bins = np.tile(rad_sphere, (rdf_n_bins, 1)).T - rdf_edges
        rdf_gauss_bins = np.exp(-rdf_alpha * (rdf_diff_bins**2))
        rdf = (rdf_gauss_bins.T * rad_prop).sum(axis=1)

        # radial WAP
        rad_abs_bins = np.abs(np.ma.masked_outside(rdf_diff_bins, -rdf_bin_size, 0.0))
        wrh_top = (rad_abs_bins.T * rad_prop).sum(axis=1).data
        wrh_btm = rad_abs_bins.sum(axis=0).data
        wap_rad_harsh = np.divide(
            wrh_top, wrh_btm, out=np.zeros(rdf_n_bins), where=(wrh_btm != 0)
        )
        wrs_btm = rdf_gauss_bins.sum(axis=0)
        wap_rad_smooth = np.divide(
            rdf, wrs_btm, out=np.zeros(rdf_n_bins), where=(wrs_btm != 0)
        )

        # radial ACSF
        acsf_rad_fcut = (np.cos(rad_sphere * np.pi / rad_cut) + 1) * 0.5
        acsf_rad_diff_bins = (
            np.tile(rad_sphere, (acsf_rad_n_bins, 1)).T - acsf_rad_edges
        )
        acsf_rad_gauss_bins = np.exp(-acsf_rad_eta * (acsf_rad_diff_bins**2))
        acsf_rad = (rad_prop * acsf_rad_fcut * acsf_rad_gauss_bins.T).sum(axis=1)

        # angular environment
        ang_range = (all_img_df[atom_label] >= r_0) & (
            all_img_df[atom_label] <= ang_cut
        )
        ang_prop = all_img_df.loc[ang_range, "property_en"].to_numpy()
        ang_sphere = all_img_df.loc[ang_range, atom_label].to_numpy()

        if len(ang_sphere) > 1:
            # angle calculation
            cart_xyz = all_img_df.loc[ang_range, cart_headers].to_numpy()
            atoms_j, atoms_k = np.array(list(combinations(range(len(ang_sphere)), 2))).T
            property_jk = ang_prop[atoms_j] * ang_prop[atoms_k]
            d_ijk = np.vstack(
                [
                    ang_sphere[atoms_j],
                    ang_sphere[atoms_k],
                    np.linalg.norm(cart_xyz[atoms_j] - cart_xyz[atoms_k], axis=1),
                ]
            )
            d_ijk_sq = d_ijk**2
            cos_theta_ijk = (d_ijk_sq[0] + d_ijk_sq[1] - d_ijk_sq[2]) / (
                2 * d_ijk[0] * d_ijk[1]
            )
            cos_theta_ijk[cos_theta_ijk < -1.0] = -1.0
            cos_theta_ijk[cos_theta_ijk > 1.0] = 1.0
            theta_ijk = np.arccos(cos_theta_ijk)

            # adf
            adf_diff_bins = np.tile(theta_ijk, (adf_n_bins, 1)).T - adf_edges
            adf_gauss_bins = np.exp(-adf_beta * (adf_diff_bins**2))
            adf = (property_jk * adf_gauss_bins.T).sum(axis=1)

            # angular WAP
            ang_abs_bins = np.ma.masked_outside(adf_diff_bins, -adf_bin_size, 0.0)
            wah_top = (ang_abs_bins.T * property_jk).sum(axis=1).data
            wah_btm = ang_abs_bins.sum(axis=0).data
            wap_ang_harsh = np.divide(
                wah_top, wah_btm, out=np.zeros(adf_n_bins), where=(wah_btm != 0)
            )
            was_btm = adf_gauss_bins.sum(axis=0)
            wap_ang_smooth = np.divide(
                adf, was_btm, out=np.zeros(adf_n_bins), where=(was_btm != 0)
            )

            # angular ACSF
            acsf_ang_fcut = ((np.cos(d_ijk * np.pi / ang_cut) + 1) * 0.5).prod(axis=0)
            acsf_ang_cos_bins = (
                1
                + acsf_ang_lambda_bins * np.tile(cos_theta_ijk, (acsf_ang_n_bins, 1)).T
            )
            acsf_ang_dist_bins = np.tile(d_ijk_sq.sum(axis=0), (acsf_ang_n_bins, 1)).T
            acsf_ang_gauss_bins = np.exp(-acsf_ang_eta_bins * acsf_ang_dist_bins)
            acsf_ang_bins = (acsf_ang_cos_bins * acsf_ang_gauss_bins).T
            acsf_ang_bins *= property_jk * acsf_ang_fcut
            acsf_ang = acsf_ang_bins.sum(axis=1)
        else:
            adf, wap_ang_harsh, wap_ang_smooth = np.zeros((3, adf_n_bins))
            acsf_ang = np.zeros(acsf_ang_n_bins)

        features_list.append(
            np.hstack(
                [
                    rdf,
                    adf,
                    acsf_rad,
                    acsf_ang,
                    wap_rad_harsh,
                    wap_rad_smooth,
                    wap_ang_harsh,
                    wap_ang_smooth,
                ]
            ),
        )
    return np.vstack(features_list)
    
def generate_graph(cif_str: str):
    print(cif_str)

    try:
        cif_path = pathlib.Path(cif_str)
        assert cif_path.exists(), f"{cif_str} DOES NOT EXIST."

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cif_parser = CifParser(cif_path)
            cif_struct = cif_parser.get_structures(primitive=False)[0]
        print("cif parser")
        coords = cif_struct.frac_coords
        atom_symbols = [atom.specie.symbol for atom in cif_struct]
        print("coord, symbols")

    #     atom_labels = []
    #     label_counter = {element: 0 for element in set(atom_symbols)}
    #     for symbol in atom_symbols:
    #         label_counter[symbol] += 1
    #         atom_labels.append(f"{symbol}{label_counter[symbol]}")
        #atom_labels = [atom_symbols[i]+str(i) for i in range(0, len(atom_symbols))]

        __cif_str = None
        with io.open(cif_path, "r", newline="\n") as __cif:
            __cif_str = __cif.read()
        for __cif_section in __cif_str.split("loop_"):
            __sect_lines = __cif_section.strip().split("\n")
            if "_atom_site_fract" in __cif_section:
                __colnames = []
                __atom_df = None
                for i in range(0, len(__sect_lines)):
                    if "_atom_" in __sect_lines[i]:
                        __colnames.append(__sect_lines[i].strip())
                    else:
                        __atom_df = pd.read_csv(io.StringIO("\n".join(__sect_lines[i:])),
                                                names=__colnames, 
                                                sep=r"\s+")
                        break
        atom_labels = __atom_df["_atom_site_label"].to_list()
    #     print(atom_labels)
        cif_dict = cif_parser.as_dict().popitem()[1]
        charge_dict = {
            label: float(charge)
            for label, charge in zip(
                atom_labels, [0]*len(cif_dict["_atom_site_label"]) #cif_dict["_atom_type_partial_charge"]
            )
        }
        print("custom cif atoms read")

        simple_features = np.array([raw_features[symbol] for symbol in atom_symbols])
        print("simple features")

        radial_features = get_radial_features(cif_struct, atom_symbols, atom_labels)
        print("radial features")

        cif_bond_info = (
            IsayevNN(tol=0.5).get_bonded_structure(structure=cif_struct).as_dict()
        )["graphs"]["adjacency"]
        print("cif_bond_info")

        bonds = np.array(
            [(i, bond["id"]) for i, bonds in enumerate(cif_bond_info) for bond in bonds]
        )
        print("bonds")

        atom_charges = np.array([charge_dict[label] for label in atom_labels])

        el2num = {key: idx for idx, key in enumerate(en_pauling.keys())}
        num2el = {idx: key for idx, key in enumerate(en_pauling.keys())}
        # print("dictionary")

    #     results = {
    # #         "node_label": np.array(atom_labels),
    #         "node_class": torch.from_numpy(np.array( list(map(lambda atsym: el2num.get(atsym), atom_symbols )) ).astype(int)).long().view(-1,),
    #         "node_target": torch.from_numpy(atom_charges).float().view(-1,),
    #         "node_simple_feature": torch.from_numpy(simple_features).float().view(-1,simple_features.shape[-1]),
    #         "node_radial_feature": torch.from_numpy(radial_features).float().view(-1,radial_features.shape[-1]),
    #         "edges": torch.from_numpy(bonds).long().view(-1,2),
    #         "lengths": torch.from_numpy(np.array(Structure.from_file(cif_path).lattice.abc)).float().view(-1,),
    #         "angles": torch.from_numpy(np.array(Structure.from_file(cif_path).lattice.angles)).float().view(-1,)
    #     }    
        results = {
            "node_class": np.array( list(map(lambda atsym: el2num.get(atsym), atom_symbols )) ).reshape(-1,),
            "node_target": atom_charges.reshape(-1,),
            "node_simple_feature": simple_features.reshape(-1, simple_features.shape[-1]),
            "node_radial_feature": radial_features.reshape(-1, radial_features.shape[-1]),
            "edges": bonds.transpose(),
            "lengths": np.array(Structure.from_file(cif_path).lattice.abc).reshape(-1,3),
            "angles": np.array(Structure.from_file(cif_path).lattice.angles).reshape(-1,3),
            "coords": coords(-1,3),
        }    
            with open("mofs_processed.pickle", "ab") as f:
                pickle.dump(results, f)
    # print("SUCCESS!")
    # print("\n")

        return results
    # return None
    except Exception as e:
        print(f"Error {e}... Passing {cif_str}...")

def parallel_process(func_name: callable=generate_graph, root_dir: str="/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax/", saved_name: str="mofs_processed.pickle"):
    import multiprocessing as mp
    ncpus = os.cpu_count()
#     root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax/"
    final_result = None
    # with mp.Pool(1) as pool:
    #     final_result = pool.map(func_name, [os.path.join(root_dir, x) for x in os.listdir(root_dir)][:10])
    final_result = list(map(func_name, [os.path.join(root_dir, x) for x in os.listdir(root_dir)]))
    with open(f"{saved_name}", "wb") as f:
        pickle.dump(final_result, f)

class TopologyData(Dataset):
    def __init__(self, root_dir, random_seed=123):
        super().__init__()
        self.root_dir = root_dir
        assert pathlib.Path(root_dir).exists(), f'{root_dir} does not exist!' # e.g. imax
#         self.files = os.listdir(self.root_dir)
        f = open(root_dir, "rb")
        self.files = pickle.load(f) #List of dicts
#         self.files = os.listdir(root_dir)
        index = list(filter(lambda inp: self.files[inp], np.arange(len(self.files)) ))
        self.atoms = [self.files[idx] for idx in index]
#         self.atoms = [read_cgd(os.path.join(root_dir,one_file)) for one_file in self.valid_files]
        
    def len(self):
        return len(self.atoms)
    
    def get(self, idx):
        one_file: ase.Atoms = self.atoms[idx] #ase.Atoms
        el2num = {key: idx for idx, key in enumerate(en_pauling.keys())}
        num2el = {idx: key for idx, key in enumerate(en_pauling.keys())}
        
        node_class = torch.from_numpy(np.array([el2num.get(at) for at in one_file.symbols])).long()

        cell = torch.from_numpy(np.array(one_file.cell)).float()
        A = cell[0]
        B = cell[1]
        C = cell[2]
        a = np.linalg.norm(A)
        b = np.linalg.norm(B)
        c = np.linalg.norm(C)
        alpha = np.arccos(B.dot(C) / b / c) / np.pi * 180.
        beta = np.arccos(A.dot(C) / a / c) / np.pi * 180.
        gamma = np.arccos(A.dot(B) / a / b) / np.pi * 180.
        lattice_info = torch.tensor([a,b,c,alpha,beta,gamma]).view(1,-1)

        tags = torch.from_numpy(np.array(one_file.__dict__["arrays"]["tags"])).long()                            
        coords = torch.from_numpy(one_file.positions).float().view(-1,3)
        coordnum = torch.from_numpy(np.array(one_file.info["cn"])).long()  
        return Data(num_nodes=len(node_class), node_class=node_class, coords=coords, cell=cell, tags=tags, coordnum=coordnum, lattice_info=lattice_info)
        
class MOFData(Dataset):
    def __init__(self, root_dir, random_seed=123):
        super().__init__()
        self.root_dir = root_dir
        assert pathlib.Path(root_dir).exists(), f'{root_dir} does not exist!' # e.g. imax
#         self.files = os.listdir(self.root_dir)
        f = open(root_dir, "rb")
        self.files = pickle.load(f) #List of dicts
        index = list(filter(lambda inp: self.files[inp], np.arange(len(self.files)) ))
        self.valid_files = [self.files[idx] for idx in index]
        
    def len(self):
        return len(self.valid_files)
    
    def get(self, idx):
#         print(idx)
        one_file = self.valid_files[idx] #dict
#         file = os.path.join(self.root_dir, file)
        node_class = torch.from_numpy(one_file.get("node_class")).long()
        node_target = torch.from_numpy(one_file.get("node_target")).float()
        node_simple_feature = torch.from_numpy(one_file.get("node_simple_feature")).float()
        node_radial_feature = torch.from_numpy(one_file.get("node_radial_feature")).float()
        lengths = torch.from_numpy(one_file.get("lengths")).float().view(-1,3)
        angles = torch.from_numpy(one_file.get("angles")).float().view(-1,3)
        coords = torch.from_numpy(one_file.get("coords")).float().view(-1,3)
        edges = torch.from_numpy(one_file.get("edges")).long().t()
#         node_class = one_file.get("node_class")
#         node_target = one_file.get("node_target")
#         node_simple_feature = one_file.get("node_simple_feature")
#         node_radial_feature = one_file.get("node_radial_feature")
#         lengths = one_file.get("lengths")
#         angles = one_file.get("angles")
#         coords = one_file.get("coords")
#         edges = one_file.get("edges")
#         print(node_class.shape, node_target.shape, node_simple_feature.shape, node_radial_feature.shape, lengths.shape, angles.shape, coords.shape, edges.shape)
        
        return Data(num_nodes=len(coords), node_class=node_class, node_target=node_target, node_simple_feature=node_simple_feature,
                    node_radial_feature=node_radial_feature, lengths=lengths, angles=angles, coords=coords, edge_index=edges)
        
class DataModuleCrystal(abc.ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        self.opt = opt = dataloader_kwargs.pop("opt")

        if get_local_rank() == 0:
            self.prepare_data()
            print(f"{get_local_rank()}-th core is parsed!")
#             self.prepare_data(opt=self.opt, data=self.data, mode=self.mode) #torch.utils.data.Dataset; useful when DOWNLOADING!

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()]) #WAITNG for 0-th core is done!
        
        root_dir = self.opt.data_dir_crystal
        if self.opt.dataset in ["cifdata"]:
            full_dataset = CIFData(root_dir)
        elif self.opt.dataset in ["gandata"]:
            # full_dataset = MOFData(root_dir)
            full_dataset = TopologyData(root_dir)
#             full_dataset = Batch.from_data_list([full_dataset[idx] for idx in index])
        elif self.opt.dataset in ["cdvaedata"]:
            full_dataset = CDVAEData(root_dir)

        self.dataloader_kwargs = {'pin_memory': opt.pin_memory, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                 'batch_size': opt.batch_size} if not self.opt.dataset in ["gandata"] else {'pin_memory': opt.pin_memory, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                 'batch_size': opt.sample_size}
        self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(full_dataset, _get_split_sizes(self.opt.train_frac, full_dataset),
                                                                generator=torch.Generator().manual_seed(0))
        
    def prepare_data(self, ):
        """ Method called only once per node. Put here any downloading or preprocessing """
        root_dir = self.opt.data_dir_crystal
        if self.opt.dataset in ["cifdata"]:
            full_dataset = CIFData(root_dir)
        elif self.opt.dataset in ["gandata"]:
            full_dataset = MOFData(root_dir)
        elif self.opt.dataset in ["cdvaedata"]:
            full_dataset = CDVAEData(root_dir)
            
    def train_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_train, shuffle=True, collate_fn=None, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_val, shuffle=False, collate_fn=None, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.opt.data_norm:
            print(cf.on_red("Not applicable for crystal..."))
        return get_dataloader(self.ds_test, shuffle=False, collate_fn=None, **self.dataloader_kwargs)    

if __name__ == "__main__":
    root_dir = "/Scr/hyunpark/ArgonneGNN/cubicgan_modified/data/trn-cifs"
#     dataset = GANData(root_dir, make_data=True)
# #     print(dataset[3])
#     print(dataset[30].aux_data, dataset[3].arr_coords)

#     dataset = CDVAEData(root_dir)
#     print(dataset[3])

#     dataset = MOFData(root_dir, make_data=False)
#     a,b=extract_function("NU-1000.cif")
#     print(a,b)

#     result = generate_graph("original_mof.cif")
#     print(result)
#     parallel_process()

#     root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax/"
#     root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/mofs_processed.pickle"
#     dataset = MOFData(root_dir)
#     dataloader = DataLoader(dataset, batch_size=4)
#     print(iter(dataloader).next().lengths)
                                      
#     datamodule = DataModuleCrystal(opt=FLAGS) #   
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()
#     test_loader = datamodule.test_dataloader()

    # root_dir = "/lus/grand/projects/ACO2RDS/hyunpark/argonne_gnn/ruijie-code/2_parse_SBU_from_existing_MOF/PORMAKE/pormake/database/topologies/"
    # root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/ruijie-code/2_parse_SBU_from_existing_MOF/PORMAKE/pormake/database/topologies/"
    # root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn_gitlab/topos_processed.pickle"
    # root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/mofs_processed.pickle"
    root_dir = "/Scr/hyunpark/ArgonneGNN/DATA/diverse_metals/cifs/"
    # root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax/"
    parallel_process(func_name=generate_graph, root_dir=root_dir, saved_name="mofs_processed.pickle")
    # dataset = MOFData(root_dir)
    # dataloader = DataLoader(dataset, batch_size=4)
    # batch = iter(dataloader).next()
    # print(batch)
