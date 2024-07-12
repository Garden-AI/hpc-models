import os
import re
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from pymatgen.io.cif import CifFile,CifParser
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


######extracy_crystals.py
LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']

root = 'data/trn-cifs/'
valid_files = os.listdir(root)


def function(file):
    # file = str(file)+'.cif'
    cif = CifFile.from_file(root+file)
    data = cif.data
    formula = list(data.keys())[0]
    block = list(data.values())[0]
    bases = block['_atom_site_type_symbol']
    #remove materials with La AND Ac rows
    #remove materials having more than three base atoms 
    occu = np.array(block['_atom_site_occupancy']).astype(float)

    if len(bases)==3 and len(set(bases))==3 and all(occu == 1.0):
        # print(file,bases)

        xs = np.array(block['_atom_site_fract_x']).reshape((3,1))
        ys = np.array(block['_atom_site_fract_y']).reshape((3,1))
        zs = np.array(block['_atom_site_fract_z']).reshape((3,1))
        coords = np.hstack([xs,ys,zs]).astype(float)
        lengths = np.array([block['_cell_length_a'],block['_cell_length_b'],block['_cell_length_c']]).astype(float)
        angles = np.array([block['_cell_angle_alpha'],block['_cell_angle_beta'],block['_cell_angle_gamma']]).astype(float)
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
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            bases,
            coords,
            matrix,
            lengths,
            angles
        ]

        b = [
            file.replace('.cif',''),
            formula,
            block['_symmetry_Int_Tables_number'],
            block['_symmetry_space_group_name_H-M'],
            ]
        return a,b


pool = mp.Pool(processes=24)
results = [pool.apply_async(function, args=(file,)) for file in valid_files]

d = {}
ternary_uniq_sites = []

for p in results:
    if p.get() is not None:
        a,b = p.get()
        d[b[0]] = a
        ternary_uniq_sites.append(b)

print(len(ternary_uniq_sites))

with open('data/ternary-dataset-pool.pkl','wb') as f:
    pickle.dump(d, f, protocol=4)


df = pd.DataFrame(np.array(ternary_uniq_sites), columns=['id','formula','spid','spsym'])
df.to_csv('data/ternary-lable-records.csv', index=False)

######utils.py
import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import Counter
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from sklearn.preprocessing import MinMaxScaler

def atom_embedding(d_elements):
    features = np.zeros((len(d_elements), 23))
    for k in d_elements:
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
        features[i][13] = sum(e.oxidation_states)/len(e.oxidation_states)
        features[i][14] = sum(e.common_oxidation_states)/len(e.common_oxidation_states)
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

def load_cubic():
    #need to change the ratio according to different dataset
    cubics_ratio = {'Fm-3m': 186344, 'F-43m': 184162,'Pm-3m':5243}
    sp2id = {'Fm-3m':0,'F-43m':1,'Pm-3m':2}
   
    prob_sp = {k:cubics_ratio[k]/sum(cubics_ratio.values()) for k in cubics_ratio}
    prob = np.zeros(len(prob_sp))
    for k in prob_sp:
        prob[sp2id[k]] = prob_sp[k]

    with open('data/ternary-dataset-pool.pkl','rb') as f:
        d = pickle.load(f)
    df = pd.read_csv('data/ternary-lable-records.csv')
    values = df.values
    ids,formulas = [],[]
    for row in values:
        ix,comp,_,symbol = row
        if symbol in cubics_ratio:
            ids.append(ix)
            formulas.append(comp)
    ids = np.array(ids).astype(str)
    np.random.shuffle(ids)
    elements = []
    for f in formulas:
        elements += list(Composition(f).as_dict().keys())
    elements = list(set(elements))
    elements.sort()
    d_elements = {}
    for i,e in enumerate(elements):
        d_elements[e]=i

    with open('data/cubic-elements-dict.json', 'w') as f:
        json.dump(d_elements, f, indent=2)
    embedding = atom_embedding(d_elements)
    np.save('data/cubic-elements-features',embedding)

    arr_sp = []
    arr_element = []
    arr_coords = []
    arr_lengths = []
    arr_angles = []
    for idx in ids:
        _,_,sp,e,coords,_,abc,angles=d[idx]
        tmp = np.rint(np.array(coords)/0.125)
        h = np.rint(np.array(angles)/30.0)
        if not np.any(np.isin(tmp, [1.0, 3.0, 5.0, 7.0])):
            arr_sp.append(sp2id[sp])
            arr_element.append([d_elements[key] for key in e])
            arr_coords.append(coords)
            arr_lengths.append(abc[0])
            arr_angles.append(angles)


    arr_sp = np.array(arr_sp).astype(int)
    arr_element = np.stack(arr_element, axis=0).astype(int)

    arr_coords = np.stack(arr_coords, axis=0).astype(float)
    m_coord_scales = np.amax(arr_coords, axis=0)/2.0

    arr_lengths = np.stack(arr_lengths, axis=0)#.reshape(len(ids),9).astype(float)
    maximum_lengths = np.amax(arr_lengths, axis=0)/2.0
    arr_angles = np.stack(arr_angles, axis=0)
    maximum_angles = np.amax(arr_angles, axis=0)/2.0

    print('for reverse\n', m_coord_scales)
    print('for reverse', maximum_lengths)
    print('for reverse', maximum_angles)    
    arr_coords = (arr_coords-m_coord_scales)/m_coord_scales
    arr_lengths = (arr_lengths-maximum_lengths)/maximum_lengths
    arr_angles = (arr_angles-maximum_angles)/maximum_angles
    print(arr_sp.shape)
    print(arr_element.shape)
    print(arr_coords.shape)
    print(arr_lengths.shape)
    print(arr_angles.shape)

    
    return (len(d_elements),len(sp2id),maximum_lengths,\
        maximum_angles,m_coord_scales,sp2id,prob),\
            (arr_sp,arr_element,arr_coords,arr_lengths,arr_angles)
    
if __name__ == '__main__':
    load_cubic()


####generate_crystal.py
import os
import json
import math
import numpy as np
# np.random.seed(123)
import pandas as pd
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from util import load_cubic
import warnings
warnings.filterwarnings("ignore")
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.core.lattice import Lattice
import multiprocessing as mp

LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']

short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']

def generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data):
    z = np.random.normal(0,1.0,(n_samples, lat_dim))

    p = aux_data[-1]

    label_sp = np.random.choice(aux_data[1],n_samples,p=p)

    with open('data/cubic-elements-dict.json', 'r') as f:
        e_d = json.load(f)

        exclude_ids = [e_d[e] for e in short_LaAc if e in e_d]
        other_ids = []
        for k in e_d:
            if e_d[k] not in exclude_ids:
                other_ids.append(e_d[k])

    label_elements = []
    for i in range(n_samples):
        fff = np.random.choice(other_ids,3,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements)

    # ix = np.random.choice(len(candidate_element_comb),n_samples)
    # label_elements = candidate_element_comb[ix]

    return [label_sp,label_elements,z]


def generate_crystal_cif(generator,lat_dim,n_samples,\
    candidate_element_comb,aux_data):
    gen_inputs = generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data)
    spacegroup,formulas = gen_inputs[0],gen_inputs[1]

    sp_d = aux_data[-2]
    rsp = {sp_d[k]:k for k in sp_d}
    spacegroup = [rsp[ix] for ix in spacegroup]

    with open('data/cubic-elements-dict.json', 'r') as f:
       e_d = json.load(f)
       re = {e_d[k]:k for k in e_d}
    arr_comb = []
    for i in range(n_samples):
        arr_comb.append([re[e] for e in formulas[i]])

    coords,arr_lengths = generator.predict(gen_inputs,batch_size=1024)
    coords = coords*aux_data[4]+aux_data[4]
    coords = np.rint(coords/0.25)*0.25
    # exit(arr_lengths)
    # arr_angles = arr_angles*aux_data[3]+aux_data[3]
    arr_lengths = arr_lengths*aux_data[2]+aux_data[2]

    if os.path.exists('generated_mat/sample-%d/'%int(FLAGS.n_samples)):
        os.system('rm -rf generated_mat/sample-%d/'%int(FLAGS.n_samples))
    os.system('mkdir generated_mat/sample-%d/'%int(FLAGS.n_samples))

    if os.path.exists('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples)):
        os.system('rm -rf generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))
    os.system('mkdir generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))
    for i in range(n_samples):
        f = open('data/cif-template.txt', 'r')
        template = f.read()
        f.close()

        lengths = arr_lengths[i][0]
        lengths = [lengths]*3

        angles = [90.0,90.0,90.0]

        template = template.replace('SYMMETRY-SG', spacegroup[i])
        template = template.replace('LAL', str(lengths[0]))
        template = template.replace('LBL', str(lengths[1]))
        template = template.replace('LCL', str(lengths[2]))
        template = template.replace('DEGREE1', str(angles[0]))
        template = template.replace('DEGREE2', str(angles[1]))
        template = template.replace('DEGREE3', str(angles[2]))
        f = open('data/symmetry-equiv/%s.txt'%spacegroup[i].replace('/','#'), 'r')
        sym_ops = f.read()
        f.close()

        template = template.replace('TRANSFORMATION\n', sym_ops)

        for j in range(3):
            row = ['',arr_comb[i][j],arr_comb[i][j]+str(j),\
                str(coords[i][j][0]),str(coords[i][j][1]),str(coords[i][j][2]),'1']
            row = '  '.join(row)+'\n'
            template+=row

        template += '\n'
        f = open('generated_mat/sample-%d/generated-cifs/%s---%d.cif'%(int(FLAGS.n_samples),spacegroup[i].replace('/','#'),i),'w')
        f.write(template)
        f.close()

if __name__ == '__main__':


    flags = tf.compat.v1.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('num_epochs', 50, "the number of epochs for training")
    flags.DEFINE_integer('batch_size', 32, "batch size")
    flags.DEFINE_integer('lat_dim', 128, "latent noise size")
    flags.DEFINE_integer('device', 0, "GPU device")
    flags.DEFINE_integer('d_repeat', 5, "GPU device")
    flags.DEFINE_integer('n_samples', 5000, "# samples to generate")
    os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.device)

    #load dataset and auxinary info
    AUX_DATA, DATA = load_cubic()
    candidate_element_comb = DATA[1]
    g_model = load_model('models/clean-wgan-generator-%d.h5'%(FLAGS.device), compile=False)

    generate_crystal_cif(g_model,FLAGS.lat_dim,FLAGS.n_samples,candidate_element_comb,AUX_DATA)


####pymagetn_valid.py
import os
import random
import warnings
warnings.filterwarnings("ignore")
from pymatgen import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
import multiprocessing as mp

import tensorflow as tf
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_samples', 5000, "# samples to generate")

if os.path.exists('generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples)):
    os.system('rm -rf generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples))
os.system('mkdir generated_mat/sample-%d/tmp-symmetrized-cifs'%int(FLAGS.n_samples))

if os.path.exists('generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples)):
    os.system('rm -rf generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples))
os.system('mkdir generated_mat/sample-%d/tmp-charge-cifs'%int(FLAGS.n_samples))

gen_cifs = os.listdir('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples))


def charge_check(crystal):
    # oxidation_states
    elements = list(crystal.composition.as_dict().keys())

    oxi = {}
    for e in elements:
        oxi[e] = Element(e).oxidation_states
    res = []
    if len(oxi) == 3:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k]}
                    crystal.add_oxidation_state_by_element(d)
                    if crystal.charge==0.0:
                        crystal.remove_oxidation_states()
                        res.append(d)
                    crystal.remove_oxidation_states()

    return res

def process(cif):
    sp = cif.split('---')[0].replace('#','/')
    i = int(cif.split('---')[1].replace('.cif',''))
    try:
        crystal = Structure.from_file('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.n_samples)+cif)
        
        formula = crystal.composition.reduced_formula
        sg_info = crystal.get_space_group_info(symprec=0.1)

        if sp == sg_info[0]:
            #only valid cif
            crystal.to(fmt='cif',\
             filename='generated_mat/sample-%d/tmp-symmetrized-cifs/%d-%s-%d-%d.cif'%\
             (int(FLAGS.n_samples),len(crystal),formula,sg_info[1],i),symprec=0.1)
            #charge
            res = charge_check(crystal)
            if len(res) > 0:
                crystal.to(fmt='cif',\
                  filename='generated_mat/sample-%d/tmp-charge-cifs/%d-%s-%d-%d.cif'%\
                  (int(FLAGS.n_samples),len(crystal),formula,sg_info[1],i),symprec=0.1)


    except Exception as e:
        pass


pool = mp.Pool(processes=18)
pool.map(process, gen_cifs)
pool.close()





















