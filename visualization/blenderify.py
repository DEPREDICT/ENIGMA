import pickle
import bpy
import nibabel.freesurfer.io as fsio
import os
import numpy as np
from collections.abc import Iterable
from random import random

def clamp(n, minn=0, maxn=1):
    if isinstance(n, Iterable):
        return [clamp(m, minn, maxn) for m in n]
    else:
        return max(min(maxn, n), minn)

def pickle_in(source: str) -> dict:
    with open(source, 'rb') as file:
        return pickle.load(file)


# parameters
hemispheres = ['lh.', 'rh.']
model_names = ['inflated', 'pial', 'white', 'orig', 'sphere']
scalar_mappings = ['thickness', 'curv', 'aparc.a2009s.annot',
                   'jacobian_white', 'sulc', 'area', 'volume',
                   'annot', 'annot_by_lut']
# Annot by lit means coefficients mappings

# subject = r'D:\repositories\ENIGMA\data\pharmo\440_TP1'
subject = r'D:\repositories\ENIGMA\data\AFFDIS\Surf\K002'
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\LH-LRC_surf-coefs-Extremes.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\LH-LRC_thick-coefs-Extremes.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\RH-LRC_surf-coefs-Extremes.pkl"
coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\LH-LRC_surf-coefs-Extremes.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\LH-LRC_surf-coefs-All.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\LH-LRC_thick-coefs-All.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\RH-LRC_surf-coefs-All.pkl"
#coef_lut = r"D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage\manual_labels\RH-LRC_thick-coefs-All.pkl"
fsaverage = r'D:\repositories\ENIGMA\scripts\s1_preprocessing\fsaverage'

### P A R A M E T E R S   A R E   S E T   H E R E ###
hm = hemispheres[0]
s_map = scalar_mappings[-1]
model = model_names[1]
#####################################################




bpy.ops.object.add(type='MESH', enter_editmode=False, location=(0, 0, 0))

o = bpy.context.object
if s_map == 'annot_by_lut':
    o.name = os.path.basename(coef_lut.split(os.extsep)[0])
else:
    o.name = hm + s_map

# Read in data
if s_map == 'annot' or s_map == 'annot_by_lut':
    scalars, c_info, names = fsio.read_annot(os.path.join(fsaverage, 'label', hm + 'aparc.annot'))
    coordinates, faces = fsio.read_geometry(os.path.join(fsaverage, 'surf', hm + model))
    if s_map == 'annot_by_lut':
        lut = pickle_in(coef_lut)
else:
    scalars = fsio.read_morph_data(os.path.join(subject, 'surf', hm + s_map))
    coordinates, faces = fsio.read_geometry(os.path.join(subject, 'surf', hm + model))

lut = {k: v + 0.2 * (random()-0.5) for k, v in lut.items()}

c = [tuple(i) for i in coordinates]
f = [tuple(i) for i in faces]
o.data.from_pydata(c, [], f)
o.rotation_euler[-1] = 3.14
o.data.update()

# Color vertices
mesh = o.data
mesh.vertex_colors.new()
color_layer = mesh.vertex_colors["Col"]
bpy.ops.object.shade_smooth()

if not s_map == 'annot':
    scalars -= scalars.min()
    scalars /= scalars.max()

i = 0

colors = []
if s_map == 'annot_by_lut':
    scaler = 1
else:
    scaler = None

for face, poly in zip(faces, mesh.polygons):
    if s_map == 'annot':
        color = np.append(c_info[s][:3] / 255, 1.0)
        # color = np.array([np.append(c_info[scalars[a]][:3] / 255, 1.0),
        #                   np.append(c_info[scalars[b]][:3] / 255, 1.0),
        #                   np.append(c_info[scalars[c]][:3] / 255, 1.0)]).mean(axis=0)
    else:
        if s_map == 'annot_by_lut':
            s = scalars[face[0]]
            if s in lut:
                c = lut[s]
            else:
                color = (0., 0., 0., 1.)
        else:
            if scaler is None:
                all_cols = [np.mean([scalars[f] for f in face]) for face in faces]
                minner = np.min(all_cols)
                #scaler = np.std(all_cols) * 4 - minner  # used for Area
                scaler = np.std(all_cols) * 8 - minner  # used for Thickness
            # s = (scalars[a] + scalars[b] + scalars[c]) / 3
            c = np.mean([scalars[f] for f in face])
            c -= minner
            c /= scaler
            c -= 0.5 # used for both Area and Thickness
            c *= 2 # used for Area
            # c *= 4  # used for Thickness
            c += 0.5  # used for both Area and Thickness
        # same code as in utils.get_rgb_cbar
        r = clamp(2 * c - 1, 0, 1)
        b = clamp(1 - 2 * c, 0, 1)
        g = clamp(0.7 - (r + b), 0, 1)
        r += g
        b += g
        color = (r, g, b, 1.)
    for idx in poly.loop_indices:
        color_layer.data[i].color = color
        i += 1


# set to vertex paint mode to see the result
#bpy.ops.object.mode_set(mode='VERTEX_PAINT')

bpy.ops.object.material_slot_add()
new_mat = bpy.data.materials.new(s_map)
new_mat.use_nodes = True
bpy.context.object.active_material = new_mat

principled_node = new_mat.node_tree.nodes.get('Principled BSDF')
principled_node.inputs[0].default_value = (1,0,0,1)

color_node = new_mat.node_tree.nodes.new('ShaderNodeVertexColor')
rgb_node = new_mat.node_tree.nodes.new('ShaderNodeRGBCurve')

# link
new_mat.node_tree.links.new(color_node.outputs["Color"], rgb_node.inputs["Color"])
new_mat.node_tree.links.new(rgb_node.outputs["Color"], principled_node.inputs["Base Color"])


# new_mat.node_tree.nodes.new("Color Attribute")
# bpy.data.materials.new(s_map)
# context.object.active_material = new_mat
#
# #bpy.ops.node.add_node(type="ShaderNodeVertexColor", use_transform=True)
#
# bpy.context.scene.node_tree
#
#

# Set to rendered
bpy.context.scene.render.engine = 'CYCLES'
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = 'RENDERED'

######


