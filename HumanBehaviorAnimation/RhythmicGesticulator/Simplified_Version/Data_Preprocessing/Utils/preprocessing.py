"""
Preprocessing Tranformers Based on sci-kit's API.
Reference: https://github.com/simonalexanderson/StyleGestures
"""


import os
import sys
import copy
import pandas as pd
import numpy as np
import transforms3d as t3d
import scipy.ndimage.filters as filters

from sklearn.base import BaseEstimator, TransformerMixin

module_path = os.path.dirname(os.path.abspath(__file__))
if module_path not in sys.path:
    sys.path.append(module_path)
from rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll
from Quaternions import Quaternions
from Pivots import Pivots


class MocapParameterizer(BaseEstimator, TransformerMixin):
    def __init__(self, param_type='euler'):
        """
        param_type = {'euler', 'quat', 'expmap', 'position', 'expmap2pos'}
        """
        self.param_type = param_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("MocapParameterizer: " + self.param_type)

        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._to_expmap(X)
        elif self.param_type == 'quat':
            return X
        elif self.param_type == 'position':
            return self._to_pos(X)
        elif self.param_type == 'expmap2pos':
            return self._expmap_to_pos(X)
        else:
            raise ValueError('param types: euler, quat, expmap, position, expmap2pos')
    
    def inverse_transform(self, X, copy=None): 
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._expmap_to_euler(X)
        elif self.param_type == 'quat':
            raise ValueError('quat2euler is not supported')
        elif self.param_type == 'position':
            print('positions 2 eulers is not supported')
            return X
        else:
            raise ValueError('param types: euler, quat, expmap, position')

    def _to_pos(self, X):
        """Converts joints rotations in Euler angles to joint positions"""

        Q = []
        for track in X:
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep.
            pos_df = pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels.
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels.
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels.
            joints = (joint for joint in track.skeleton)
            
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']

                # Get the rotation columns that belong to this joint.
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint.
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order.
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values['%s_%srotation'%(joint, rot_order[0])], track.values['%s_%srotation'%(joint, rot_order[1])], track.values['%s_%srotation'%(joint, rot_order[2])]]))

                if pc.shape[1] < 3:
                    pos_values = np.asarray([[0, 0, 0] for f in pc.iterrows()])
                else:
                    pos_values = np.asarray([[f[1]['%s_Xposition'%joint],
                                  f[1]['%s_Yposition'%joint], 
                                  f[1]['%s_Zposition'%joint]] for f in pc.iterrows()])
                
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)
                
                tree_data[joint] = [
                                    [],  # to store the rotation matrix.
                                    []  # to store the calculated position.
                                 ] 
                if track.root_name == joint:
                    tree_data[joint][0] = quats  # rotmats.
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent.
                    tree_data[joint][0] = tree_data[parent][0]*quats  # np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i.
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])

                    # multiply k to the rotmat of the parent for every frame i.
                    q = tree_data[parent][0]*k  # np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i.
                    tree_data[joint][1] = tree_data[parent][1] + q  # q.reshape(k.shape[0],3) + tree_data[parent][1]

                # Create the corresponding columns in the new DataFrame.
                pos_df['%s_Xposition' % joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Yposition' % joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Zposition' % joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)


            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)

        return Q

    def _expmap2rot(self, expmap):

        theta = np.linalg.norm(expmap, axis=1, keepdims=True)
        nz = np.nonzero(theta)[0]

        expmap[nz,:] = expmap[nz,:]/theta[nz]

        nrows=expmap.shape[0]
        x = expmap[:,0]
        y = expmap[:,1]
        z = expmap[:,2]

        s = np.sin(theta*0.5).reshape(nrows)
        c = np.cos(theta*0.5).reshape(nrows)

        rotmats = np.zeros((nrows, 3, 3))

        rotmats[:,0,0] = 2*(x*x-1)*s*s+1
        rotmats[:,0,1] = 2*x*y*s*s-2*z*c*s
        rotmats[:,0,2] = 2*x*z*s*s+2*y*c*s
        rotmats[:,1,0] = 2*x*y*s*s+2*z*c*s
        rotmats[:,1,1] = 2*(y*y-1)*s*s+1
        rotmats[:,1,2] = 2*y*z*s*s-2*x*c*s
        rotmats[:,2,0] = 2*x*z*s*s-2*y*c*s
        rotmats[:,2,1] =  2*y*z*s*s+2*x*c*s
        rotmats[:,2,2] =  2*(z*z-1)*s*s+1

        return rotmats

    def _expmap_to_pos(self, X):
        '''Converts joints rotations in expmap notation to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=exp_df.index)

            # Copy the root rotations into the new DataFrame
            # rxp = '%s_Xrotation'%track.root_name
            # ryp = '%s_Yrotation'%track.root_name
            # rzp = '%s_Zrotation'%track.root_name
            # pos_df[rxp] = pd.Series(data=euler_df[rxp], index=pos_df.index)
            # pos_df[ryp] = pd.Series(data=euler_df[ryp], index=pos_df.index)
            # pos_df[rzp] = pd.Series(data=euler_df[rzp], index=pos_df.index)
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)

            tree_data = {}
                        
            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                
                if 'Nub' not in joint:
                    r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                    expmap = r.values
                    #expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()]
                else:
                    expmap = np.zeros((exp_df.shape[0], 3))

                # Convert the eulers to rotation matrices
                #rotmats = np.asarray([Rotation(f, 'expmap').rotmat for f in expmap])
                #angs = np.linalg.norm(expmap,axis=1, keepdims=True)
                rotmats = self._expmap2rot(expmap)
                
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 
                pos_values = np.zeros((exp_df.shape[0], 3))

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + track.skeleton[joint]['offsets']

                    # multiply k to the rotmat of the parent for every frame i
                    q = np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = q.reshape(k.shape[0],3) + tree_data[parent][1]


                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition'%joint] = pd.Series(data=tree_data[joint][1][:,0], index=pos_df.index)
                pos_df['%s_Yposition'%joint] = pd.Series(data=tree_data[joint][1][:,1], index=pos_df.index)
                pos_df['%s_Zposition'%joint] = pd.Series(data=tree_data[joint][1][:,2], index=pos_df.index)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _to_expmap(self, X):
        """Converts Euler angles to Exponential Maps"""

        Q = []
        for track in X:
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep.
            exp_df = euler_df.copy()

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]]  # Get the columns that belong to this joint.
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation' % (joint, rot_order[0])
                r2_col = '%s_%srotation' % (joint, rot_order[1])
                r3_col = '%s_%srotation' % (joint, rot_order[2])
                
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                exps = unroll(np.array([euler2expmap(f, rot_order, True) for f in euler]))  # Convert the exp maps to eulers.

                # Create the corresponding columns in the new DataFrame.
                exp_df.insert(loc=0, column='%s_gamma' % joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta' % joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha' % joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q

    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = exp_df.copy()

            # Copy the root positions into the new DataFrame
            #rxp = '%s_Xposition'%track.root_name
            #ryp = '%s_Yposition'%track.root_name
            #rzp = '%s_Zposition'%track.root_name
            #euler_df[rxp] = pd.Series(data=exp_df[rxp], index=euler_df.index)
            #euler_df[ryp] = pd.Series(data=exp_df[ryp], index=euler_df.index)
            #euler_df[rzp] = pd.Series(data=exp_df[rzp], index=euler_df.index)
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_alpha'%joint, '%s_beta'%joint, '%s_gamma'%joint], axis=1, inplace=True)
                expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                rot_order = track.skeleton[joint]['order']
                #euler_rots = [Rotation(f, 'expmap').to_euler(True, rot_order) for f in expmap] # Convert the exp maps to eulers
                euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q


class Mirror(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        """
        Mirrors the data .
        """

        self.axis = axis
        self.append = append

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("Mirror: " + self.axis)
        Q = []
        
        if self.append:
            for track in X:
                Q.append(track)
            
        for track in X:
            if self.axis == "X":
                signs = np.array([1, -1, -1])
            if self.axis == "Y":
                signs = np.array([-1, 1, -1])
            if self.axis == "Z":
                signs = np.array([-1, -1, 1])

            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep.
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame.
            rxp = '%s_Xposition' % track.root_name
            ryp = '%s_Yposition' % track.root_name
            rzp = '%s_Zposition' % track.root_name
            new_df[rxp] = pd.Series(data=-signs[0]*euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=-signs[1]*euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=-signs[2]*euler_df[rzp], index=new_df.index)
            
            # List the joints that are left or right.
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint and 'Nub' not in joint)
            for lft_joint in lft_joints:
                rgt_joint = lft_joint.replace('Left', 'Right')

                # Create the corresponding columns in the new DataFrame.
                new_df['%s_Xrotation'%lft_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%rgt_joint], index=new_df.index)
                new_df['%s_Yrotation'%lft_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%rgt_joint], index=new_df.index)
                new_df['%s_Zrotation'%lft_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%rgt_joint], index=new_df.index)
                
                new_df['%s_Xrotation'%rgt_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%lft_joint], index=new_df.index)
                new_df['%s_Yrotation'%rgt_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%lft_joint], index=new_df.index)
                new_df['%s_Zrotation'%rgt_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%lft_joint], index=new_df.index)
    
            # List the joints that are not left or right, i.e. are on the trunk.
            joints = (joint for joint in track.skeleton if 'Nub' not in joint and 'Left' not in joint and 'Right' not in joint)
            for joint in joints:
                # Create the corresponding columns in the new DataFrame.
                new_df['%s_Xrotation'%joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%joint], index=new_df.index)
                new_df['%s_Yrotation'%joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%joint], index=new_df.index)
                new_df['%s_Zrotation'%joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%joint], index=new_df.index)

            new_track = track.clone()
            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X


class EulerReorder(BaseEstimator, TransformerMixin):
    def __init__(self, new_order):
        """
        Add a 
        """
        self.new_order = new_order
        
    
    def fit(self, X, y=None):
        self.orig_skeleton = copy.deepcopy(X[0].skeleton)
        print(self.orig_skeleton)
        return self
    
    def transform(self, X, y=None):
        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                
                euler = [[f[1]['%s_Xrotation'%(joint)], f[1]['%s_Yrotation'%(joint)], f[1]['%s_Zrotation'%(joint)]] for f in r.iterrows()]
                new_euler = [euler_reorder(f, rot_order, self.new_order, True) for f in euler]
                #new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)
                
                # Create the corresponding columns in the new DataFrame
                new_df['%s_%srotation'%(joint, self.new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)
    
                new_track.skeleton[joint]['order'] = self.new_order

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X
#        Q = []
#
#        for track in X:
#            channels = []
#            titles = []
#            euler_df = track.values
#
#            # Create a new DataFrame to store the exponential map rep
#            new_df = pd.DataFrame(index=euler_df.index)
#
#            # Copy the root positions into the new DataFrame
#            rxp = '%s_Xposition'%track.root_name
#            ryp = '%s_Yposition'%track.root_name
#            rzp = '%s_Zposition'%track.root_name
#            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
#            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
#            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)
#
#            # List the columns that contain rotation channels
#            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
#
#            # List the joints that are not end sites, i.e., have channels
#            joints = (joint for joint in track.skeleton if 'Nub' not in joint)
#
#            new_track = track.clone()
#            for joint in joints:
#                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
#                rot_order = track.skeleton[joint]['order']
#                new_order = self.orig_skeleton[joint]['order']
#                print("rot_order:" + str(rot_order))
#                print("new_order:" + str(new_order))
#
#                euler = [[f[1]['%s_%srotation'%(joint, rot_order[0])], f[1]['%s_%srotation'%(joint, rot_order[1])], f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in r.iterrows()]
#                #new_euler = [euler_reorder(f, rot_order, new_order, True) for f in euler]
#                new_euler = euler_reorder2(np.array(euler), rot_order, self.new_order, True)
#
#                # Create the corresponding columns in the new DataFrame
#                new_df['%s_%srotation'%(joint, new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
#                new_df['%s_%srotation'%(joint, new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
#                new_df['%s_%srotation'%(joint, new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)
#
#                new_track.skeleton[joint]['order'] = new_order
#                
#            new_track.values = new_df
#            Q.append(new_track)
#        return Q


class JointSelector(BaseEstimator, TransformerMixin):
    """
    Allows for filtering the moCap data to include only the selected joints.
    """

    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            selected_joints.append(X[0].root_name)
        
        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o and 'Nub' not in o])
        
        self.selected_joints = selected_joints
        self.selected_channels = selected_channels
        self.not_selected = X[0].values.columns.difference(selected_channels)
        self.not_selected_values = {c:X[0].values[c].values[0] for c in self.not_selected}

        self.orig_skeleton = X[0].skeleton
        return self

    def transform(self, X, y=None):
        print("JointSelector")
        Q = []
        for track in X:
            t2 = track.clone()
            for key in track.skeleton.keys():
                if key not in self.selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[self.selected_channels]

            Q.append(t2)
      

        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            t2.skeleton = self.orig_skeleton
            for d in self.not_selected:
                t2.values[d] = self.not_selected_values[d]
            Q.append(t2)

        return Q


class Numpyfier(BaseEstimator, TransformerMixin):
    """
    Just converts the values in a MocapData object into a numpy array.
    Useful for the final stage of a pipeline before training.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        print("Numpyfier")
        Q = []
        
        for track in X:
            Q.append(track.values.values)
            
        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q


class Slicer(BaseEstimator, TransformerMixin):
    '''
    Slice the data into intervals of equal size 
    '''
    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        print("Slicer")
        Q = []
        
        for track in X:
            vals = track.values.values
            nframes = vals.shape[0]
            overlap_frames = (int)(self.overlap*self.window_size)
            
            n_sequences = (nframes-overlap_frames)//(self.window_size-overlap_frames)
            
            if n_sequences>0:
                y = np.zeros((n_sequences, self.window_size, vals.shape[1]))

                # extract sequences from the input data
                for i in range(0,n_sequences):
                    frameIdx = (self.window_size-overlap_frames) * i
                    Q.append(vals[frameIdx:frameIdx+self.window_size,:])

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q


class RootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method, position_smoothing=0, rotation_smoothing=0):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
            hip_centric
        """

        self.method = method
        self.position_smoothing = position_smoothing
        self.rotation_smoothing = rotation_smoothing
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("RootTransformer")
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name
                
                x=track.values[xpcol].copy()
                z=track.values[zpcol].copy()
                
                if self.position_smoothing>0:
                    x_sm = filters.gaussian_filter1d(x, self.position_smoothing, axis=0, mode='nearest')    
                    z_sm = filters.gaussian_filter1d(z, self.position_smoothing, axis=0, mode='nearest')                    
                    dx = pd.Series(data=x_sm, index=new_df.index).diff()
                    dz = pd.Series(data=z_sm, index=new_df.index).diff()
                    new_df[xpcol] = x-x_sm
                    new_df[zpcol] = z-z_sm
                else:
                    dx = x.diff()
                    dz = z.diff()
                    new_df.drop([xpcol, zpcol], axis=1, inplace=True)
                    
                dx[0] = dx[1]
                dz[0] = dz[1]
                
                new_df[dxpcol] = dx
                new_df[dzpcol] = dz
                
                new_track = track.clone()
                new_track.values = new_df
            
            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition' % track.root_name
                yp_col = '%s_Yposition' % track.root_name
                zp_col = '%s_Zposition' % track.root_name

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation' % (track.root_name, rot_order[0])
                r2_col = '%s_%srotation' % (track.root_name, rot_order[1])
                r3_col = '%s_%srotation' % (track.root_name, rot_order[2])

                # Delta columns
                dxp_col = '%s_dXposition'%track.root_name
                dzp_col = '%s_dZposition'%track.root_name

                dxr_col = '%s_dXrotation'%track.root_name
                dyr_col = '%s_dYrotation'%track.root_name
                dzr_col = '%s_dZrotation'%track.root_name

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                """ Get Trajectory and smooth it"""                
                trajectory_filterwidth = self.position_smoothing
                reference = positions.copy()*np.array([1,0,1])
                if trajectory_filterwidth>0:
                    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
                
                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)                
                velocity = np.vstack((velocity[0,:], velocity))

                """ Remove Root Translation """
                positions = positions-reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                #forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                #print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)
                forward = quats*np.array([[0,0,1]])
                forward[:,1] = 0

                """ Smooth Forward Direction """                
                direction_filterwidth = self.rotation_smoothing
                if direction_filterwidth>0:
                    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    

                forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

                """ Remove Y Rotation """
                target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:,np.newaxis]    
                positions = (-rotation[:,0]) * positions
                new_rotations = (-rotation[:,0]) * quats

                """ Get Root Rotation """
                #print(rotation[:,0])
                velocity = (-rotation[:,0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in new_rotations])*180.0/np.pi
                
                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:,0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:,1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:,2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:,0], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:,2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:,0], index=new_df.index)
                
                #new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                #new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                #new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df

            elif self.method == 'hip_centric':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition' % track.root_name
                yp_col = '%s_Yposition' % track.root_name
                zp_col = '%s_Zposition' % track.root_name

                # xr_col = '%s_Xrotation' % track.root_name
                # yr_col = '%s_Yrotation' % track.root_name
                # zr_col = '%s_Zrotation' % track.root_name
                
                new_df = track.values.copy()

                all_zeros = np.zeros(track.values[xp_col].values.shape)

                new_df[xp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)

                # new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
                # new_df[yr_col] = pd.Series(data=all_zeros, index=new_df.index)
                # new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_track.values = new_df

            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        #TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0]-1):
                    recx.append(recx[i]+dx[i+1])
                    recz.append(recz[i]+dz[i+1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]
                if self.position_smoothing > 0:                    
                    new_df[xpcol] = pd.Series(data=new_df[xpcol]+recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=new_df[zpcol]+recz, index=new_df.index)
                else:
                    new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)
                
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                dxp_col = '%s_dXposition'%track.root_name
                dzp_col = '%s_dZposition'%track.root_name

                dyr_col = '%s_dYrotation'%track.root_name

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                #rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_zp = [0]

                #rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0]-1):
                    #print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i+1]), np.array([0,1,0]))
                    rec_ry[i+1] = q_y*rec_ry[i]
                    #print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i+1]*np.array([dx[i+1], 0, dz[i+1]])
                    rec_xp.append(rec_xp[i]+dp[0,0])
                    rec_zp.append(rec_zp[i]+dp[0,2])
                    
                rec_r=rec_ry*quats
                pp=rec_ry*positions
                rec_xp = rec_xp + pp[:,0]
                rec_zp = rec_zp + pp[:,2]
                
                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in rec_r])*180.0/np.pi
                
                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                
                new_df[xp_col] = pd.Series(data=rec_xp, index=new_df.index)
                new_df[zp_col] = pd.Series(data=rec_zp, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                new_df.drop([dyr_col, dxp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df
                
            #print(new_track.values.columns)
            Q.append(new_track)

        return Q


class RootCentricPositionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            for joint in all_but_root:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]-projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]-projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]-projected_root_pos[rzp], index=new_df.index)
            

            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]+projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]+projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]+projected_root_pos[rzp], index=new_df.index)
                

            new_track.values = new_df

            Q.append(new_track)
        
        return Q


class Flattener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)


# class ConstantsRemover(BaseEstimator, TransformerMixin):
#     """
#     For now it just looks at the first track.
#     """
#
#     def __init__(self, eps=1e-6):
#         self.eps = eps
#
#     def fit(self, X, y=None):
#         stds = X[0].values.std()
#         cols = X[0].values.columns.values
#
#         const_dims_tmp = [c for c in cols if (stds[c] < self.eps).any()]
#         joint_infos = {}
#         for col in const_dims_tmp:
#             infos = col.split('_')
#             channel = infos[-1][1:]
#             joint = '_'.join(infos[:-1])
#
#             if joint not in joint_infos.keys():
#                 joint_infos[joint] = [channel]
#             else:
#                 joint_infos[joint].append(channel)
#
#         self.const_dims_ = []
#         for joint in joint_infos.keys():
#             if "position" in joint_infos[joint]:
#                 counter = 0
#                 for c in joint_infos[joint]:
#                     if c == "position":
#                         counter += 1
#                 if counter == 3:
#                     self.const_dims_.append(f"{joint}_Xposition")
#                     self.const_dims_.append(f"{joint}_Yposition")
#                     self.const_dims_.append(f"{joint}_Zposition")
#
#             if "rotation" in joint_infos[joint]:
#                 counter = 0
#                 for c in joint_infos[joint]:
#                     if c == "rotation":
#                         counter += 1
#                 if counter == 3:
#                     self.const_dims_.append(f"{joint}_Xrotation")
#                     self.const_dims_.append(f"{joint}_Yrotation")
#                     self.const_dims_.append(f"{joint}_Zrotation")
#
#             if ("position" not in joint_infos[joint]) and ("rotation" not in joint_infos[joint]):
#                 print(joint_infos[joint])
#                 raise ValueError
#
#         # self.const_dims_ = [c for c in cols if (stds[c] < self.eps).any()]
#
#         self.const_values_ = {c:X[0].values[c].values[0] for c in self.const_dims_}
#
#         return self
#
#     def transform(self, X, y=None):
#         Q = []
#
#         for track in X:
#             t2 = track.clone()
#             t2.values.drop(self.const_dims_, axis=1, inplace=True)
#             Q.append(t2)
#
#         return Q
#
#     def inverse_transform(self, X, copy=None):
#         Q = []
#
#         for track in X:
#             t2 = track.clone()
#             for d in self.const_dims_:
#                 t2.values[d] = self.const_values_[d]
# #                t2.values.assign(d=pd.Series(data=self.const_values_[d], index = t2.values.index))
#             Q.append(t2)
#
#         return Q


class ConstantsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        root_name = X[0].root_name

        self.const_dims1_ = [
            f"{root_name}_Xposition",
            f"{root_name}_Yposition",
            f"{root_name}_Zposition",
            # f"{root_name}_Xrotation",
            # f"{root_name}_Yrotation",
            # f"{root_name}_Zrotation"
        ]

        self.const_values1_ = {c:X[0].values[c].values[0] for c in self.const_dims1_}


        stds = X[0].values.std()
        cols = X[0].values.columns.values

        const_dims_tmp = [c for c in cols if (stds[c] < self.eps).any()]
        joint_infos = {}
        for col in const_dims_tmp:
            infos = col.split('_')
            channel = infos[-1][1:]
            joint = '_'.join(infos[:-1])

            if joint not in joint_infos.keys():
                joint_infos[joint] = [channel]
            else:
                joint_infos[joint].append(channel)

        self.const_dims_ = []
        for joint in joint_infos.keys():
            if "position" in joint_infos[joint]:
                counter = 0
                for c in joint_infos[joint]:
                    if c == "position":
                        counter += 1
                if counter == 3:
                    self.const_dims_.append(f"{joint}_Xposition")
                    self.const_dims_.append(f"{joint}_Yposition")
                    self.const_dims_.append(f"{joint}_Zposition")

            if "rotation" in joint_infos[joint]:
                counter = 0
                for c in joint_infos[joint]:
                    if c == "rotation":
                        counter += 1
                if counter == 3:
                    self.const_dims_.append(f"{joint}_Xrotation")
                    self.const_dims_.append(f"{joint}_Yrotation")
                    self.const_dims_.append(f"{joint}_Zrotation")

            if ("position" not in joint_infos[joint]) and ("rotation" not in joint_infos[joint]):
                print(joint_infos[joint])
                raise ValueError

        # self.const_dims_ = [c for c in cols if (stds[c] < self.eps).any()]

        self.const_values_ = {c:X[0].values[c].values[0] for c in self.const_dims_}

        # print(self.const_dims1_)
        # print(self.const_dims_)

        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            t2 = track.clone()
            t2.values.drop(self.const_dims1_, axis=1, inplace=True)
            Q.append(t2)

        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]
            #                t2.values.assign(d=pd.Series(data=self.const_values_[d], index = t2.values.index))
            Q.append(t2)

        return Q


class ListStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)


class ListMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_max_ = np.max(X_train_flat, axis=0)
        self.data_min_ = np.min(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_min_) / (self.data_max_ - self.data_min_) 
            else:
                normalized_track = (track - self.data_min_) / (self.data_max_ - self.data_min_)

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * (self.data_max_ - self.data_min_)) + self.data_min_
            else:
                unnormalized_track = (track * (self.data_max_ - self.data_min_)) + self.data_min_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)


class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, tgt_fps, keep_all=False):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            orig_fps = round(1.0 / track.framerate)
            rate = orig_fps // self.tgt_fps
            if orig_fps % self.tgt_fps != 0:
                print("error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            else:
                print("downsampling with rate: " + str(rate))

            for ii in range(0, rate):
                new_track = track.clone()
                new_track.values = track.values[ii::rate].copy()
                new_track.framerate = 1.0 / self.tgt_fps
                Q.append(new_track)

                if not self.keep_all:
                    break
        
        return Q


    def inverse_transform(self, X, copy=None):
      return X


class ReverseTime(BaseEstimator, TransformerMixin):
    def __init__(self, append=True):
        self.append = append
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        if self.append:
            for track in X:
                Q.append(track)
        for track in X:
            new_track = track.clone()                            
            new_track.values = track.values[-1::-1]
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X


#TODO: JointsSelector (x)
#TODO: SegmentMaker
#TODO: DynamicFeaturesAdder
#TODO: ShapeFeaturesAdder
#TODO: DataFrameNumpier (x)


class TemplateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

