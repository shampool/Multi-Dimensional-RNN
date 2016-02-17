from __future__ import absolute_import
# -*- coding: utf-8 -*-
import numpy as np
import random
from six.moves import range
import matplotlib.pyplot as plt
from local_utils import imshow

def mysqueeze(a, axis = None):
    if axis == None:
        return np.squeeze(a)
    if a.shape[axis] != 1:
        return a
    else:
        return np.squeeze(a, axis = axis)
def getImg_from_Grid(grid_vec, patchsize):
    patchRow, patchCol = patchsize    
    indx =  -1
    #if len(img_vec) == 0:
    #    return None
    imgchannel = int(grid_vec.shape[-1]//(patchRow*patchCol)) 
    numberofImg = grid_vec.shape[0]
    gridshape = (grid_vec[0,:,:,:].shape[0],grid_vec[0,:,:,:].shape[1])
    imgs = np.zeros((grid_vec.shape[0], gridshape[0]*patchRow, gridshape[1]*patchCol, imgchannel ))
    imgs = mysqueeze(imgs, axis = -1)
    
    for imgidx  in range(numberofImg):   
        for colid in range(gridshape[1]):   
           for rowid in range(gridshape[0]):
              indx = indx + 1
              this_vec =  grid_vec[imgidx,rowid,colid,:]
              this_patch = np.reshape(this_vec, (patchRow,patchCol,imgchannel ))
              this_patch = mysqueeze(this_patch,axis = -1)
              startRow, endRow = rowid *patchRow, (rowid+1)*patchRow
              startCol, endCol = colid *patchCol, (colid+1)*patchCol
              #print this_patch.shape
              imgs[imgidx,startRow:endRow,startCol: endCol] = this_patch
              #imshow(img)
    return imgs
def showVec(img_vec,gridshape, patchsize):
    patchRow, patchCol = patchsize    
    indx =  -1
    imgchannel = int(img_vec.shape[-1]//(patchRow*patchCol)) 
    img = np.zeros((gridshape[0]*patchRow, gridshape[1]*patchCol,imgchannel ))
    img = mysqueeze(img,axis = -1)
    for colid in range(gridshape[1]):   
       for rowid in range(gridshape[0]):
          indx = indx + 1
          this_vec =  img_vec[indx,:]
          this_patch = np.reshape(this_vec, (patchRow,patchCol,imgchannel))
          this_patch = mysqueeze(this_patch,axis = -1)
          startRow, endRow = rowid *patchRow, (rowid+1)*patchRow
          startCol, endCol = colid *patchCol, (colid+1)*patchCol
          img[startRow:endRow,startCol: endCol] = this_patch
          imshow(img)
          
def showGrid(grid_vec,gridshape, patchsize):
    patchRow, patchCol = patchsize    
    imgchannel = int(grid_vec.shape[-1]//(patchRow*patchCol)) 
    img = np.zeros((gridshape[0]*patchRow, gridshape[1]*patchCol ,imgchannel))
    img = mysqueeze(img,axis = -1)
    for colid in range(gridshape[1]):   
       for rowid in range(gridshape[0]):
          this_vec =  grid_vec[rowid,colid,:]
          this_patch = np.reshape(this_vec, (patchRow,patchCol,imgchannel ))
          this_patch = mysqueeze(this_patch,axis = -1)
          startRow, endRow = rowid *patchRow, (rowid+1)*patchRow
          startCol, endCol = colid *patchCol, (colid+1)*patchCol
          img[startRow:endRow,startCol: endCol] = this_patch
          imshow(img)
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    
def pad_vector_grid(vector_sequences, grid_shape_sequences, max_grid_shape = None, dtype = 'int32', padding = 'pre', truncating =  'pre', value = 0.):
    """
        pad vector_sequences a  list (nb_samples, 1) each contain(timestep, dim).
        return as (nb_samples, maxlen, dim) numpy tensor
        The mask will also be returned as well for those shorter than the maxlen
        
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.
        
        Supports post-padding and pre-padding (default).         
     """
	
    row_lengths = [s[0] for s in grid_shape_sequences]
    col_lengths = [s[1] for s in grid_shape_sequences]
    assert  vector_sequences[0] is not None     
    dim = vector_sequences[0].shape[1] 
    nb_samples = len(vector_sequences)
    if max_grid_shape is None:
        max_grid_shape = (np.max(row_lengths), np.max(col_lengths))
    x =  np.ones( (nb_samples,)+ max_grid_shape +(dim,)).astype(dtype)* value    
    mask = np.zeros((nb_samples,)+max_grid_shape)
    for idx, vs in enumerate(vector_sequences):
        if len(vs) == 0:
            continue
        grid_vec = np.reshape(vs,(tuple(grid_shape_sequences[idx]) + (dim,)) , order='F')
        # testiing code
        #patchRow, patchCol = [25,25]
        #showGrid(grid_vec, grid_shape_sequences[idx], [patchRow, patchCol])        
        if truncating == 'pre':            
            trunc = grid_vec[-max_grid_shape[0]:,-max_grid_shape[1]:,:]
        elif truncating == 'post':
            trunc = grid_vec[:max_grid_shape[0],:max_grid_shape[1],:]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        
        if padding == 'post':
           x[idx,:trunc.shape[0],:trunc.shape[1],:] = trunc.copy()
           mask[idx, :trunc.shape[0],:trunc.shape[1]] = 1
        elif padding == 'pre':
           x[idx, -trunc.shape[0]:,-trunc.shape[1]:, :] = trunc.copy()
           mask[idx, -trunc.shape[0]:, -trunc.shape[1]:] = 1
        else:
           raise ValueError("PAdding type '%s' not understood" % padding)           
        #showGrid(x[idx,::], max_grid_shape, [patchRow, patchCol])        
    return x , mask# -*- coding: utf-8 -*-

def pad_vector_grid_sequence(vector_sequences, grid_shape_sequences, max_grid_shape = None, dtype = 'int32', padding = 'pre', truncating =  'pre', value = 0.):
    """
        pad vector_sequences a  list (nb_samples, 1) each contain(timestep, dim).
        return as (nb_samples, maxlen, dim) numpy tensor
        The mask will also be returned as well for those shorter than the maxlen
        
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen. Truncation happens off either the beginning (default) or
        the end of the sequence.
        
        Supports post-padding and pre-padding (default).         
     """
	
    grid_X, grid_mask = pad_vector_grid(vector_sequences, grid_shape_sequences, max_grid_shape, dtype, padding , truncating, value)
    
    padded_X = np.reshape(grid_X, (grid_X.shape[0], np.prod(max_grid_shape,grid_X.shape[-1])))
    padded_mask = np.reshape(grid_mask, (grid_mask.shape[0], np.prod(max_grid_shape)), order='F')
    # needs to test if the reshape works just fine.
    return padded_X, padded_mask
