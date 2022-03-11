import numpy as np
import time
#from skimage.draw import polygon_perimeter
from skimage.draw import rectangle
from skimage.measure import regionprops
from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import watershed
from PIL import Image

def my_prepro(state, pellet_list=None):
    """
    Preprocess state (210, 160, 3) image into
    a (40, 40, 142) image in BW

    source: https://scikit-image.org/docs/dev/auto_examples/segmentation/
            plot_expand_labels.html#sphx-glr-auto-examples-segmentation-plot-expand-labels-py

    """
    # TODO: enforce rectangle max size!!
    debug = False
    if debug: 
        print('Debugging vision component!')
        orig = state

    state = state[5:165,:,:]  # crop
    state = state[::2,::2,:] # downsample by factor of 2
    
    if debug:
        im = Image.fromarray(state)#.convert('RGB')
        now=time.time()
        im.save(f'imgs/img_array{now}.png')
#        np.save(f'imgs/img_array{now}.png', state)


    # prep output vector
    output = np.zeros((state.shape[0], state.shape[1], 1+4+4+149+2))
    
    colors = np.load('utils/pawn_colors.npy')        

    # detect pacman and four ghosts 
    # TODO: make this parallel    
    for i, color in enumerate(colors):
        X,Y = np.where(np.all(state[:,:,0:1]==color[0:1],axis=-1))
        if X.shape[0] != 0 and Y.shape[0]!=0:
            minr, minc, maxr, maxc = Y.min(), X.min(), Y.max(), X.max()
            # top left and bottom right points of our box
            start, end = (minc, minr), (maxc, maxr)
            # get tracking rectangle
            rr, cc = rectangle(start, end=end)
            output[rr, cc, 0 + i] = 255
            # draw tarcking boxes
            if debug: state[rr, cc] = [255,0,0]        

    # detect blue ghosts
    #TODO: get rid of magic numbers
    img = state
    count = 4
    arr = (img[..., 0] > 64) & (img[..., 0] < 67) & (img[..., 1] > 112) & \
            (img[..., 1] < 115) & (img[..., 2] > 192) & (img[..., 2] < 196)
    rr, cc = np.where(arr)
    var = np.zeros((img.shape[0], img.shape[1], 1))
    var[rr, cc,-1]=255
    
    if rr.shape[0]>0:
        img = var[...,-1]
        edges = sobel(img)
        
        markers = np.zeros_like(img)
        foreground, background = 1, 2
        markers[img < 30.0] = background
        markers[img > 150.0] = foreground
        
        ws = watershed(edges, markers)
        seg1 = label(ws == foreground)
                
        if len(seg1) > 0:
            regions = regionprops(seg1)
            
            for region in regions:
                if region.area >= 5 and region.area < 90 and count > 0:
                    # control output length
                    count -= 1
                    minr, minc, maxr, maxc = region.bbox
                    start, end = (minc, minr), (maxc-1, maxr-1)
                    rr, cc = rectangle(start, end=end)
                    # collect pellets
                    # TODO: axis inversion might throw troubles
                    output[cc, rr, 4 + i] = 255
                    # draw tarcking boxes
                    if debug: state[cc, rr] = [0,255,0]
            if debug:
                im = Image.fromarray(state)#.convert('RGB')
                im.save(f'imgs/debug/img_array{time.time()}.png')

    
    # detect pellets    
    pellet_col = np.array([228, 111, 111], dtype=np.int16)
    
    for i, pellet in enumerate(pellet_list):
        if (state[tuple(pellet)][-1][-1][0] == pellet_col[0]).all():
            output[pellet[0], pellet[1], 9 + i] = 255
            if debug: state[tuple(pellet)] = [255,255,255]
    
    if debug: 
        state=state[::2,::2,:]
        im = Image.fromarray(orig)#.convert('RGB')
        im.save(f'imgs/final_full.png')
        im = Image.fromarray(state)#.convert('RGB')
        im.save(f'imgs/final_scaled.png')


        return state

    output = output[::2,::2,:]
#    np.save('output', output)
    
#    np.save('utils/pellets.npy', output[..., 9:])
    return output#[..., 1:] + output[..., 0][..., np.newaxis]

def pics(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
#    state = np.reshape(state, [210, 160, 3])#.astype(np.float32)

    # crop
#    state = state.copy()
    state = state[10:170,:,:]  # crop
    state = state[::2,::2,:] # downsample by factor of 2
    
    #save imgs
    now = time.time()
    np.save(f'imgs/blues/img_array{now}', state)

    from PIL import Image
    im = Image.fromarray(state*255)#.convert('RGB')
#    im.show()
    im.save(f'imgs/blues/img_array{now}.png')
                
    return state
