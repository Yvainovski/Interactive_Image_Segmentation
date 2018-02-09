
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

def help_message():
   print("Usage: [Input_Image] ")
   print("[Input_Image]")
   print("Path to the input image")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png ")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    #18.5  - 6.6959
    #18.45 - 6.6956
    #18.46 - 6.6953
    segments = slic(img, n_segments=500, compactness=18.46,convert2lab =True)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    
    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])
    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    res = []
    for h in histograms:
        if(h.sum()==0):
            res.append(h)
        else:
            res.append(h / h.sum())
    return np.float32(res)

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):

        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

def SLIC_CUT(img, img_marking):
    centers,color_hists,superpixels,neighbors = superpixels_histograms_neighbors(img)
    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

    norm_hists = normalize_histograms(color_hists)
    fgbg_hists = [fg_cumulative_hist,bg_cumulative_hist]
    fgbg_superpixels=[fg_segments,bg_segments]
    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)


    segments_ids = np.unique(superpixels)
    astronaut_segmentation = np.zeros_like(img)
    astronaut_segmentation = astronaut_segmentation
    #
    cut_segments_ids = segments_ids[np.nonzero(graph_cut)]
    cut_locs_pairs = np.array([np.nonzero(superpixels==i) for i in cut_segments_ids]) 

    for loc in cut_locs_pairs:
        astronaut_segmentation[loc[0],loc[1]] = 255

    return astronaut_segmentation

is_drawing = False
is_fg = False
def GET_MARKING(img):
    # drawing fg first : True
    global is_fg 
    is_fg = True
    # init a white blank marking
    img_marking = np.ones_like(img,dtype=np.uint8)
    img_marking[:,:]=(255,255,255)
    def DRAW_MARKING(event,x,y,flags,param):
        global is_drawing, is_fg
        thickness = 4
        if(event == cv2.EVENT_LBUTTONDOWN):
            is_drawing = True
        elif(event == cv2.EVENT_MOUSEMOVE and is_drawing==True):
            # color : red for fg
            drawing_color = (0,0,255) 
            if(is_fg==False):
                # color : blue for bg
                drawing_color = (255,0,0) 
            cv2.circle(img,(x,y),thickness,drawing_color,-1)
            img_marking[ y-thickness:y+thickness , x-thickness:x+thickness] = drawing_color
        if(event == cv2.EVENT_LBUTTONUP):
            is_drawing =False
        pass

    cv2.namedWindow('Please Draw the Marking')
    cv2.setMouseCallback('Please Draw the Marking',DRAW_MARKING)

    while(1):
        cv2.imshow('Please Draw the Marking',img)
        key_code = cv2.waitKey(20)
        # pre Q to stop drawing after drawing bg
        if(key_code & 0xFF ==113 and is_fg == False):
            break
        # press key Q to change the mode drawing from fg to bg
        elif(key_code & 0xFF ==113):
            is_fg = False
    # return marking and img with marking
    return img_marking, img


if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

    img_marking ,img_with_drawing= GET_MARKING(img)

    mask = SLIC_CUT(img,img_marking)
    cv2.imshow('Please Draw the Marking',np.hstack((img_with_drawing,mask)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
