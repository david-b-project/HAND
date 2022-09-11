print('Importing...',end='')
import time
start=time.time()
mylast=time.time()

import rasterio
from skimage import filters
import numpy as np
import math 

from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd

def seconds():
    global mylast
    end = time.time() 
    diff = end - start
    delta = end - mylast
    print('{} min {} s, delta {} s'.format(math.floor(diff/60), round(diff%60,2), round(delta,2) ) );
    mylast = end

print('Python package import complete')
seconds()

## 9514026  9515736  9515730  9517818

def merge(huc6, ix, s_limit, i_limit):

    print(f'\n-----{huc6}_{ix}_s{s_limit}_i{i_limit}-----')

    file1 = '/home/db1142/HAND/ModelFlow/avg5maps/'+str(huc6)+'_'+str(ix)+'.tif'
    # file1 = '/scratch/db1142/hand_v020/'+str(huc6)+'_'+str(ix)+'_avg5.tif'  ## 020301_2_5_avg5.tif
    # file1 = '/home/db1142/HAND/fc-nwm/Ida/20210902_' + str(ix) + '.tif'
    
    file2 = '/scratch/db1142/HAND/hand_v020/'+str(huc6)+'/'+str(huc6)+'hand.tif'
    file3 = '/scratch/db1142/HAND/hand_v020/'+str(huc6)+'/'+str(huc6)+'catchhuc.tif'
    file4 = '/scratch/db1142/HAND/hand_v020/'+str(huc6)+'/'+str(huc6)+'fel.tif'

    catchprop = pd.read_csv('/home/db1142/HAND/ModelFlow/Catchment_Regression_Data.csv')
    htable = pd.read_csv('/home/db1142/HAND/ModelFlow/'+str(huc6)+'/hydrogeo-fulltable-'+str(huc6)+'.csv')
    
    fcfile='/home/db1142/HAND/ModelFlow/avg5depth/inun-hq-table-at-20200101_000000-for-2020010'+ix[:-1]+'0'+ix[-1]+'0000.csv'
    # print(fcfile)
    fctable = pd.read_csv(fcfile)
    
    dataset1 = rasterio.open(file1)
    x1 = dataset1.read(1)
    print(f'x1 = {x1.shape}')
    # figure(figsize=(8, 6)); plt.title('Flood_raw'); plt.imshow(x1, cmap='plasma'); plt.show()
    ds_ht = dataset1.height
    ds_wd = dataset1.width
    ds_ts = dataset1.transform
    ds_cr = dataset1.crs

    x2 = x1.copy()
    x2[x2<0] = 0 # replace negative flood depth with zero flood depth
    print(f'x2 = {x2.shape}')
    del x1, dataset1

    edges_x2 = filters.sobel(x2.copy())
    print(f'edges_x2 = {edges_x2.shape}')

    print('\nFlood_sobel histogram\ncount   :   bin')
    count, bins = np.histogram(edges_x2)
    for i in range(len(count)):
        print(f'{count[i]} : {round(bins[i],3)}')
    print('')

    # s_limit = 4 # The maximum value allowed for the Sobel filter. Refer to sobel histogram to pick a value
    # i_limit = 2 # Number of iterations
    if np.max(edges_x2) < s_limit:
        print('No values greater than s_limit {s_limit}')
        return

    dataset2 = rasterio.open(file2)
    y1 = dataset2.read(1)
    print(f'y1 = {y1.shape}')
    del dataset2

    dataset3 = rasterio.open(file3)
    z1 = dataset3.read(1)
    print(f'z1 = {z1.shape}')
    del dataset3

    dataset4 = rasterio.open(file4)
    w1 = dataset4.read(1)
    print(f'w1 = {w1.shape}')
    del dataset4

    buf = (w1.shape[1] - z1.shape[1])/2
    print(f'w1 buffer = {buf}')

    if (huc6 == '020301') or (huc6 == '020403'): hzb = slice(1080,-1079)
    else: hzb = slice(1080,-1080)
    # hzb = slice(1079,-1080)
    vtb = slice(1080,-1080)

    w2 = w1[hzb,vtb].copy()
    print(f'w2 = {w2.shape}')

    seconds()
    
    ## Locate strong edges in flood image and merge catchments

    flood = x2
    sobel = edges_x2
    hand = y1
    catch = z1
    dem_pit = w2

    k_size = 3 # kernel size is x * x square, where x must be odd.
    k_pad  = k_size // 2

    count = 0 # Iteration counter
    ierr  = 0 # error counter
    perr  = 0 # error counter
    oerr  = 0 # error counter
    s_val = 0 # Number of remaining sobel pixels, used to determine convergence
    
    keep_last = None
    toss_last = None

    h1 = 0
    h2 = 0
    
    pairs = []
    tossed = []

    print(f'Looking for sobel values > {s_limit}. There are {len(sobel[sobel>s_limit])} values above the limit')

    ## Run iterations until you have corrected all sobel values greater than limit 
    # while sobel.max() > s_limit:
    while count <= i_limit: # use this for testing

        ## Return x and y indicies of sobel values above the limit
        xa, ya = np.where(sobel > s_limit)

        ## Iterate through every problem pixel
        for val in range(len(xa)):
            x = xa[val]
            y = ya[val]
            # print(val)
            
            ## Work on sobel pixel at x,y - determine the indices of the kernel window
            ## These variables make the following code a bit cleaner
            lf = x - k_size // 2     # left   window limit
            rt = x + k_size // 2 + 1 # right  window limit
            dn = y - k_size // 2     # bottom window limit  
            up = y + k_size // 2 + 1 # top    window limit

    #         print(f'Sobel edge at x={x},y={y}')

            ## Ignore pixels at the boundary of flood image - fix later with padding?
    #         if (x <= k_pad) or (x >= sobel.shape[0] - k_pad): continue
    #         if (y <= k_pad) or (y >= sobel.shape[1] - k_pad): continue

            ## create k*k subset of sobel image for visual analysis
            sobel_ss = sobel[lf : rt, dn : up]   #; print(f'sobel_subset = \n{sobel_ss}')

            ## create a subset of catchments
            catch_ss = catch[lf : rt, dn : up] ; #print(f'catch_ss = \n{catch_ss}')

            ## determine catchment orders and define keep and toss catchments
            list_a = np.unique(catch_ss)
            if  len(list_a) == 1:
    #             print("Internal edge, not a boundary issue, skipping...")
                ierr += 1
                continue
            elif len(list_a) > 2:
    #             print("More than 2 catchments, skipping...");
                perr += 1
                continue
            else:
    #             print(list_a)
                if len(catchprop.loc[catchprop.CatchId == list_a[0], 'StreamOrde']) == 0:
                    print('len aO = 0, ended.')
                    continue
                if len(catchprop.loc[catchprop.CatchId == list_a[1], 'StreamOrde']) == 0:
                    print('len bO = 0, ended.')
                    continue
                aO = catchprop.loc[catchprop.CatchId == list_a[0], 'StreamOrde'].values[0]
                bO = catchprop.loc[catchprop.CatchId == list_a[1], 'StreamOrde'].values[0]
                if   aO > bO: keep = list_a[0]; toss = list_a[1]
                elif aO < bO: keep = list_a[1]; toss = list_a[0]
                else:
    #                 print('Catchments are same order, skipping...');
                    oerr += 1
                    continue
    #             print(f'{list_a[0]}: Order {aO} -- {list_a[1]}: Order {bO}') 
            

            pairx = str(keep) + ' ' + str(toss) 
            if pairx in pairs: ## if we've already added this pair once, no need to add it again
                # print(f'duplicate merge, aborted')
                continue
            
            if (keep == keep_last) and (toss == toss_last):
                # print(f'duplicate merge, aborted')
                continue
            else:
                keep_last = keep
                toss_last = toss
            
            if toss in tossed:
                print(f'Already tossed this toss : keep={keep}  toss={toss}.')
                continue
            
            ## use this for checking problem areas
            # if keep not in [9513372, 9513346, 9513500, 9513382, 9513478, 9513452, 9513420, 9513466, 9513478, 9513380]: ## [9515730, 9517818]:   ## DEBUGG    6246680  6249470  6251052  26814495  6250024  6250336  9512882            ##9514026,  9515736,  9515730,  9517818 
            #     continue
            # # print(f'{keep}   h1 = {str(round(h1,3))},  h2 = {str(round(h2,3))}   CHECK')
            
            print(f'Merge keep = {keep} & toss = {toss}...',end='')

            ## Create slices to work with the merge data
            ya1, xa1 = np.where(catch == toss)
            ya2, xa2 = np.where(catch == keep)
            xmin = int(min(min(xa1), min(xa2))); xmax = int(max(max(xa1), max(xa2)))
            ymin = int(min(min(ya1), min(ya2))); ymax = int(max(max(ya1), max(ya2)))
            hzx = slice(xmin, xmax, 1)
            vtx = slice(ymin, ymax, 1)

            hand2  =  hand[vtx, hzx].copy()
            catch1 = catch[vtx, hzx].copy() #clean
            catch2 = catch[vtx, hzx].copy() #modified
            flood1 = flood[vtx, hzx].copy() #clean
            flood2 = flood[vtx, hzx].copy() #modified
            dem_pit2 = dem_pit[vtx, hzx].copy()

            ## redefine HAND and CatchmentID for the toss catchment - HIGHLY SIMPLIFIED
            ## new hand elevation in toss = original DEM elevation minus the minimum elevation of the keep
            hand2[np.where(catch2 == toss)] = dem_pit2[np.where(catch2 == toss)] - \
                                              dem_pit2[np.where(catch2 == keep)].min()
            
            # testVal1 = dem_pit2[np.where(catch2 == toss)].min()  ;  # testVal2 = dem_pit2[np.where(catch2 == keep)].min()
            # print(f're-HAND  elev1 = {str(round( testVal1, 3))}  elev2 = {str(round( testVal2, 3))}')
            catch2[np.where(catch2 == toss)] = keep

            ## clip the rating curve at the keep catchment
            clip = htable.loc[htable.CatchId == keep].copy() 
            if len(clip) == 0:
                print('len(clip) == 0, aborted.   ')
                continue
            
            ## get Q from forecast, interpolate to get depth
            try:
                flow = fctable.loc[fctable['COMID'] == keep, 'Q'].values[0]
            except IndexError:
                print('No flow data, skipping.   ')
                continue

            flow = fctable.loc[fctable['COMID'] == keep, 'Q'].values[0]
            h1 = np.interp(flow, clip['Discharge (m3s-1)'], clip['Stage'])
            if h1 == 0.0:
                print('h1 = 0.0, aborted.   ')
                continue
            
            # h1 = flood2[np.where(catch1 == keep)].max()
            # flow = np.interp(h1, clip['Stage'], clip['Discharge (m3s-1)'])
            # print(f'q={str(round(flow,3))}...fcQ={str(round(fcflow,3))}',end='')
            
            
            
            ## recalculate the rating curve
            for x in clip['Stage']:  ## for every 1-foot stage increment, count pixels
                clip.loc[clip['Stage'] == x, 'Number of Cells'] = len(hand2[np.where((catch2==keep) & (hand2 <= x))])
                # print(f'cells = {len(hand2[np.where((catch2==keep) & (hand2 <= x))])}')
            
            ## Calculate area from pixels, calculate BedArea from SurfaceArea
            clip['SurfaceArea (m2)'] = clip['Number of Cells']  *   80   ## APPROXIMATE, ranges from 79.8-80.8
            clip['BedArea (m2)']     = clip['SurfaceArea (m2)'] * 1.01   ## APPROXIMATE, ranges from 1.001-1.03

            arr0 = []  ## calculate cumulative volume at each increment
            for x in clip['SurfaceArea (m2)']:
                arr0.append(x)
                clip.loc[clip['SurfaceArea (m2)'] == x, 'Volume (m3)'] = \
                   clip.loc[clip['SurfaceArea (m2)'] == x, 'Stage'] * np.average(arr0)
                # print(f'Vol = ',end="")
                # print(clip.loc[clip['SurfaceArea (m2)'] == x, 'Stage'] * np.average(arr0)) 

            clip['TopWidth (m)'] = clip['SurfaceArea (m2)'] / clip['LENGTHKM'] / 1000
            clip['WettedPerimeter (m)'] = clip['BedArea (m2)'] / clip['LENGTHKM'] / 1000

            c1 = 1
            while c1 <= (len(clip.index) - 1): ## APPROXIMATE VOLUME CALC, within 1%
                clip['Volume (m3)'].iloc[c1] = (clip['SurfaceArea (m2)'].iloc[c1] + \
                                                clip['SurfaceArea (m2)'].iloc[c1 - 1]) / 2 * 0.3048
                clip['Volume (m3)'].iloc[c1] = clip['Volume (m3)'].iloc[c1] + clip['Volume (m3)'].iloc[c1 - 1]
                c1 += 1 

            clip['WetArea (m2)'] = clip['Volume (m3)'] / clip['LENGTHKM'] / 1000
            clip['HydraulicRadius (m)'] = clip['WetArea (m2)'] /  clip['WettedPerimeter (m)']

            clip['Discharge (m3s-1)'] = clip['WetArea (m2)'] * pow(clip['HydraulicRadius (m)'],2.0/3) * \
                    pow(clip['SLOPE'],0.5) / clip['Roughness_y']

            ## recalculate the height from the flow data
            h2 = np.interp(flow, clip['Discharge (m3s-1)'], clip['Stage'])
            
            pairx = str(keep) + ' ' + str(toss) 
            pairs.append(pairx)
            
            rx = str(round(clip['Roughness_y'].iloc[0],3))

            if h2 > h1: ## if our height is greater after merging, abort. height should go down after merging
                print(f'h2 > h1, n={rx}  aborted.                               ',end='')
                seconds()
                continue
            
            if h2 < h1 * 0.3: ## if our height is decreased by too much after merging, abort. height should go down a bit after merging
                print(f'h2 < h1 * 0.3, n={rx}  aborted.                               ',end='')
                seconds()
                continue
            
            htable.loc[htable.CatchId == keep] = clip.copy()
            
            print(f'rough = {rx}...',end='')
            
            ## calculate flooding in new merged catchment
    #         flood2 = flood.copy()

            ## new flooding for keep
            flood2[np.where(catch2 == keep)] = h2 - hand2[np.where(catch2 == keep)].copy()
            flood2[flood2 < 0 ] = 0

            ## maximum of new and old flooding in tossed catchment
            flood2[np.where(catch1==toss)] = np.maximum(flood1[np.where(catch1==toss)],\
                                                            flood2[np.where(catch1==toss)])

            flood[vtx, hzx] = flood2
            catch[vtx, hzx] = catch2
            hand[vtx, hzx]  = hand2
            
            print('complete!  ',end='')
            print(f'{keep}   h1 = {str(round(h1,3))},  h2 = {str(round(h2,3))}   ', end="")
            
            # test_depth = flood[np.where(catch == 9514908)].max()
            # print(f'max depth in 9514908 = {test_depth}') ## dem_pit2[np.where(catch2 == keep)].min() 
            
            tossed.append(toss)
            seconds()
            # print(clip)

        s_val = len(sobel[sobel>s_limit])
        print(f'Iteration {count} complete: {s_val} sobel values above limit')
        count = count + 1

    print(f'Skipped internal edge pixels  = {ierr}')
    print(f'Skipped multiple catch pixels = {perr}')
    print(f'Skipped same order pixels     = {oerr}')
    
    flood[flood > 50] = -10
    # flood[flood <-10] = -10
    
    if ierr + perr + oerr == s_val:
        print('No changes made.')
    else:
        print('Complete!')

        fname = '/scratch/db1142/HAND/merged/' + str(huc6) + '_' + str(ix) + '_avg5_merge_lim' + str(s_limit) + 'iter' + str(i_limit) + '.tif'
        # fname = '/scratch/db1142/merged/20210902_' + str(ix) + '_merge.tif'
        print(f'\nCreating {fname}...',end='')

        with rasterio.open(
            fname,
            'w',
            driver='GTiff',
            height=ds_ht,
            width=ds_wd,
            count=1,
            dtype=np.single,
            crs=ds_cr,
            transform=ds_ts,
        ) as dst:
            dst.write(flood, 1)
        # dtype=np.float64
        print(f'complete!')







# for huc6 in ['020401', '020402', '020403',]: #   '020200', '020301', 
#     for a in ['1','2']: ##, '2']:
#         for b in ['0','1','2','3','4','5']:
#             ix = a  + "_" + b
#             s_limit=3
#             i_limit=1
#             merge(huc6,ix,s_limit,i_limit)
#             seconds()

for huc6 in ['020301',]: #   '020200', '020301', 
    for a in ['2']: ##, '2']:
        for b in ['5']:
            ix = a  + "_" + b
            s_limit=3
            i_limit=0
            merge(huc6,ix,s_limit,i_limit)
            seconds()



