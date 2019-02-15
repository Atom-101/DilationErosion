import glob
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def get_args():
   argparser = ArgumentParser(description="Erode and dilate masks in a folder by a specified amount."
                                            "Can also overlay processed mask on an original image")
   argparser.add_argument("-e","--erosion_filter_size",help="Size of erosion filter in format w,h",required=True)
   argparser.add_argument("-I","--erosion_filter_iters",help="Iterations of erosion filter",required=False,default=1)
   argparser.add_argument("-d","--dilation_filter_size",help="Size of dilation filter",required=True)
   argparser.add_argument("-i","--dilation_filter_iters",help="Iterations of dilation filter in format w,h",required=False,default=1)
   argparser.add_argument("-p","--images_path",help="Path to folder containing masks",required=True)
   argparser.add_argument("-m","--mode",help="Dilation erosion mode",required=False,default='other')

   argparser.add_argument("-s","--save_path",help="Path to folder where processed masks will be saved",required=True)
   argparser.add_argument("-o","--original_path",help="Path to original image",required=False)

   args = argparser.parse_args()
   return args

def erode_dilate(addresses,args,test=False,mode='other'):
    for addr in tqdm(addresses):
        # print(addr)
        mask = cv2.imread(addr,cv2.IMREAD_GRAYSCALE)
        # print(mask.shape)
        if(mode=='other'):    
            erosion_filter = np.ones((int(args.erosion_filter_size.split(',')[0]),
                                    int(args.erosion_filter_size.split(',')[1])),np.uint8)
            dilation_filter = np.ones((int(args.dilation_filter_size.split(',')[0]),
                                    int(args.dilation_filter_size.split(',')[1])),np.uint8)
            #Preliminary closing
            mask =  cv2.dilate(mask,dilation_filter,iterations=1)
            mask = cv2.erode(mask,erosion_filter,iterations=1)
            #Opening            
            mask = cv2.erode(mask,erosion_filter,iterations=int(args.erosion_filter_iters))
            mask =  cv2.dilate(mask,dilation_filter,iterations=int(args.dilation_filter_iters))
            #Compensatory erosion in case dilation was heavier
            if(int(args.dilation_filter_iters)>1):
                mask = cv2.erode(mask,erosion_filter,iterations=int(args.erosion_filter_iters)-
                                    int(args.dilation_filter_iters))
        elif(mode=='road'):
            dilation_w,dilation_h =int(args.dilation_filter_size.split(',')[0]),int(args.dilation_filter_size.split(',')[1])
            erosion_w,erosion_h = int(args.erosion_filter_size.split(',')[0]),int(args.erosion_filter_size.split(',')[1])
            iters = max(int(args.erosion_filter_iters),int(args.dilation_filter_iters))
            for _ in range(iters):
                #Vertical closing
                mask = cv2.dilate(mask,np.ones((dilation_w,dilation_h*2),dtype=np.uint8))
                mask = cv2.erode(mask,np.ones((erosion_w,erosion_h),dtype=np.uint8))
                #Horizontal closing
                mask = cv2.dilate(mask,np.ones((dilation_w*2,dilation_h),dtype=np.uint8))
                mask = cv2.erode(mask,np.ones((erosion_w,erosion_h),dtype=np.uint8))
                #Compensatory erode
                mask = cv2.erode(mask,np.ones((erosion_w,erosion_h),dtype=np.uint8))
        
        if(test):
            return mask
        
        cv2.imwrite(args.save_path+addr.split('/')[-1],mask)

def overlay(img,mask):
    # print(img.shape)
    # print(np.dstack((mask,mask,mask)).shape)
    overlayed = cv2.addWeighted(img,0.6,np.dstack((mask,mask,mask)),0.4,0.2)
    cv2.imshow('Overlayed',overlayed)
    cv2.imwrite('Overlayed.jpg',overlayed)
    cv2.imwrite('Mask.jpg',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()
    if(args.original_path):
        addresses = [args.images_path+args.original_path.split('/')[-1]]
        mask = erode_dilate(addresses,args,test=True,mode=args.mode.lower())
        ###########################
        #Wont be required in final#
        ###########################
        mask = cv2.resize(mask,(4800,3705),interpolation=cv2.INTER_AREA)
        overlay(cv2.imread(args.original_path),mask)
    else:
        addresses = glob.glob(args.images_path+'*')
        erode_dilate(addresses,args)