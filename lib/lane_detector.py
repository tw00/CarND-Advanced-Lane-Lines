import numpy as np
import cv2

class LaneDetector:
    def __init__(self):
        # window settings
        self.window_width = 10 
        self.window_height = 80 # Break image into 9 vertical layers since image height is 720
        self.margin = 40 # How much to slide left and right for searching

    def window_mask(self, img_ref, center, level):
        width  = self.window_width
        height = self.window_height
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output
    
    def find_window_centroids(self, warped):
        window_width = self.window_width;
        window_height = self.window_height;      
        margin = self.margin;              
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    def draw_overlay_image(self, binary_warp, image_warp, window_centroids):
        
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(binary_warp)
            r_points = np.zeros_like(binary_warp)

            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(binary_warp,window_centroids[level][0],level)
                r_mask = self.window_mask(binary_warp,window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) )] = 255
                r_points[(r_points == 255) | ((r_mask == 1) )] = 255

            # Draw the results
            template     = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channle 
            template     = np.array(cv2.merge((zero_channel,template,zero_channel)), np.uint8) # make window pixels green
            warpage      = np.array(cv2.merge((binary_warp,binary_warp,binary_warp)), np.uint8) # making the original road pixels 3 color channels
            output1      = cv2.addWeighted(warpage*255, 1, template, 0.5, 0.0) # overlay the orignal road image with window results        
            output2      = cv2.addWeighted(image_warp, 1, template, 0.5, 0.0) # overlay the orignal road image with window results    

        # If no window centers found, just display orginal road image
        else:
            output1 = np.array(cv2.merge((binary_warped, binary_warped, binary_warped)),np.uint8)
            output2 = output1  
        return output1, output2    
    
    def pipeline(self, cam, imgfilter, image):
        img_width = 120
        image = cam.undistort_img(image)
        image_warp, src, dst, M, Minv = cam.warp_birdeye(image)
        image_warp_org = image_warp.copy()
        image_warp = image_warp[:,:img_width,:];
        delme, binary_img = imgfilter.pipeline(image)
        binary_img = np.array(np.dstack((binary_img, binary_img, binary_img)), np.uint8)
        binary_warp, src, dst, M, Minv = cam.warp_birdeye(binary_img)
        binary_warp = binary_warp[:,:img_width,0];
        
        window_centroids = self.find_window_centroids(binary_warp);
        output_binary, output_image = self.draw_overlay_image(binary_warp, image_warp, window_centroids)
        return (output_binary, output_image, binary_warp, image_warp), window_centroids
    
    def draw_lanes_warped(self, lanes, image):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(image).astype(np.uint8)
        
        ploty = (self.window_height/2 + np.linspace(0, 630, num=720/self.window_height)) / 1
        (left_fitx, right_fitx) = lanes.get_plot(ploty);

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left  = np.array([np.transpose(np.vstack([left_fitx, color_warp.shape[0]-ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, color_warp.shape[0]-ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
#        if True:
#            result0 = cv2.addWeighted(image_warp_org, 1, color_warp, 0.3, 0)
#            plt.imshow(result0)
#            plt.show()

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)  
        return result
