import numpy as np
import cv2
from .lane_line import LaneLine

class LaneDetector:
    def __init__(self):
        # window settings
        self.window_height      = 120 # Break image into 9 vertical layers since image height is 720
        self.window_width       = 80
        self.margin             = 80 # How much to slide left and right for searching
        self.lane_left           = LaneLine();
        self.lane_right          = LaneLine();

    def window_mask(self, img_ref, center, level):
        width  = self.window_width
        height = self.window_height
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output
    
    def find_window_centroids(self, warped):
        window_width  = self.window_width;
        window_height = self.window_height;
        margin        = self.margin;
        
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        l_confidence = 1;
        r_confidence = 1;

        # Add what we found for the first layer
        window_centroids.append((window_height/2, l_center, r_center, l_confidence, r_confidence))

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
            l_confidence = np.sum(conv_signal[l_min_index:l_max_index]/np.sum(conv_signal))

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            r_confidence = np.sum(conv_signal[r_min_index:r_max_index]/np.sum(conv_signal))

            # Add what we found for that layer
            window_centroids.append((window_height/2 + window_height*level, l_center, r_center, l_confidence, r_confidence))

        return window_centroids

    def draw_overlay_image(self, binary_warp, image_warp, window_centroids, outlier_l, outlier_r):

        #assert binary_warp.shape[2] == 1, "binary_warp not binary"
        
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(binary_warp)
            r_points = np.zeros_like(binary_warp)

            # Go through each level and draw the windows    
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(binary_warp,window_centroids[level][1],level)
                r_mask = self.window_mask(binary_warp,window_centroids[level][2],level)
                # Add graphic points from window mask here to total pixels found 
                l_points[(l_points == 255) | ((l_mask == 1) )] = 50 + 200*outlier_l[level];
                r_points[(r_points == 255) | ((r_mask == 1) )] = 50 + 200*outlier_r[level];

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

    def draw_lanes(self, image, returnOverlayOnly = False):
        overlay = np.zeros_like(image).astype(np.uint8)
        
        (ploty, left_fitx)  = self.lane_left.generate_road(10, 650);
        (ploty, right_fitx) = self.lane_right.generate_road(10, 650);
        offset_x = 1.5; # line width in px
        left_fitx += offset_x;
        right_fitx += offset_x;

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left  = np.array([np.transpose(np.vstack([left_fitx, image.shape[0]-ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, image.shape[0]-ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, np.int_([pts]), (0, 255, 0))

        # Combine the result with the original image
        if returnOverlayOnly:
            return overlay;
        else:
            result = cv2.addWeighted(image, 1, overlay, 0.5, 0)  
            return result

    def draw_lanes_warped(self, cam, image):
        color_warp = self.draw_lanes(image, returnOverlayOnly=True)

        # Draw original points:
        for k in range(2):
            if k == 0: lane = self.lane_left;
            if k == 1: lane = self.lane_right;
            for i in range(len(lane.ally)):
                x0 = int(lane.allx[i])
                y0 = color_warp.shape[0]-int(lane.ally[i])
                cv2.circle(color_warp, (x0,y0), int(2+i*1.2), (0,0,255), -1)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cam.unwarp_birdeye(color_warp);

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)

        # Print text
        r1 = self.lane_left.get_radius()
        r2 = self.lane_right.get_radius()
        d1 = self.lane_left.get_vehicle_offset()
        d2 = self.lane_right.get_vehicle_offset()
        p1 = self.lane_left.last_plausible;
        p2 = self.lane_right.last_plausible;
        r = np.mean([r1,r2]);
        d = np.sum([d1,d2]);
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Curve radius: %.0fm' % (r), (20,50), font, 1.5, (255,255,255), 3)
        cv2.putText(result, 'Vehicle is %.3fm off center' % (d), (20,100), font, 1.5, (255,255,255), 3)
        cv2.putText(result, '%d|%d' % (p1,p2), (20,150), font, 1.5, (255,255,255), 3)
        return result

    def mask_image(self, img):
        img = img.copy();
        mask_right = np.array([(850,img.shape[0]),(img.shape[1],0),(img.shape[1],img.shape[0])])
        mask_left = np.array([(150,img.shape[0]),(0,0),(0,img.shape[0])])
        img = cv2.fillConvexPoly(img, mask_right, 0)
        img = cv2.fillConvexPoly(img, mask_left, 0)
        return img

    def pipeline(self, cam, imgfilter, img, includeDebugImages = True):

        # 1. undisdort
        img = img.copy();
        img_undist = cam.undistort_img(img);

        # 2. calculate gradients
        _, img_gradient = imgfilter.gradient_filter(img_undist)
        img_gradient    = np.array(np.dstack((img_gradient, img_gradient, img_gradient)), np.uint8)

        # 3. warp to birdeye view
        img_gradient_warp = cam.warp_birdeye(img_gradient)
        img_warp          = cam.warp_birdeye(img_undist)

        # 3b. mask image
        img_gradient_warp = self.mask_image(img_gradient_warp);

        # 4. find centroids
        window_centroids = self.find_window_centroids(img_gradient_warp[:,:,0]);
        window_centroids = np.array(window_centroids);
        self.previous_centroids = window_centroids;

        # 5. fit lanes
        outlier_l= self.lane_left.fit_next(window_centroids[:,0], window_centroids[:,1], window_centroids[:,3]);
        outlier_r = self.lane_right.fit_next(window_centroids[:,0], window_centroids[:,2], window_centroids[:,4]);

        # Draw overlay images
        if includeDebugImages:
            img_gradient_overlay, img_overlay = self.draw_overlay_image(img_gradient_warp[:,:,0], img_warp, window_centroids, outlier_l, outlier_r)
            img_overlay_fit     = self.draw_lanes(img_warp);
            img_warp_with_lanes = self.draw_lanes_warped(cam, img_undist);

            return {'gradient':      img_gradient_overlay,
                    'image':         img_overlay,
                    'gradient_warp': img_gradient_warp[:,:,0],
                    'image_warp':    img_warp,
                    'image_fit':     img_overlay_fit,
                    'final':         img_warp_with_lanes}
        else:
            img_warp_with_lanes = self.draw_lanes_warped(cam, img_undist);
            return {'final':         img_warp_with_lanes}
