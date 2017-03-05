# TODO: Zeitabhängigkeit
# TODO: Real world coordinates
# TODO: print offset from center
# TODO: print radius
import numpy as np

# Define a class to receive the characteristics of each line detection
class LaneLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #
        self.left_fit = None
        self.right_fit = None

    def fit(self, window_centroids):
        data   = np.array(window_centroids)
        ploty  = data[:,0]
        leftx  = data[:,1]
        rightx = data[:,2]
    
        # Fit a second order polynomial to pixel positions in each fake lane line
        self.left_fit   = np.polyfit(ploty, leftx, 2)
        self.right_fit  = np.polyfit(ploty, rightx, 2)
        
    def generate_road(self, offset, window_height):
        offset_x   = 30; # todo: in camera
        offset_y   = -5
        offset_x   = 0;
        offset_y   = 0;
        ploty      = offset + np.linspace(0, window_height, num=200)
        left_fitx  = self.left_fit[0]*ploty**2  + self.left_fit[1]*ploty  + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        return (ploty-offset_y, left_fitx-offset_x, right_fitx-offset_x)
        
#        left_fitx  = self.left_fit[0]*ploty**2  + self.left_fit[1]*ploty  + self.left_fit[2]
#        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
#        return (left_fitx, right_fitx)
    
#    def fit_world(self, ploty, leftx, rightx): # Das stimmt was nicht ploty länge ist nicht leftx länge
#        # Define conversions in x and y from pixels space to meters
#        ym_per_pix = 30/720 # meters per pixel in y dimension
#        xm_per_pix = 3.7/700 # meters per pixel in x dimension
#
#        # Fit new polynomials to x,y in world space
#        left_fit_cr  = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
#        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
 
    def get_radius(self):
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m        

    def get_next(self):
        return "TODO"
