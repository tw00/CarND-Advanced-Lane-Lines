# xTODO: Zeitabhängigkeit
# xTODO: print offset from center
# xTODO: print radius
# xTODO: LaneLiNe = Single Lane Line
# TODO: Real world coordinates
# TODO: Sync with other lane line (confidence)
import numpy as np
import scipy
import scipy.signal

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
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #
        self.last_fit = None
        self.last_plausible = False;

    def fit_next(self, y, x):
        self.ally = y
        self.allx = x
        self.last_fit = self.current_fit;

        # Fit a second order polynomial to pixel positions in each fake lane line
        if True:
            idx = self.idx_reject_outliers(self.allx);
            if np.sum( idx ) > 3:
                self.current_fit = np.polyfit(self.ally[idx], self.allx[idx], 2)
            else:
                self.current_fit = np.array([0.,0.,0.]);
        else:
            self.current_fit = self.fit_poly_RANSAC(self.ally, self.allx);

        if self.detected:
            self.diffs = self.current_fit - self.last_fit;
            ploty = np.array([50, 150, 250, 350, 450, 550])
            fit1  = self.last_fit[0]*ploty**2 + self.last_fit[1]*ploty + self.last_fit[2]
            fit2  = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
            err   = np.sqrt(np.mean((fit1-fit2)**2))
#            print(err, "px")
#            if any( np.abs(self.diffs) > 90.50*np.abs(self.last_fit) ):
            #if np.abs(self.diffs[2]) > 4 or np.sum(self.current_fit) == 0:
            if err > 10 or np.sum(self.current_fit) == 0:
                # Found implausible fit (more than 20% deviation)
                # TODO: Später einschalten
                self.last_plausible = False;
#                self.current_fit = self.best_fit;
            else:
                self.last_plausible = True;
                self.best_fit = self.best_fit - 0.05 * (self.best_fit - self.current_fit);
        else:
            self.best_fit = self.current_fit
            self.detected = True;
        

    def fit_poly_RANSAC(self, y, x):
        best_mse    = 1e99;
        best_result = None;
        best_choice = None;
        for i in range(100):
            r = np.ones(shape=(len(x),), dtype=np.bool)
            r[np.random.randint(0,len(x))] = False;
            r[np.random.randint(0,len(x))] = False;
            result = np.polyfit(y[r], x[r], 2, full=True)
            mse = result[1];
            mse = 0 if mse.size == 0 else mse[0]
            if mse < best_mse:
                best_mse    = mse;
                best_result = result;
                best_choice = r;
        return best_result[0]

    def idx_reject_outliers(self, data, m = 2., low = 5):
        # data = scipy.signal.detrend(data);
        d = np.abs(data - np.median(data[data > low]))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.*d
        return np.logical_and(s < m, data > low )

    def generate_road(self, offset, window_height):
        ploty = offset + np.linspace(0, window_height, num=200)
        #fitx  = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        fitx  = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return (ploty, fitx)
        
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
        #radius of curvature of the line in some units
        #self.radius_of_curvature = 0 
        # Calculate the new radii of curvature
#        self.fit_cr = self.current_fit; # HACK
        self.fit_cr = self.best_fit; # HACK
        y_eval = np.max(self.ally)
        ym_per_pix = 5/100 # meters per pixel in y dimension
        xm_per_pix = 5/100 # meters per pixel in x dimension
        curverad = ((1 + (2*self.fit_cr[0]*y_eval*ym_per_pix + self.fit_cr[1])**2)**1.5) / np.absolute(2*self.fit_cr[0])
        return curverad # TODO: DAS SIND FEET

    def get_vehicle_offset(self):
#        self.fit_cr = self.current_fit; # HACK
        self.fit_cr = self.best_fit; # HACK
        xm_per_pix = 5/100 # meters per pixel in x dimension
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 
        center_px = 59
        return (center_px - self.fit_cr[2])*xm_per_pix # TODO: Das ist feet

        #left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        #right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m        

    def get_next(self):
        return "TODO"
