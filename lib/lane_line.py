import numpy as np
import scipy
import scipy.signal

# Define a class to receive the characteristics of each line detection
class LaneLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #latest lane fit
        self.last_fit = None
        # was the last detection plausible?
        self.last_plausible = False;

    def fit_next(self, y, x, conf):
        self.ally = y
        self.allx = x
        self.last_fit = self.current_fit;

        # Fit a second order polynomial to pixel positions in each lane line
        if True:
            idx = self.idx_reject_outliers(self.allx, conf);
            # make sure at least 3 data points are present for fitting
            if np.sum( idx ) >= 3:
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
            if err > 10 or np.sum(self.current_fit) == 0:
                self.last_plausible = False;
            else:
                self.best_fit = self.best_fit - 0.25 * (self.best_fit - self.current_fit);
                self.last_plausible = True;
        else:
            self.best_fit = self.current_fit
            self.detected = True;
        return idx

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

    def idx_reject_outliers(self, data, conf, m = 3.0, low = 5):
        d = np.abs(data - np.median(data[data > low]))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.*d
        return np.logical_and( np.logical_and(s < m, data > low ), conf > 0.1 );

    def generate_road(self, offset, window_height):
        ploty = offset + np.linspace(0, window_height, num=200)
        fitx  = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return (ploty, fitx)
 
    def get_radius(self):
        #radius of curvature of the line in some units
        self.fit_cr = self.best_fit;
        y_eval = np.max(self.ally)
        factor_x     = 50 # px/ft
        factor_y     = 5 # px/ft
        ft_to_m      = 0.3048 # m/ft
        ym_per_pix   = (1/factor_y) * ft_to_m # ft/px * m/ft => m/px
        xm_per_pix   = (1/factor_x) * ft_to_m # ft/px * m/ft => m/px
        curverad     = ((1 + (2*self.fit_cr[0]*y_eval*ym_per_pix + self.fit_cr[1])**2)**1.5) / np.absolute(2*self.fit_cr[0])
        return curverad

    def get_vehicle_offset(self):
        self.fit_cr = self.best_fit; 
        factor_x   = 50 # px/ft
        ft_to_m    = 0.3048 # m/ft
        xm_per_pix = (1/factor_x) * ft_to_m # ft/px * m/ft => m/px
        center_px  = 450
        return (center_px - self.fit_cr[2])*xm_per_pix
