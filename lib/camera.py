import numpy as np
import cv2
import glob
import pickle

# Define a class to receive the characteristics of each line detection
class Camera():
    def __init__(self):
        # 
        self.mtx    = None
        self.dist   = None        
        self.ret    = None        
        # 
        self.calibration_objpoints = []
        self.calibration_imgpoints = []
        #
        self.M      = None # Matrix image to birdeye
        self.Minv   = None # Matrix birdeye to image
        self.cut_x  = None # Pixels to cut after birdeye transformation
        self.src    = None # source coordinates
        self.dst    = None # destination coordinates
        
    def calibrate(self, img_dir):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(img_dir + '/calibration*.jpg')

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8,6), corners, ret)
        
        self.calibration_objpoints = objpoints;
        self.calibration_imgpoints = imgpoints;    

        # Do camera calibration given object points and image points
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        print("ret:",ret)

        self.ret  = ret;
        self.mtx  = mtx;
        self.dist = dist;
        return self
    
    def save_calibration(self):        
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump( dist_pickle, open( "camera_cal/cam_calibration.p", "wb" ) )
        
    def load_calibration(self):
        cam_data  = pickle.load( open( "camera_cal/cam_calibration.p", "rb" ) )
        self.mtx  = cam_data['mtx']
        self.dist = cam_data['dist']
        
    def undistort_img(self, img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst  
    
    def corners_unwarp(self, img, nx, ny):
        # Define a function that takes an image, number of x and y points, 
        # camera matrix and distortion coefficients
        mtx = self.mtx;
        dist = self.dist;
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # Convert undistorted image to grayscale
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        # Search for corners in the grayscaled image
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            # If we found corners, draw them! (just for fun)
            cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            offset = 100 # offset for dst points
            # Grab the image shape
            img_size = (gray.shape[1], gray.shape[0])

            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                              [img_size[0]-offset, img_size[1]-offset], 
                              [offset, img_size[1]-offset]])
            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # Warp the image using OpenCV warpPerspective()
            warped = cv2.warpPerspective(undist, M, img_size)

        # Return the resulting image and matrix
        return warped, M    

    def warp_birdeye(self, img):

        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])

        if self.M is None:
            # USA Lane width: 3.7 m (12ft)
            # The finding holds implications for traffic safety. Each dashed line measures 10 feet, and the empty spaces in-between measure 30 feet.
            #
            # Choose offset from image corners to plot detected corners
            # This should be chosen to present the result at the proper aspect ratio
            # My choice of 100 pixels is not exact, but close enough for our purpose here
            factor_x   = 50
            factor_y   = 5

            # For source points I'm grabbing the outer four detected corners
            src = np.float32([[308,648],[1000,648],[579,460],[703,460]])

            Lane_W = factor_x*(12)          # ft (lande width)
            Lane_D = factor_y*(30+10+30+10) # ft (lane distance)

            # offset_x = 200  # offset for dst points
            # offset_y = -30
            offset_x = int(Lane_W/4);
            offset_y = -5;
            cut_x    = img_size[0]; # 2*offset_x + Lane_W;

            # For destination points, I'm arbitrarily choosing some points to be
            # a nice fit for displaying our warped result 
            # again, not exact, but close enough for our purposes
            dst = np.float32([[offset_x,        img_size[1]+offset_y],
                              [Lane_W+offset_x, img_size[1]+offset_y], 
                              [offset_x,        img_size[1]-Lane_D+offset_y], 
                              [Lane_W+offset_x, img_size[1]-Lane_D+offset_y]])    

            # Given src and dst points, calculate the perspective transform matrix
            self.src   = src;
            self.dst   = dst;
            self.M     = cv2.getPerspectiveTransform(src, dst)
            self.Minv  = cv2.getPerspectiveTransform(dst, src)    
            self.cut_x = cut_x;

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, self.M, img_size)
        warped = warped[:,0:self.cut_x,:];

        # Return the resulting image and matrix
        return warped

    def unwarp_birdeye(self, img):
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        img_unwarp = cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0])) 
        return img_unwarp
