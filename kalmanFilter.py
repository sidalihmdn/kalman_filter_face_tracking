import numpy as np

class KalmanFilter:
    def __init__(self , dt , point) -> None:
        '''
        this is the initialisation methode of this class
        it's needs a starting point and a time variable
        '''
        self.dt = dt

        # stat vector
        self.E = np.matrix([[point[0]], [point[1]], [0], [0]])

        # dynamic matrix
        self.A = np.matrix([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        # Observation matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        #
        self.Q = np.eye(4,4)
        self.R = np.eye(2,2)
        self.P = np.eye(self.A.shape[1])
    

    def predict(self):
        '''
        this function is used to predict the futur state of the systeme based on it
        current state
        '''
        self.E = np.dot(self.A , self.E)
        # covarience
        self.P = np.dot(np.dot(self.A , self.P), self.A.T) + self.Q
        return self.E

    def update(self, z):
        '''
        this function is used the correct the curent state of the system using the 
        measurment z and the current prediction
        '''
        # Kalman gain
        S = np.dot(self.H, np.dot(self.P , self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # correction
        self.E = np.round(self.E + np.dot(K, (z-np.dot(self.H, self.E))))
        # identity matrix
        I = np.eye(self.H.shape[1])
        self.P = (I-(K*self.H))*self.P

        return self.E


