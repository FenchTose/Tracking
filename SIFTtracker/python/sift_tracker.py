import cv2
import cv2.cv as cv
from scipy.spatial import distance as Dis
from sklearn.cluster import DBSCAN
import numpy as np
import itertools
import sys
import math

class SIFTTracker:
    def __init__(self, source=0, bb=None):
        self.mouse_p1 = None
        self.mouse_p2 = None
        self.mouse_drag = False
        self.bb = None
        self.img = None
        self.source = source
        self.detector = cv2.FeatureDetector_create(sys.argv[1])
        self.descriptor = cv2.DescriptorExtractor_create(sys.argv[1])
        self.flann_params = dict(algorithm=1, trees=4)
        self.kalman = None
        self.predict_pt = (0, 0)
        self.state_pt = (0, 0)
        self._kalman()
        self.createClassifierFlag = False
        self.pos = []
        self.frame_no=0
        self.theta = None
        
        if source:
            self.cam = cv2.VideoCapture(source)
        else:
            self.cam = cv2.VideoCapture(0)
        if not bb:
            _, self.img = self.cam.read()
            self.start()
        else:
            self.bb = bb
            _, self.img = self.cam.read()
            self.SIFT()

    def start(self):
        _, self.img = self.cam.read()
        cv2.imshow("img", self.img)
        cv.SetMouseCallback("img", self.__mouseHandler, None)
        if not self.bb:
            _, self.img = self.cam.read()
            cv2.imshow("img", self.img)
            cv2.waitKey(30)
        cv2.waitKey(0)

    def __mouseHandler(self, event, x, y, flags, params):
        _, self.img = self.cam.read()
        if event == cv.CV_EVENT_LBUTTONDOWN and not self.mouse_drag:
            self.mouse_p1 = (x, y)
            self.mouse_drag = True
        elif event == cv.CV_EVENT_MOUSEMOVE and self.mouse_drag:
            cv2.rectangle(self.img, self.mouse_p1, (x, y), (255, 0, 0), 1, 8, 0)
        elif event == cv.CV_EVENT_LBUTTONUP and self.mouse_drag:
            self.mouse_p2 = (x, y)
            self.mouse_drag=False
        cv2.imshow("img",self.img)
        cv2.waitKey(01)
        if self.mouse_p1 and self.mouse_p2:
            cv2.destroyWindow("img")
            xmax = max((self.mouse_p1[0],self.mouse_p2[0]))
            xmin = min((self.mouse_p1[0],self.mouse_p2[0]))
            ymax = max((self.mouse_p1[1],self.mouse_p2[1]))
            ymin = min((self.mouse_p1[1],self.mouse_p2[1]))
            self.bb = [xmin,ymin,xmax-xmin,ymax-ymin]
            self.SIFT()
            #cv2.destroyAllWindows()
            return None

    def SIFT(self):
        #temp = self.img[150:400,200:400]
        old_pts=None
        bb = self.bb
        self.template = self.img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        self.tkp, self.td = self.getKeyPoints(self.template)
        self.tkp_total, self.td_total = self.getKeyPoints(self.img)
        sub = []
        for i in range(len(self.tkp_total)):
            if (int(self.tkp_total[i].pt[0]) >= self.bb[0] and 
                int(self.tkp_total[i].pt[0]) < self.bb[0] + self.bb[2] and
                int(self.tkp_total[i].pt[1]) >= self.bb[1] and
                int(self.tkp_total[i].pt[1]) < self.bb[1] + self.bb[3]):
                sub.append(1)
            else:
                sub.append(0)
        self.createClassifier(self.td_total, sub)

        for i in range(len(self.tkp)):
            cv2.circle(self.template, (int(self.tkp[i].pt[0]), int(self.tkp[i].pt[1])), 2, (0, 255, 255), -1)
        cv2.imshow("Template", self.template)

        while True:
            _, img = self.cam.read()
            self.img = img
            self.kmeans_img = np.copy(img)
            skp, tkp = self.matchKeyPoints(img, self.template)
            print len(skp), "keypoints being matched"
            if len(skp) < 5:
                cv2.imshow("img",img)
                k=cv2.waitKey(30)
                self.img = img
                if (k==27):
                    break
                continue
            bb, new_center = self.__predictBB(bb, old_pts, skp)
            print new_center[0]
            self.bb = bb
            cv2.circle(img, (int(new_center[0][0]), int(new_center[0][1])), 5, (0,0,255), -1)
            cv2.rectangle(img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 0), 2)
            cv2.circle(img, (int(self.state_pt[0]), int(self.state_pt[1])), 5, (0, 255, 255), -1)
            cv2.circle(img, (int(self.predict_pt[0]), int(self.predict_pt[1])), 5, (255, 0, 255), -1)
            cv2.imshow("img",img)
            k=cv2.waitKey(30)
            old_pts = skp
            if (k==27):
                break
        cv2.destroyAllWindows()
        return None

    def __predictBB(self, bb, old_pts, new_pts):
        pts = []
        for kp in new_pts:
            pts.append((kp.pt[0], kp.pt[1]))
            cv2.circle(self.kmeans_img, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 255), -1)

        np_pts = np.asarray(pts)
        t, pts, new_center = cv2.kmeans(np.asarray(np_pts, dtype=np.float32), K=1, bestLabels=None,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), attempts=1, 
                            flags=cv2.KMEANS_RANDOM_CENTERS)
        print new_center

        cv2.circle(self.kmeans_img, (new_center[0][0], new_center[0][1]), 8, (0, 0, 255), 2)
        
        max_x = int(max(np_pts[:, 0]))
        min_x = int(min(np_pts[:, 0]))
        max_y = int(max(np_pts[:, 1]))
        min_y = int(min(np_pts[:, 1]))
        
        rad = ((new_center[0][0]-max_x)**2+(new_center[0][1]-max_y)**2)**0.5
        self.pos.append((new_center[0][0], new_center[0][1]))
        cv2.circle(self.kmeans_img, (new_center[0][0], new_center[0][1]), int(rad) , (0, 0, 255), 2)
        cv2.imshow("K-Means", self.kmeans_img)
        new_bb =  (min_x-5, min_y-5, max_x-min_x+5, max_y-min_y+5)
        print new_bb, new_center
        #new_center[0][0] += new_bb[0]
        #new_center[0][1] += new_bb[1]
        self._setKalman(new_center[0][0], new_center[0][1], self.predict_pt[0], self.predict_pt[1])
        self._predictKalman()
        self._changeMeasure(new_center[0][0], new_center[0][1])
        self._correctKalman()
        print self.state_pt, self.predict_pt
        
        return new_bb, new_center

    def getKeyPoints(self, img):
        detector = self.detector
        descriptor = self.descriptor
        skp = detector.detect(img)
        skp, sd = descriptor.compute(img, skp)
        return skp, sd


    def matchKeyPoints(self, img, template, distance=40):
        self.frame_no+=1
        detector = self.detector
        descriptor = self.descriptor
        dimg = np.copy(img)
        #dtemp = np.copy(template)
        skp = detector.detect(img)
        skp, sd = descriptor.compute(img, skp)

        #tkp = detector.detect(template)
        #tkp, td = descriptor.compute(template, tkp)
        tkp = self.tkp
        td = self.td
        print len(tkp), "tkp"
        print len(sd), "sd"
        flann = cv2.flann_Index(sd, self.flann_params)
        idx, dist = flann.knnSearch(td, 1, params={})
        del flann

        dist = dist[:,0]/2500.0
        dist = dist.reshape(-1,).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        print dist, "dist"
        print idx, "idx", len(idx)
        skp_final = []
        sd_final = []
        skp_final_labelled=[]
        sd_final_labelled=[]
        skp_final_labelled_b=[]
        sd_final_labelled_b=[]
        pos=[]
        data_cluster=[]
        
        for i, dis in itertools.izip(idx, dist):
            if dis < distance:
                skp_final.append(skp[i])
                sd_final.append(sd[i])
                data_cluster.append((skp[i].pt[0], skp[i].pt[1]))
        n_data = np.asarray(data_cluster)
        D = Dis.squareform(Dis.pdist(n_data))
        S = 1 - (D/np.max(D))
       
        eps_val = 0.7
        
        db = DBSCAN(eps=eps_val, min_samples=5).fit(S)
        core_samples = db.core_sample_indices_
        labels = db.labels_
        print len(set(labels))
        print set(labels)
        print len(labels)
        print len(skp_final)
        for label, i in zip(labels, range(len(labels))):
            if label==0:
                cv2.circle(dimg, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (255, 0, 0), -1)
                skp_final_labelled.append(skp_final[i])
                pos.append(1)
            else:
                if self.frame_no < 2:
                    if (int(data_cluster[i][0]) >= self.bb[0] and 
                        int(data_cluster[i][0]) < self.bb[0] + self.bb[2] and
                        int(data_cluster[i][1]) >= self.bb[1] and
                        int(data_cluster[i][1]) < self.bb[1] + self.bb[3]):

                        cv2.circle(dimg, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (255, 0, 0), -1)
                        skp_final_labelled.append(skp_final[i])
                        pos.append(1)
                    else:
                        cv2.circle(dimg, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (255, 0, 255), -1)
                        pos.append(0)
                else:
                    cv2.circle(dimg, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (255, 0, 255), -1)
                    pos.append(0)

        cv2.imshow("DBSCAN", dimg)
        print len(skp_final), len(skp_final_labelled), len(skp)

        """
        flann = cv2.flann_Index(td, self.flann_params)
        idx, dist = flann.knnSearch(sd, 1, params={})
        del flann
        
        
        dist = dist[:,0]/2500.0
        dist = dist.reshape(-1,).tolist()
        idx = idx.reshape(-1).tolist()
        indices = range(len(dist))
        indices.sort(key=lambda i: dist[i])
        dist = [dist[i] for i in indices]
        idx = [idx[i] for i in indices]
        data_cluster=[]
        tkp_final = []
        tkp_final_labelled = []
        for i, dis in itertools.izip(idx, dist):
            if dis < distance:
                tkp_final.append(tkp[i])
                data_cluster.append((tkp[i].pt[0], tkp[i].pt[1]))
        n_data = np.asarray(data_cluster)
        D = Dis.squareform(Dis.pdist(n_data))
        S = 1 - (D/np.max(D))

        db = DBSCAN(eps=2.4, min_samples=10).fit(S)
        core_samples = db.core_sample_indices_
        labels = db.labels_
        print len(set(labels))
        print set(labels)
        print len(labels)
        print len(tkp_final)
        for label, i in zip(labels, range(len(labels))):
            #print data_cluster[i]
            if label==0:
                cv2.circle(dtemp, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (255, 0, 0), -1)
                tkp_final_labelled.append(tkp_final[i])
            else:
                cv2.circle(dtemp, (int(data_cluster[i][0]), int(data_cluster[i][1])), 3, (0, 0, 255), -1)
        cv2.imshow("DBSCAN-temp", dtemp)
        #cv2.waitKey(0)
        """

        tkp_final_labelled = []
        pos = self.classify(sd)
        #return skp_final, []
        """
        if self.frame_no < 30:
            patches = self.getPatchVals(skp_final)
            self.createClassifier(patches, pos)
            #self.createClassifier(sd_final, pos)
        else:
            patches = self.getPatchVals(skp_final)
            pos = self.classify(patches)
            #print pos.shape
            
            ave = np.average(pos)
            skp_final_labelled_c = []
            for i in range(pos.shape[0]):
                if pos[i] >= ave:
                    skp_final_labelled_c.append(skp_final[i])
            return skp_final_labelled_c, tkp_final_labelled
        """
        #return skp_final, tkp_final_labelled
        return skp_final_labelled, tkp_final_labelled

    def _kalman(self):
        self.kalman = cv.CreateKalman(4, 2, 0)
        self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)  # (phi, delta_phi)
        self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)

        self.kalman.transition_matrix[0,0] = 1
        self.kalman.transition_matrix[0,1] = 0
        self.kalman.transition_matrix[0,2] = 1
        self.kalman.transition_matrix[0,3] = 0
        self.kalman.transition_matrix[1,0] = 0
        self.kalman.transition_matrix[1,1] = 1
        self.kalman.transition_matrix[1,2] = 0
        self.kalman.transition_matrix[1,3] = 1
        self.kalman.transition_matrix[2,0] = 0
        self.kalman.transition_matrix[2,1] = 0
        self.kalman.transition_matrix[2,2] = 1
        self.kalman.transition_matrix[2,3] = 0
        self.kalman.transition_matrix[3,0] = 0
        self.kalman.transition_matrix[3,1] = 0
        self.kalman.transition_matrix[3,2] = 0
        self.kalman.transition_matrix[3,3] = 1

        cv.SetIdentity(self.kalman.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(1e-5))
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(1))

    def _setKalman(self, x, y, predict_x, predict_y):
        self.kalman.state_pre[0,0]  = x
        self.kalman.state_pre[1,0]  = y
        self.kalman.state_pre[2,0]  = predict_x
        self.kalman.state_pre[3,0]  = predict_y

    def _predictKalman(self):
        self.kalman_prediction = cv.KalmanPredict(self.kalman)
        self.predict_pt  = (self.kalman_prediction[0,0], self.kalman_prediction[1,0])

    def _correctKalman(self):
        self.kalman_estimated = cv.KalmanCorrect(self.kalman, self.kalman_measurement)
        self.state_pt = (self.kalman_estimated[0,0], self.kalman_estimated[1,0])

    def _changeMeasure(self, x, y):
        self.kalman_measurement[0, 0] = x
        self.kalman_measurement[1, 0] = y

    def createClassifier(self, descriptors, pos):
        features_len = descriptors.shape[1]+1
        print descriptors.shape
        self.createClassifierFlag=True
        print "createClassifier"
        pos = np.asarray(pos).reshape(len(pos),1)
        if isinstance(self.theta, type(None)):
            self.theta = np.zeros((features_len,1))
        dataset_size = len(descriptors)
        h = np.zeros((dataset_size,1))
        cost = np.zeros((dataset_size,1))
        grad = np.zeros((features_len,1))
        features = np.zeros((dataset_size, features_len))
        iterations = 400
        learning_rate = 0.01
        for i in xrange(dataset_size):
            features[i] = np.append(np.ones((1,1)), descriptors[i].reshape(descriptors[i].shape[0],1)/255.0)
        print "features created"
        for i in xrange(iterations):
            for j in xrange(dataset_size):
                #print features[j]
                h[j] = 1.0/(1+math.exp(-np.dot(features[j],self.theta)))
                #print theta
                #print features[j]
                #print h[j], "h"
                
            
            for k in xrange(features_len):
                grad[k] = np.sum((h-pos)*(features[:,k]))/dataset_size

            cost = -pos * np.log(h) - (1-pos) *np.log(1-h)
            #print cost, "cost"
                
            self.theta = self.theta - learning_rate*grad
            
        print self.theta

        for j in xrange(dataset_size):
            print "some value"
            print pos[j]
            print self.theta, features[j]
            print 1.0/(1+math.exp(-np.dot(features[j],self.theta)))
        self.thresh = np.average(h)*(np.sum(pos))/pos.shape[0]
        print self.thresh

    def classify(self, descriptors):
        features_len = descriptors.shape[1]+1
        print "classify"
        print len(descriptors), "points to be classified"
        dataset_size = len(descriptors)
        features = np.zeros((dataset_size, features_len))
        for i in xrange(dataset_size):
            features[i] = np.append(np.ones((1,1)), descriptors[i].reshape(descriptors[i].shape[0],1)/255.0)
        h = np.zeros((dataset_size,1))
        for j in range(dataset_size):
            h[j] = 1.0/(1+math.exp(-np.dot(features[j],self.theta)))
        print h
        return h

    def getPatchVals(self, skp):
        patches = np.zeros((len(skp), 1))
        gray = cv2.cvtColor(self.img, cv2.cv.CV_BGR2GRAY)
        for i in xrange(len(skp)):
            patches[i] = np.sum(gray[skp[i].pt[0]-8:skp[i].pt[0]+8, skp[i].pt[1]-8:skp[i].pt[1]+8])/64.0
        print patches.shape
        return patches





#s=SIFTTracker(source="/home/jay/Python/SimpleCV/vid3.mp4")
#s=SIFTTracker(source="test.avi")
s=SIFTTracker(-1)
