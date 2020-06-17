import cv2
import numpy as np
import dlib
import time
import argparse
import random
import string

def getFLIndex(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def dnnLandmarks(image,gray,h,w):
    # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300,300)), 1, (300,300), (104, 177, 123))

    detector.setInput(inputBlob)
    faces = detector.forward()

    for i in range(0,faces.shape[2]):

        predicitionScore = faces[0,0,i,2]
        if predicitionScore < args.threshold:
            continue

        boundingBox = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = boundingBox.astype("int")

        y1, x2 = int(y1 * 1.15), int(x2 * 1.05)

        landmarks = predictor(gray,dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
        landmarksPoints = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarksPoints.append((x,y))
        return landmarksPoints



if __name__ == '__main__':
    # Get the arguement pass from the terminal
    start = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-src","--source",required=True,help="path to source face")
    argparser.add_argument("-dest","--destination",required=True,help="path to destination face")
    argparser.add_argument("-p","--predictor",required=True,help="Path to predictor file for dlib")
    argparser.add_argument("-ptxt","--prototxt",required=True,help="Path to deploy Prototxt file for OpenCV DNN")
    argparser.add_argument("-m","--model",required=True,help="Path to pre-trained Caffe Model")
    argparser.add_argument("-t","--threshold",required=True,type=float, default=0.6,help="Confidence threshold of the detection")
    args = argparser.parse_args()

    #open the source face image
    srcImg = cv2.imread(args.source)
    srcImgGray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    srcMap= np.zeros_like(srcImgGray)
    srcImgHeight,srcImgWidth = srcImg.shape[:2]

    #open the video
    capture = cv2.VideoCapture(args.destination)

    # Get the video dimensions
    vidWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    vidHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)


    detector = cv2.dnn.readNetFromCaffe(args.prototxt,args.model)
    predictor = dlib.shape_predictor(args.predictor)
    srcLandmarkPoints = dnnLandmarks(srcImg,srcImgGray,srcImgHeight,srcImgWidth)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test4.mp4',fourcc, fps, (vidWidth,vidHeight))
    framesnum = 0
    missedFrames = 0

    # Loop over the video frames
    while(capture.isOpened()):
        ret,destImg = capture.read()
        if ret == True:
            destImgGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
            height, width, channels = destImg.shape
            destImgNewFace = np.zeros((height, width, channels), np.uint8)

            #get the landmark points array
            destLandmarkPoints = []
            destLandmarkPoints = dnnLandmarks(destImg,destImgGray,height,width)
            if destLandmarkPoints is None:
                framesnum += 1
                missedFrames += 1
                print("Frame Number: %d  (Missed Frame)" % framesnum )
                out.write(destImg)
                continue
            #Get the landmark points of the srcLandmarkPoints as a numpy array
            srcPoints = np.array(srcLandmarkPoints, np.int32)
            #create the convex hull of the largest polygon
            facePolygon = cv2.convexHull(srcPoints)
            cv2.fillConvexPoly(srcMap, facePolygon, 255)

            # Triangulation
            rect = cv2.boundingRect(facePolygon)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(srcLandmarkPoints)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            #get the points of the triangles
            triangleIndexes = []
            for t in triangles:
                pointOne = (t[0], t[1])
                pointTwo = (t[2], t[3])
                pointThree = (t[4], t[5])

                #Get which point the triangle point is in terms of the landmark index
                indexpointOne = np.where((srcPoints == pointOne).all(axis=1))
                indexpointOne = getFLIndex(indexpointOne)

                indexpointTwo = np.where((srcPoints == pointTwo).all(axis=1))
                indexpointTwo = getFLIndex(indexpointTwo)

                indexpointThree = np.where((srcPoints == pointThree).all(axis=1))
                indexpointThree = getFLIndex(indexpointThree)

                #check that the points are valid and an actual triangle
                if indexpointOne is not None and indexpointTwo is not None and indexpointThree is not None:
                    triangle = [indexpointOne, indexpointTwo, indexpointThree]
                    triangleIndexes.append(triangle)

            destPoints = np.array(destLandmarkPoints, np.int32)
            destfacePolygon = cv2.convexHull(destPoints)

            # Triangulation of both faces
            for tIndex in triangleIndexes:

                # Triangulation of the first face
                srcTr1 = srcLandmarkPoints[tIndex[0]]
                srcTr2 = srcLandmarkPoints[tIndex[1]]
                srcTr3 = srcLandmarkPoints[tIndex[2]]
                #create the triangle
                srcTriangle = np.array([srcTr1, srcTr2, srcTr3], np.int32)

               #create the bounding rectangle around the triangle
                srcRect = cv2.boundingRect(srcTriangle)
                (x, y, w, h) = srcRect
                #crop the triangle and the rectangle
                triangleCropped = srcImg[y: y + h, x: x + w]
                croppedSrcTriangleMask = np.zeros((h, w), np.uint8)

                #get the points of the triangle relative to the bounding box
                srcPoints = np.array([[srcTr1[0] - x, srcTr1[1] - y],
                                   [srcTr2[0] - x, srcTr2[1] - y],
                                   [srcTr3[0] - x, srcTr3[1] - y]], np.int32)

                cv2.fillConvexPoly(croppedSrcTriangleMask, srcPoints, 255)

                # Triangulation of second face
                destTriangle1 = destLandmarkPoints[tIndex[0]]
                destTriangle2 = destLandmarkPoints[tIndex[1]]
                destTriangle3 = destLandmarkPoints[tIndex[2]]
                #create the destination triangle
                destTriangle = np.array([destTriangle1, destTriangle2, destTriangle3], np.int32)

               #create the bounding rectangle around the triangle
                destRect = cv2.boundingRect(destTriangle)
                (x, y, w, h) = destRect
                #create the mask to get only the triangle from the cropped box
                destTriangleCroppedMask = np.zeros((h, w), np.uint8)
                #get the coordinates of the triangle relative to the bounding box
                destPoints = np.array([[destTriangle1[0] - x, destTriangle1[1] - y],
                                    [destTriangle2[0] - x, destTriangle2[1] - y],
                                    [destTriangle3[0] - x, destTriangle3[1] - y]], np.int32)

                cv2.fillConvexPoly(destTriangleCroppedMask, destPoints, 255)

                # Warp triangles
                #convert the points into numpy floats
                srcPoints = np.float32(srcPoints)
                destPoints = np.float32(destPoints)
                #define the metrics for the transfrom
                metrics = cv2.getAffineTransform(srcPoints, destPoints)
                warpedTriangle = cv2.warpAffine(triangleCropped, metrics, (w, h))
                warpedTriangle = cv2.bitwise_and(warpedTriangle, warpedTriangle, mask=destTriangleCroppedMask)

                # Reconstructing destination face and remove Lines
                destImgNewFaceRect = destImgNewFace[y: y + h, x: x + w]
                destImgNewFaceRectGrey = cv2.cvtColor(destImgNewFaceRect, cv2.COLOR_BGR2GRAY)
                _, triangleLineMask = cv2.threshold(destImgNewFaceRectGrey, 1, 255, cv2.THRESH_BINARY_INV)
                warpedTriangle = cv2.bitwise_and(warpedTriangle, warpedTriangle, mask=triangleLineMask)

                destImgNewFaceRect = cv2.add(destImgNewFaceRect, warpedTriangle)
                destImgNewFace[y: y + h, x: x + w] = destImgNewFaceRect

            destImgFaceMask = np.zeros_like(destImgGray)
            destImgHeadMask = cv2.fillConvexPoly(destImgFaceMask, destfacePolygon, 255)
            destImgFaceMask = cv2.bitwise_not(destImgHeadMask)


            destImgHead = cv2.bitwise_and(destImg, destImg, mask=destImgFaceMask)
            result = cv2.add(destImgHead, destImgNewFace)

            (x, y, w, h) = cv2.boundingRect(destfacePolygon)
            faceCentre = (int((x + x + w) / 2), int((y + y + h) / 2))

            seamlessclone = cv2.seamlessClone(result, destImg, destImgHeadMask, faceCentre, cv2.NORMAL_CLONE)

            out.write(seamlessclone)

            framesnum += 1
        else:
            break

    end =time.time()
    print("Processing Frames per Second: %f FPS" % (framesnum/(end-start)))
    print("Processing took: %f Seconds" % (end-start))
    print("Total Missed Frames: %d " % missedFrames)
    print("Percentage of Missed Frames: %f " % ((100*missedFrames)/totalFrames))
    out.release()
    capture.release()
    cv2.destroyAllWindows()
