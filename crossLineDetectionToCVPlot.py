import pandas as pd
import numpy as np
import time
import os
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import datetime
import psycopg2


conn = psycopg2.connect(database="ybpei", user="stork", password="stork", host="192.168.7.9", port="14103")
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS frameTrajectoryToCV(frameNumber integer, lane0 real, lane1 real,lane2 real,lane3 real,lane4 real,lane5 real,lane6 real,lane7 real,lane8 real,lane9 real,lane10 real,lane11 real,lane12 real,lane13 real,lane14 real,lane15 real,lane16 real,lane17 real);")
cur.execute("CREATE TABLE IF NOT EXISTS numberVehiclesInQueue(recordTime time, allLanes varchar);")
cur.execute('CREATE TABLE IF NOT EXISTS vehiclecrosslinehaitangwanCvplot(time varchar, vehicletype varchar, direction varchar, lanenumber varchar);')    # vehicle passing record

cur.execute('SELECT * FROM lanecoord')
results = cur.fetchall()
allCoor = []
for direction in range(4):
    directionTotalData = results[-1][direction].split('-')    # using the last modified points.
    allCoor.append([])
    for seq, coor in enumerate(directionTotalData):
        allCoor[direction].append(float(coor))



def choosePointInBox(x, y, box):
    # -----p1-------
    # --p4-----p2---
    # -----p3-------
    p1 = box[0]
    p2 = box[1]
    p3 = box[2]
    p4 = box[3]
    if (p2[1] - p1[1]) * (x - p1[0]) + (p1[1] - y) * (p2[0] - p1[0]) < 0 and \
       (p3[1] - p2[1]) * (x - p2[0]) + (p2[1] - y) * (p3[0] - p2[0]) < 0 and \
       (p4[1] - p3[1]) * (x - p3[0]) + (p3[1] - y) * (p4[0] - p3[0]) < 0 and \
       (p1[1] - p4[1]) * (x - p4[0]) + (p4[1] - y) * (p1[0] - p4[0]) < 0:
        return x, y
    else:
        return None



def initialVehicleTracker(vehicleOnFrameData):
    """
    :param vehicleOnFrameData: the vehicle data from the next frame which is not matched with other frame
            format example:
            (1, 1, 0.7664297819137573, (1740.23779296875, 217.07000732421875, 57.69049072265625, 45.55839920043945))
    :return: vehicleTracker: a list that contains the information of the object, the format of the data is:
            [[car count times(0), bus count times(1), truck count times(2),
            label the initial frame number as the new item being initialed(3),
             count the times of the item not matched in the next frame(4), time delta~ total frames(5), f1.P],
            [frame number, vehicle type, confidenceLevel, width size, longitude size, x_position, x_velocity, y_position, y_velocity]]
    """
    vehicleTracker = [[0, 0, 0, 0, 0, 0, np.eye(4)*2.0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    vehicleTracker[0][3] = vehicleOnFrameData[0]
    vehicleTracker[1][0] = vehicleOnFrameData[0]
    vehicleTracker[1][1] = vehicleOnFrameData[1]
    vehicleTracker[1][2] = vehicleOnFrameData[2]
    vehicleTracker[1][3] = vehicleOnFrameData[3][2]  # lx
    vehicleTracker[1][4] = vehicleOnFrameData[3][3]  # ly
    vehicleTracker[1][5] = vehicleOnFrameData[3][0]  # x
    vehicleTracker[1][7] = vehicleOnFrameData[3][1]  # y
    return vehicleTracker


def kalman(ve_xxx, frame_xxx):
    """
    :param ve_xxx: all the coordinate and velocity information of the vehicle 'item' in the container
    :param frame_xxx: the specified('closest' to the 'item') vehicle on the next frame
    :return: the predict and updated coordinates and averaged(using the last 12 points) velocity
    """
    f1 = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1

    f1.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])
    f1.u = 0
    f1.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    f1.R = np.eye(2) * 1
    f1.Q = np.eye(4) * 3
    f1.P = np.eye(4) * ve_xxx[0][-1]
    f1.x = np.array([ve_xxx[-1][5], ve_xxx[-1][6], ve_xxx[-1][7], ve_xxx[-1][8]]).T
    z = np.array(frame_xxx)
    # print(f1.P)
    f1.predict()
    f1.update(z)

    if len(ve_xxx) < 20:
        return f1.x[0], f1.x[0] - ve_xxx[-1][5], f1.x[2], f1.x[2] - ve_xxx[-1][7], f1.P
    else:
        return f1.x[0], (f1.x[0] - ve_xxx[-5][5] + ve_xxx[-1][5] - ve_xxx[-6][5] + ve_xxx[-2][5] - ve_xxx[-7][5]) / 15., \
               f1.x[2], (f1.x[2] - ve_xxx[-5][7] + ve_xxx[-1][7] - ve_xxx[-6][7] + ve_xxx[-2][7] - ve_xxx[-7][7]) / 15., \
               f1.P


def IOU(x1, y1, sx1, sy1, x2, y2, sx2, sy2):
    intersection_x = 0 if (x1+sx1<x2 or x1>x2+sx2) else min(abs(x1-x2-sx2), abs(x1+sx1-x2))
    intersection_y = 0 if (y1+sy1<y2 or y1>y2+sy2) else min(abs(y1-y2-sy2), abs(y1+sy1-y2))
    intersection = intersection_x * intersection_y
    union = sx1*sy1 + sx2*sy2 - intersection
    return intersection / union


def matchTheVehicles(halfHourFramesInOneDiction, startMovieID, totalFrames, vehicleContainer, firstFrameName, dircSeq, lane, laneBox):
    p1 = laneBox[0]
    p2 = laneBox[1]
    p3 = laneBox[2]
    p4 = laneBox[3]
    for tmpVehicleItem in halfHourFramesInOneDiction[firstFrameName]:
        vehicleContainer.append(initialVehicleTracker(tmpVehicleItem))
    # initial the total number of vehicle in the movie
    totalNumberOfVehicle = 0
    processedFile = open(startMovieID + 'haitangwanPreviousIOU.pickle', 'wb')
    completedVehicleTrajectory = {}
    for k in range(totalFrames - 2):
        tmpFrameName = 'frame_' + str(k + 2)
        tmpFrame = halfHourFramesInOneDiction[tmpFrameName]
        # print(tmpFrame)
        tmpMatchedIndex = []
        tmpNewItem = []
        for j, tmpItemPre in enumerate(vehicleContainer):
            if tmpItemPre[0][4] > 2:
                totalNumberOfVehicle += 1
                tmpItemPre[0][5] = len(tmpItemPre) - 1
                newItemName = ('ve_%s' % str(totalNumberOfVehicle))
                # exec(newItemName + '=tmpItemPre')
                completedVehicleTrajectory[newItemName] = tmpItemPre
                del vehicleContainer[j]  # can not delete thoroughly
        for i in range(len(tmpFrame)):
            distance = []
            for tmpItemIndex, tmpItem in enumerate(vehicleContainer):
                # if abs(tmpItem[-1][5] - tmpFrame[i][3][0]) < 50 and abs(tmpItem[-1][7] - tmpFrame[i][3][1]) < 50:
                if tmpFrame[i][1] != tmpItem[-1][1]:
                    distance.append(IOU(tmpItem[-1][5], tmpItem[-1][7], tmpItem[-1][3], tmpItem[-1][4], tmpFrame[i][3][0], tmpFrame[i][3][1], tmpFrame[i][3][2], tmpFrame[i][3][3])-0.2)
                else:
                    distance.append(IOU(tmpItem[-1][5], tmpItem[-1][7], tmpItem[-1][3], tmpItem[-1][4], tmpFrame[i][3][0], tmpFrame[i][3][1], tmpFrame[i][3][2], tmpFrame[i][3][3]))
            distance.append(-1)
            if max(distance) < 0.3:
                vehicleContainer.append(initialVehicleTracker(tmpFrame[i]))
            else:
                minDistanceIndex = distance.index(max(distance))
                tmpMatchedIndex.append(minDistanceIndex)
                if tmpFrame[i][1] == 0:
                    vehicleContainer[minDistanceIndex][0][0] += 1
                elif tmpFrame[i][1] == 1:
                    vehicleContainer[minDistanceIndex][0][1] += 1
                elif tmpFrame[i][1] == 2:
                    vehicleContainer[minDistanceIndex][0][2] += 1
                toKalman = vehicleContainer[minDistanceIndex]
                predictValue = kalman(ve_xxx=toKalman, frame_xxx=[[tmpFrame[i][3][0]], [tmpFrame[i][3][1]]])
                # ve_xxx = toKalman
                # print([ve_xxx[-1][5], ve_xxx[-1][6], ve_xxx[-1][7], ve_xxx[-1][8]])
                vehicleContainer[minDistanceIndex][0][-1] = predictValue[4]
                vehicleContainer[minDistanceIndex].append(
                    [tmpFrame[i][0], tmpFrame[i][1], tmpFrame[i][2], tmpFrame[i][3][2], tmpFrame[i][3][3]
                        , predictValue[0], predictValue[1], predictValue[2], predictValue[3]])
        for jj, iitem in enumerate(vehicleContainer):
            if jj not in tmpMatchedIndex:
                iitem[0][4] += 1
                # print(iitem)
                iitem.append([iitem[-1][0], iitem[-1][1], iitem[-1][2], iitem[-1][3], iitem[-1][4], iitem[-1][5]+iitem[-1][6]*0.1,
                             iitem[-1][6], iitem[-1][7] + iitem[-1][8]*0.1, iitem[-1][8]])
    for vehicleItem in vehicleContainer:
        totalNumberOfVehicle += 1
        vehicleItem[0][5] = len(vehicleItem) - 1
        # print('add', totalNumberOfVehicle, tmpItemPre)
        newItemName = ('ve_%s' % str(totalNumberOfVehicle))
        completedVehicleTrajectory[newItemName] = vehicleItem
    x = {}
    y = {}
    x[lane] = {}
    y[lane] = {}
    zzzz = 0
    startTime = datetime.datetime(2000 + int(startMovieID[0:2]), int(startMovieID[3:5]), int(startMovieID[6:8]), int(startMovieID[9:11]),
                          int(startMovieID[12:14]), int(startMovieID[15:17]))
    for key in completedVehicleTrajectory:
        x[lane][key] = []
        y[lane][key] = []
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)

        inBox = [p1, p1 + 0.4 * (p2 - p1), p4 + 0.4 * (p3 - p4), p4]
        outBox = [p1 + 0.6 * (p2 - p1), p2, p3, p4 + 0.6 * (p3 - p4)]
        if (choosePointInBox(completedVehicleTrajectory[key][1][5], completedVehicleTrajectory[key][1][7], inBox)) or\
                (choosePointInBox(completedVehicleTrajectory[key][-1][5], completedVehicleTrajectory[key][-1][7], outBox)):
            zzzz += 1
            interval = datetime.timedelta(seconds=completedVehicleTrajectory[key][1][0]/25)
            cur.execute(
                "INSERT INTO vehiclecrosslinehaitangwan(time, vehicletype, direction, lanenumber)VALUES(%s,%s,%s,%s)",
                (startTime+interval, completedVehicleTrajectory[key][1][1], dircSeq, lane))
            for tmp in range(1, len(completedVehicleTrajectory[key])):
                x[lane][key].append(completedVehicleTrajectory[key][tmp][5])
                y[lane][key].append(completedVehicleTrajectory[key][tmp][7])
    # print(zzzz)


def matchLastFrameWithCurrent(lastFramePosition, currentFrame, startTime, frameSeqNum):
    '''
    :param lastFramePosition: [
               laneContainer     [[total number of vehicles, total number of type1, total numnber of type2, total number of type3, last record number],  # lane totoal information
                                 [[vehicle tracker Sequence1(0), bus count times(1), truck count times(2),     # first active tracker information
                                    label the initial frame number as the new item being initialed(3),
                                     count the times of the item not matched in the next frame(4), time delta~ total frames(5), f1.P],
                                    [frame number, vehicle type, confidenceLevel, width size, longitude size, x_position, x_velocity, y_position, y_velocity]], .(next traker).., [] ],

                                [count sequence number seperatedly in different lanes],
                                 ...,
                                 [] ] 18 lanes
    :param currentFrame: frame_xx
    :return: currentFramePosition[[], [], [],...[]] 18lanes
    '''
    laneNumber = 0  # to process different lanes
    currentFramePosition = []

    for dirSeq in range(len(allCoor)):   # attention!!!
        for laneSeq in range(int(len(allCoor[dirSeq]) / 6 - 2)):

            # compute every lane separately
            laneBox = chooseCrossLineBoxFromAllMarkedPoint(allCoor, dirSeq, laneSeq)   # define zone of the specified lane
            p1 = np.array(laneBox[0])
            p2 = np.array(laneBox[1])
            p3 = np.array(laneBox[2])
            p4 = np.array(laneBox[3])
            inBox = [p1, p1 + 0.5 * (p2 - p1), p4 + 0.5 * (p3 - p4), p4]
            outBox = [p1 + 0.6 * (p2 - p1), p2, p3, p4 + 0.6 * (p3 - p4)]
            vehicleContainer = lastFramePosition[laneNumber]   # the vehicle information in the small lane box
            laneNumber += 1
            for tmpSeq in range(1, len(vehicleContainer)):  # container of vehicles in the specified lane
                try:
                    if vehicleContainer[tmpSeq][0][4] > 2:
                        interval = datetime.timedelta(seconds=frameSeqNum / 25)
                        if laneSeq == 0:   # for the left turn lane, inbox-outbox
                            if (choosePointInBox(vehicleContainer[tmpSeq][1][5], vehicleContainer[tmpSeq][1][7], inBox))and (choosePointInBox(vehicleContainer[tmpSeq][-1][5], vehicleContainer[tmpSeq][-1][7], outBox)) and ((frameSeqNum - vehicleContainer[0][4])>10):  # len(vehicleContainer[tmpSeq]) > 5
                                vehicleContainer[0][0] += 1  # total information: mark the number of vehicles passed.
                                vehicleContainer[0][int(vehicleContainer[tmpSeq][1][1])+1]+= 1  # total information: mark number of different type of vehicles passed.
                                vehicleContainer[0][4] = frameSeqNum  # record the time when the last vehicle passed
                                cur.execute("INSERT INTO vehiclecrosslinehaitangwanCvplot(time, vehicletype, direction, lanenumber)VALUES(%s,%s,%s,%s)", (startTime + interval, int(vehicleContainer[tmpSeq][1][1]), dirSeq, laneSeq))
                        else:       # for the straight lane and right turn lane: len()>5 and outBox
                            if len(vehicleContainer[tmpSeq]) > 5 and (choosePointInBox(vehicleContainer[tmpSeq][-1][5], vehicleContainer[tmpSeq][-1][7], outBox)) and ((frameSeqNum - vehicleContainer[0][4])>10):
                                vehicleContainer[0][0] += 1  # total information: mark the number of vehicles passed.
                                vehicleContainer[0][int(vehicleContainer[tmpSeq][1][1])+1] += 1  # total information: mark number of different type of vehicles passed.
                                vehicleContainer[0][4] = frameSeqNum
                                cur.execute("INSERT INTO vehiclecrosslinehaitangwanCvplot(time, vehicletype, direction, lanenumber)VALUES(%s,%s,%s,%s)",
                                    (startTime + interval, int(vehicleContainer[tmpSeq][1][1]), dirSeq, laneSeq))
                        del vehicleContainer[tmpSeq]  # can not delete thoroughly In this case it can be done!
                except:
                    break

            # select vehicles in the specified lane from the total data.
            laneBox = chooseCrossLineBoxFromAllMarkedPoint(allCoor, dirSeq, laneSeq)
            vehicleInLane = []
            for vehicleItem in currentFrame:
                if choosePointInBox(vehicleItem[3][0], vehicleItem[3][1], laneBox) is not None:
                    vehicleInLane.append(vehicleItem)

            # match the vehicles on the frames(in small box in this case) with the trajectories in the small container
            tmpMatchedIndex = []
            for i in range(len(vehicleInLane)):
                distance = []
                for tmpItemIndex in range(1, len(vehicleContainer)):
                    if vehicleInLane[i][1] != vehicleContainer[tmpItemIndex][-1][1]:
                        distance.append(IOU(vehicleContainer[tmpItemIndex][-1][5] - 0.5*vehicleContainer[tmpItemIndex][-1][3], vehicleContainer[tmpItemIndex][-1][7] - 0.5*vehicleContainer[tmpItemIndex][-1][4], vehicleContainer[tmpItemIndex][-1][3], vehicleContainer[tmpItemIndex][-1][4], vehicleInLane[i][3][0] - 0.5*vehicleInLane[i][3][2],
                                            vehicleInLane[i][3][1] - 0.5*vehicleInLane[i][3][3], vehicleInLane[i][3][2], vehicleInLane[i][3][3]) - 0.2)
                    else:
                        distance.append(IOU(vehicleContainer[tmpItemIndex][-1][5] - 0.5*vehicleContainer[tmpItemIndex][-1][3], vehicleContainer[tmpItemIndex][-1][7] - 0.5*vehicleContainer[tmpItemIndex][-1][4], vehicleContainer[tmpItemIndex][-1][3], vehicleContainer[tmpItemIndex][-1][4], vehicleInLane[i][3][0] - 0.5*vehicleInLane[i][3][2],
                                            vehicleInLane[i][3][1] - 0.5*vehicleInLane[i][3][3], vehicleInLane[i][3][2], vehicleInLane[i][3][3]))
                distance.append(-1)
                if max(distance) < 0.5:
                    vehicleContainer.append(initialVehicleTracker(vehicleInLane[i]))

                else:
                    minDistanceIndex = distance.index(max(distance))
                    tmpMatchedIndex.append(minDistanceIndex)
                    toKalman = vehicleContainer[minDistanceIndex+1]
                    predictValue = kalman(ve_xxx=toKalman, frame_xxx=[[vehicleInLane[i][3][0]], [vehicleInLane[i][3][1]]])
                    vehicleContainer[minDistanceIndex+1][0][-1] = predictValue[4]
                    vehicleContainer[minDistanceIndex+1].append(
                        [vehicleInLane[i][0], vehicleInLane[i][1], vehicleInLane[i][2], vehicleInLane[i][3][2], vehicleInLane[i][3][3]
                            , predictValue[0], predictValue[1], predictValue[2], predictValue[3]])
            for jj in range(1, len(vehicleContainer)):
                if (jj-1) not in tmpMatchedIndex:
                    vehicleContainer[jj][0][4] += 1

            currentFramePosition.append(vehicleContainer)
    return currentFramePosition


def chooseCrossLineBoxFromAllMarkedPoint(totalMarkedPoint, dircSeq, laneSeq):
    '''
    :param totalMarkedPoint:  all the point in all directions::    # eg. lanes of 4 dires are 5,4,5,4,respectively: large box:[[1, 2, 3,.....42], [1,2,3,...,36],[1,2,3,...,42],[1,2,3,...,36]]
    :param dircSeq:   specify the direction with 0,1,2,3
    :param laneSeq:   specify the lane number of the specified direction.  0 turn left, 1 straight, ..4 turn right
    :return:   the small box of the 'm-th' direction 'n-th' lane
    '''
    # small box four point
    # eg. upper left direction and turn left lane
    # ------ p0 -------
    # --p3 ------p1 ---
    # -------p2 -------
    # large box:[[x1, y1, x2, y2,.....x12, y12], [x1, y1, x2, y2,...,x10, y10],[x1,y1,...,x12, y12],[x1,y1,...,x10, y10]]
    laneSeq += 1
    p0 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 2], totalMarkedPoint[dircSeq][laneSeq * 6 + 3]]
    p1 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 0], totalMarkedPoint[dircSeq][laneSeq * 6 + 1]]
    p2 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 6], totalMarkedPoint[dircSeq][laneSeq * 6 + 7]]
    p3 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 8], totalMarkedPoint[dircSeq][laneSeq * 6 + 9]]
    smallBox = [p0, p1, p2, p3]
    return smallBox


def chooseNumVehInQueBoxFromAllMrkPnt(totalMarkedPoint, dircSeq, laneSeq):
    """
    totoalMarkedPoint: the same as before
    dircSeq: the same as before
    laneSeq: add on lane of each direction to statistic number of vehicles on the road past the intersection, which will be the first number in the list of each direction
    """
    # long box four point
    # eg. upper left direction and turn left lane
    # ------ p0 -------
    # --p3 ------p1 ---
    # -------p2 -------
    # large box:[[x1, y1, x2, y2,.....x12, y12], [x1, y1, x2, y2,...,x10, y10],[x1,y1,...,x12, y12],[x1,y1,...,x10, y10]]
    p0 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 4], totalMarkedPoint[dircSeq][laneSeq * 6 + 5]]
    p1 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 0], totalMarkedPoint[dircSeq][laneSeq * 6 + 1]]
    p2 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 6], totalMarkedPoint[dircSeq][laneSeq * 6 + 7]]
    p3 = [totalMarkedPoint[dircSeq][laneSeq * 6 + 10], totalMarkedPoint[dircSeq][laneSeq * 6 + 11]]
    smallBox = [p0, p1, p2, p3]
    return smallBox


def chooseIntersectionBoxFromAllMrkPnt(totalMarkedPoint):
    """
    totoalMarkedPoint: the same as before
    dircSeq: the same as before
    laneSeq: add on lane of each direction to statistic number of vehicles on the road past the intersection, which will be the first number in the list of each direction
    """
    # long box four point
    # eg. upper left direction and turn left lane
    # ------ p0 -------
    # --p3 ------p1 ---
    # -------p2 -------
    # large box:[[x1, y1, x2, y2,.....x12, y12], [x1, y1, x2, y2,...,x10, y10],[x1,y1,...,x12, y12],[x1,y1,...,x10, y10]]
    p0 = [totalMarkedPoint[0][0], totalMarkedPoint[0][1]]
    p1 = [totalMarkedPoint[1][0], totalMarkedPoint[1][1]]
    p2 = [totalMarkedPoint[2][0], totalMarkedPoint[2][1]]
    p3 = [totalMarkedPoint[3][0], totalMarkedPoint[3][1]]
    smallBox = [p0, p1, p2, p3]
    return smallBox


def writeQueueInforToDB(frame_xxx, startTime, frameSeqNum):

    vehicleInLanes = []
    for dircSeq in range(len(allCoor)):
        for laneSeq in range(int(len(allCoor[dircSeq])/6-1)):
            vehicleInLaneNum = 0
            # select vehicles in the specified lane from the total data.
            laneBox = chooseNumVehInQueBoxFromAllMrkPnt(allCoor, dircSeq, laneSeq)
            for vehicleItem in frame_xxx:
                if choosePointInBox(vehicleItem[3][0], vehicleItem[3][1], laneBox) is not None:
                    vehicleInLaneNum += 1
            vehicleInLanes.append(vehicleInLaneNum)
    
    vehicleInLaneNum = 0
    # select vehicles in the specified lane from the total data.
    intersectionBox = chooseIntersectionBoxFromAllMrkPnt(allCoor)
    for vehicleItem in frame_xxx:
        if choosePointInBox(vehicleItem[3][0], vehicleItem[3][1], intersectionBox) is not None:
            vehicleInLaneNum += 1
    vehicleInLanes.append(vehicleInLaneNum)
    interval = datetime.timedelta(seconds=frameSeqNum / 25)
    recordTime = startTime + interval
    cur.execute("INSERT INTO numberVehiclesInQueue(recordTime, allLanes)VALUES(%s,%s)", (recordTime, vehicleInLanes))


def getFramesFromOneFile(item):
    """
    # fileName = item 
    convert the txt file to the data format can be used by python.
    item: to be processed file name, in txt format
    return: diction contains all vehicles on each frames: frame_xxx
    """
    startMovieID = item[-41:-4]  # mark the start movie ID to calculate time.
    print(startMovieID)
    tic = time.time()
    halfHourFramesInOneDiction = {}  # initial the diction to store the half hour frames(6w)
    totalFrames = 0
    fr = open(item)
    for line in fr.readlines():
        allVehicleStr = line.split(":")[1]
        allVehicleStrList = allVehicleStr.split(
            ";")  # examples print(allVehicleStrList): ['2,1648,681,98,82,0.708', '1,564,364,45,68,0.6', '1,1471,536,84,52,0.559', '1,1436,571,81,52,0.547', '1,472,165,60,50,0.672', '1,1598,616,84,57,0.673', '\n']
        variable = []
        totalFrames += 1

        for vehicleOnFrame in allVehicleStrList:
            vehicleOnFrame = vehicleOnFrame.strip("\n")
            vehicles = vehicleOnFrame.split(",")
            if vehicles != ['']:
                vehicleList = [totalFrames, int(vehicles[0]), float(vehicles[5]),
                               [float(vehicles[1]) + 0.5 * float(vehicles[3]),
                                float(vehicles[2]) + 0.5 * float(vehicles[4]), float(vehicles[3]),
                                float(vehicles[4])]]
                variable.append(vehicleList)
        frameName = 'frame_' + str(totalFrames)
        halfHourFramesInOneDiction[frameName] = variable
    return halfHourFramesInOneDiction

tic = time.time()


def initialVehiclesOnLanes(frameFile):
    laneNumber = 0
    firstFrameContainer = []
    for dirSeq in range(len(allCoor)):   # attention!!!
        for laneSeq in range(int(len(allCoor[dirSeq]) / 6 - 2)):
            laneNumber += 1
            laneBox = chooseCrossLineBoxFromAllMarkedPoint(allCoor, dirSeq, laneSeq)
            vehicleInLane = [[0, 0, 0, 0, 0]]
            for vehicleItem in frameFile:
                if choosePointInBox(vehicleItem[3][0], vehicleItem[3][1], laneBox) is not None:
                    vehicleInLane[0][0] = 0
                    vehicleInLane[0][initialVehicleTracker(vehicleItem)[1][1] + 1] = 0
                    vehicleInLane.append(initialVehicleTracker(vehicleItem))
            firstFrameContainer.append(vehicleInLane)
    return firstFrameContainer


import pickle
# position = open('position.pkl', 'wb')
# countPickle = open('count.pkl', 'wb')


def toRealTimeDisplayCountRecord(lastFrame, currentFrame, startTime, frameSeqNum):
    '''
    lastFrame: 18 lanes, 18 lists; 18 lists->[[total information], [detector 1], [detctor 2]...]
    currentFrame: frame_xxx format.
    startTime: pass timestamp to matchLast..(), to write time of data in data base.
    return: currentFramePoistionList: update 18 lists by using currentFrame information, this value will be used in the next loop.
    '''
    if frameSeqNum == 1:
        return initialVehiclesOnLanes(currentFrame)
    else:
        currentFramePoistionList = matchLastFrameWithCurrent(lastFrame, currentFrame, startTime, frameSeqNum)

        frameCoor = []
        countItem = []
        for i in range(18):
            countItem.append(currentFramePoistionList[i][0])
        for item in currentFramePoistionList:
            for pos in range(1, len(item)):
                frameCoor.append([item[pos][-1][5], item[pos][-1][7], item[pos][-1][3], item[pos][-1][4]])
    return currentFramePoistionList, countItem, frameCoor


def main(fileName):
    halfHourInOneDic = getFramesFromOneFile(fileName)
    lastFrameContainer = initialVehiclesOnLanes(halfHourInOneDic['frame_1'])
    startMovieID = fileName[-41:-4]
    startTime = datetime.datetime(2000 + int(startMovieID[0:2]), int(startMovieID[3:5]), int(startMovieID[6:8]),
                                  int(startMovieID[9:11]), int(startMovieID[12:14]), int(startMovieID[15:17]))
    endTime = datetime.datetime(int(startMovieID[18:22]), int(startMovieID[23:25]), int(startMovieID[26:28]),
                                int(startMovieID[29:31]), int(startMovieID[32:34]), int(startMovieID[35:37]))
    cur.execute("DELETE FROM vehiclecrosslinehaitangwanCvplot WHERE '%s'<time and time<'%s'"%(startTime, endTime))  # 删除重复的数据
    frameSeqNum = 0
    for key in halfHourInOneDic:
        frameSeqNum += 1

        
        # ----------- write count times and vehicles location data --------------
        # tmp = matchLastFrameWithCurrent(lastFrameContainer, halfHourInOneDic[key], startTime, frameSeqNum)
        # lastFrameContainer = tmp
        # frameCoor = []
        # countItem = []
        # # print(lastFrameContainer[17][0])
        # for i in range(18):
        #     countItem.append(lastFrameContainer[i][0])
        # for item in lastFrameContainer:
        #     for pos in range(1, len(item)):
        #         # print(item[pos])
        #         frameCoor.append([item[pos][-1][5], item[pos][-1][7]])
        # # pickle.dump(frameCoor, position)
        # # pickle.dump(countItem, countPickle)
        # if frameSeqNum%100 == 0:
        #     print('------------------------------')

        # ------------write the queue information in data base -----------------
        intervalFrames = 125
        if frameSeqNum % intervalFrames == 0:
            writeQueueInforToDB(halfHourInOneDic[key], startTime, frameSeqNum)


tobeProcFilePath = r"H:\haitangwanVideo\2018-08-15"
fileList = []
for file in os.listdir(tobeProcFilePath):
    filePath = os.path.join(tobeProcFilePath, file)
    if file[-3:] == 'txt':
        fileList.append(filePath)

if __name__ == '__main__':
    for tobeprcfl in fileList:
        main(tobeprcfl)
conn.commit()

toc = time.time()
print('The total time used is:', toc - tic, '(s)')