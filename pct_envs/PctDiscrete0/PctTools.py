import numpy as np

################################################################
################### maintain up layer ##########################
################################################################
def binary_search(the_array, item, start, end):
    if start == end:
        if the_array[start] > item:
            return start
        else:
            return start + 1
    if start > end:
        return start
    mid = round((start + end) / 2)
    if the_array[mid] < item:
        return binary_search(the_array, item, mid + 1, end)
    elif the_array[mid] > item:
        return binary_search(the_array, item, start, mid - 1)
    else:
        return mid

def maintainEvent(cbox, x_up, y_left, x_bottom, y_right):
    cx_min, cy_min, cx_max, cy_max = cbox

    if cx_min not in x_up:
        index = binary_search(x_up, cx_min, 0, len(x_up) - 1)
        x_up.insert(index, cx_min)

    if cx_max not in x_bottom:
        index = binary_search(x_bottom, cx_max, 0, len(x_bottom) - 1)
        x_bottom.insert(index, cx_max)

    if cy_min not in y_left:
        index = binary_search(y_left, cy_min, 0, len(y_left) - 1)
        y_left.insert(index, cy_min)

    if cy_max not in y_right:
        index = binary_search(y_right, cy_max, 0, len(y_right) - 1)
        y_right.insert(index, cy_max)

def maintainEventBottom(cbox, x_start, y_start, x_end, y_end, plain_size):
    cx_start, cy_start, cx_end, cy_end = cbox

    if cx_end not in x_start and cx_end < plain_size[0]:
        index = binary_search(x_start, cx_end, 0, len(x_start) - 1)
        x_start.insert(index, cx_end)

    if cx_start not in x_end:
        index = binary_search(x_end, cx_start, 0, len(x_end) - 1)
        x_end.insert(index, cx_start)

    if cy_end not in y_start and cy_end < plain_size[1]:
        index = binary_search(y_start, cy_end, 0, len(y_start) - 1)
        y_start.insert(index, cy_end)

    if cy_start not in y_end:
        index = binary_search(y_end, cy_start, 0, len(y_end) - 1)
        y_end.insert(index, cy_start)

def AddNewEMSZ(cbox3d, seleBin):
    cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = cbox3d
    cbox2d = [cx_min, cy_min, cx_max, cy_max]
    if cz_max in seleBin.ZMAP.keys():
        r = seleBin.ZMAP[cz_max] # information recorder
        maintainEvent(cbox2d, r['x_up'], r['y_left'], r['x_bottom'], r['y_right'])
    else:
        addflags = []
        delflags = []
        seleBin.ZMAP[cz_max] = dict()
        r = seleBin.ZMAP[cz_max]
        r['x_up'] = []
        r['y_left'] = []
        r['x_bottom'] = []
        r['y_right'] = []
        maintainEvent(cbox2d, r['x_up'], r['y_left'], r['x_bottom'], r['y_right'])
        # r['EMS2D'] = dict()
        # r['EMS2D'][seleBin.serial_number] = [cx_min, cy_min, cx_max, cy_max, seleBin.serial_number]
        # addflags.append(seleBin.serial_number)
        seleBin.serial_number += 1
    # return delflags, addflags

class smallBox():
    def __init__(self,lxs, lys, lxe, lye):
        self.lx = lxs
        self.ly = lys
        self.x = lxe - lxs
        self.y = lye - lys
        self.lxe = lxe
        self.lye = lye

def deleteEps2D(currentBox, allEps):
    delFlag = []
    for i in range(len(allEps)):
        eps = allEps[i]
        if eps[0] >= currentBox.lx and eps[0] < currentBox.lx + currentBox.x and \
           eps[1] >= currentBox.ly and eps[1] < currentBox.ly + currentBox.y:
            delFlag.append(i)
    return [allEps[i] for i in range(len(allEps)) if i not in delFlag]

def IsProjectionValid2D(newItem, item, direction = 0):
    if direction == 0:
        return newItem.lx >= item.lx + item.x and newItem.ly + newItem.y < item.ly + item.y
    if direction == 2:
        return newItem.ly >= item.ly + item.y and newItem.lx + newItem.x < item.lx + item.x


def extreme2D(cboxList):
    if len(cboxList) == 0: return [(0,0,0)]

    cboxList = sorted(cboxList, key= lambda box: (box.ly, box.lxe))
    demo = [smallBox(-1, 0, 0, 10), smallBox(0, -1, 10, 0)]
    alleps = []
    for i in range(0, len(cboxList)):
        subCboxList = cboxList[0:i]
        newItem = cboxList[i]

        maxBound = [-10, -10, -10, -10, -10, -10]  # defined as YX YZ XY XZ ZX ZY
        newEps = {}

        for box in demo + subCboxList:
            projectedX = box.lx + box.x
            projectedY = box.ly + box.y

            if IsProjectionValid2D(newItem, box, 0) and projectedX > maxBound[0]: # YX operation
                newEps[0] = (projectedX, newItem.ly + newItem.y)
                maxBound[0] = projectedX

            if IsProjectionValid2D(newItem, box, 2) and projectedY > maxBound[2]:
                newEps[2] = (newItem.lx + newItem.x, projectedY)
                maxBound[2] = projectedY

        alleps = deleteEps2D(newItem, alleps)

        alleps.extend(list(set(newEps.values())))
    return alleps

def corners2D(cboxList):
    if len(cboxList) == 0: return [(0, 0)]

    cboxList = sorted(cboxList, key=lambda box: (box[3], box[2]), reverse=True)
    xRecord = 0
    m = 0
    em = []

    # Phase 1. Identify the extreme items.
    for i in range(len(cboxList)):
        cbox = cboxList[i]
        if cbox[2] > xRecord:
            em.append(cbox)
            m += 1
            xRecord = cbox[2]

    # Phase 2. Determine the corner points.
    CI = [(0, cboxList[0][3])]
    for idx in range(1, m):
        CI.append((em[idx - 1][2], em[idx][3]))
    CI.append((em[m - 1][2], 0))

    return CI