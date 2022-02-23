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
    cz_max = round(cz_max, 6)
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
        # r['x_up'] = [0]
        # r['y_left'] = [0]
        # r['x_bottom'] = [seleBin.plain_size[0]]
        # r['y_right'] = [seleBin.plain_size[1]]
        maintainEvent(cbox2d, r['x_up'], r['y_left'], r['x_bottom'], r['y_right'])
        seleBin.serial_number += 1
