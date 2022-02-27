import numpy as np
from functools import reduce
from .convex_hull import ConvexHull, point_in_polygen
from .PctTools import AddNewEMSZ, maintainEventBottom

class Stack(object):
    def __init__(self, centre, mass):
        self.centre = centre
        self.mass = mass

class DownEdge(object):
    def __init__(self, box):
        self.box = box
        self.area = None
        self.centre2D = None

def IsUsableEMS(xlow, ylow, zlow, x1, y1, z1, x2, y2, z2):
    if ((x2 - x1 + 1e-6 >= xlow) and (y2 - y1 + 1e-6 >= ylow) and (z2 - z1 + 1e-6 >= zlow)):
        return True
    return False

class Box(object):
    def __init__(self, x, y, z, lx, ly, lz, density, virtual=False):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz

        self.centre = np.array([self.lx + self.x / 2, self.ly + self.y / 2, self.lz + self.z / 2])
        self.vertex_low = np.array([self.lx, self.ly, self.lz])
        self.vertex_high = np.array([self.lx + self.x, self.ly + self.y, self.lz + self.z])
        self.mass = x * y * z * density
        if virtual: self.mass *= 1.0
        self.bottom_edges = []
        self.bottom_whole_contact_area = None

        self.up_edges = {}
        self.up_virtual_edges = {}

        self.thisStack = Stack(self.centre, self.mass)
        self.thisVirtualStack = Stack(self.centre, self.mass)
        self.involved = False


    def calculate_new_com(self, virtual=False):
        new_stack_centre = self.centre * self.mass
        new_stack_mass = self.mass

        for ue in self.up_edges.keys():
            if not ue.involved:
                new_stack_centre += self.up_edges[ue].centre * self.up_edges[ue].mass
                new_stack_mass += self.up_edges[ue].mass

        for ue in self.up_virtual_edges.keys():
            if ue.involved:
                new_stack_centre += self.up_virtual_edges[ue].centre * self.up_virtual_edges[ue].mass
                new_stack_mass += self.up_virtual_edges[ue].mass

        new_stack_centre /= new_stack_mass
        if virtual:
            self.thisVirtualStack.mass = new_stack_mass
            self.thisVirtualStack.centre = new_stack_centre
        else:
            self.thisStack.mass = new_stack_mass
            self.thisStack.centre = new_stack_centre

    def calculated_impact(self):
        if len(self.bottom_edges) == 0:
            return True
        elif not point_in_polygen(self.thisStack.centre[0:2],
                                  self.bottom_whole_contact_area):
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisStack
                self.bottom_edges[0].box.up_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com()
                if not self.bottom_edges[0].box.calculated_impact():
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisStack.centre[0] - e.area[0] > 1e-6 and e.area[2] - self.thisStack.centre[0] > 1e-6  \
                            and self.thisStack.centre[1] - e.area[1] > 1e-6 and e.area[3] - self.thisStack.centre[1] > 1e-6:
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_edges[self] = self.thisStack
                            edge.box.calculate_new_com()
                        else:
                            edge.box.up_edges[self] = Stack(self.thisStack.centre, 0)
                            edge.box.calculate_new_com()

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact():
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisStack.centre[2]])

                    stack0 = Stack(com0, self.thisStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisStack.mass * ratio1)

                    self.bottom_edges[0].box.up_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com()

                    self.bottom_edges[1].box.up_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com()

                    if not self.bottom_edges[0].box.calculated_impact():
                        return False
                    if not self.bottom_edges[1].box.calculated_impact():
                        return False

                else:
                    com2D = self.thisStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = - ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    assgin_ratio = np.linalg.lstsq(coefficient, value, rcond=None)[0]

                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisStack.centre[2]])
                        e.box.up_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com()

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact():
                            return False
            return True

    def calculated_impact_virtual(self, first=False):
        self.involved = True
        if len(self.bottom_edges) == 0:
            self.involved = False
            return True
        elif not point_in_polygen(self.thisVirtualStack.centre[0:2],
                                  self.bottom_whole_contact_area):
            self.involved = False
            return False
        else:
            if len(self.bottom_edges) == 1:
                stack = self.thisVirtualStack
                self.bottom_edges[0].box.up_virtual_edges[self] = stack
                self.bottom_edges[0].box.calculate_new_com(True)
                if not self.bottom_edges[0].box.calculated_impact_virtual():
                    self.involved = False
                    return False
            else:
                direct_edge = None
                for e in self.bottom_edges:
                    if self.thisVirtualStack.centre[0] - e.area[0] > 1e-6 and e.area[2] - self.thisVirtualStack.centre[0] > 1e-6  \
                            and self.thisVirtualStack.centre[1] - e.area[1] > 1e-6 and e.area[3] - self.thisVirtualStack.centre[1] > 1e-6 :
                        direct_edge = e
                        break

                if direct_edge is not None:
                    for edge in self.bottom_edges:
                        if edge == direct_edge:
                            edge.box.up_virtual_edges[self] = self.thisVirtualStack
                            edge.box.calculate_new_com(True)
                        else:
                            edge.box.up_virtual_edges[self] = Stack(self.centre, 0)
                            edge.box.calculate_new_com(True)

                    for edge in self.bottom_edges:
                        if not edge.box.calculated_impact_virtual():
                            self.involved = False
                            return False

                elif len(self.bottom_edges) == 2:
                    com2D = self.thisVirtualStack.centre[0:2]

                    tri_base_line = self.bottom_edges[0].centre2D - self.bottom_edges[1].centre2D
                    tri_base_len = np.linalg.norm(tri_base_line)
                    tri_base_line /= tri_base_len ** 2

                    ratio0 = abs(np.dot(com2D - self.bottom_edges[1].centre2D, tri_base_line))
                    ratio1 = abs(np.dot(com2D - self.bottom_edges[0].centre2D, tri_base_line))

                    com0 = np.array([*self.bottom_edges[0].centre2D, self.thisVirtualStack.centre[2]])
                    com1 = np.array([*self.bottom_edges[1].centre2D, self.thisVirtualStack.centre[2]])

                    stack0 = Stack(com0, self.thisVirtualStack.mass * ratio0)
                    stack1 = Stack(com1, self.thisVirtualStack.mass * ratio1)

                    self.bottom_edges[0].box.up_virtual_edges[self] = stack0
                    self.bottom_edges[0].box.calculate_new_com(True)
                    self.bottom_edges[1].box.up_virtual_edges[self] = stack1
                    self.bottom_edges[1].box.calculate_new_com(True)

                    if not self.bottom_edges[0].box.calculated_impact_virtual() \
                            or not self.bottom_edges[1].box.calculated_impact_virtual():
                        self.involved = False
                        return False

                else:
                    com2D = self.thisVirtualStack.centre[0:2]
                    length = len(self.bottom_edges)
                    coefficient = np.zeros((int(length * (length - 1) / 2 + 1), length))
                    value = np.zeros((int(length * (length - 1) / 2 + 1), 1))
                    counter = 0
                    for i in range(length - 1):
                        for j in range(i + 1, length):
                            tri_base_line = self.bottom_edges[i].centre2D - self.bottom_edges[j].centre2D
                            molecular = np.dot(com2D - self.bottom_edges[i].centre2D, tri_base_line)
                            if molecular != 0:
                                ratioI2J = abs(np.dot(com2D - self.bottom_edges[j].centre2D, tri_base_line)) / molecular
                                coefficient[counter, i] = 1
                                coefficient[counter, j] = -ratioI2J
                            counter += 1

                    coefficient[-1, :] = 1
                    value[-1, 0] = 1
                    x = np.linalg.lstsq(coefficient, value, rcond=None)
                    assgin_ratio = x[0]
                    for i in range(length):
                        e = self.bottom_edges[i]
                        newAdded_mass = self.thisVirtualStack.mass * assgin_ratio[i][0]
                        newAdded_com = np.array([*e.centre2D, self.thisVirtualStack.centre[2]])
                        e.box.up_virtual_edges[self] = Stack(newAdded_com, newAdded_mass)
                        e.box.calculate_new_com(True)

                    for e in self.bottom_edges:
                        if not e.box.calculated_impact_virtual():
                            self.involved = False
                            return False

            if first:
                for e in self.bottom_edges:
                    e.box.up_virtual_edges.pop(self)
            self.involved = False
            return True


class Space(object):
    def __init__(self, width=10, length=10, height=10, size_minimum=0, holder = 60):
        self.plain_size = np.array([width, length, height])
        self.max_axis = max(width, length)
        self.height = height
        self.low_bound = size_minimum

        self.upLetter = np.zeros((holder, 5))
        self.box_vec = np.zeros((holder, 9))
        self.firstEMS = np.array([0, 0, 0, *self.plain_size])
        self.EMS = np.zeros((1000, 6))
        self.NOEMS  = 1
        self.reset()


    def reset(self):
        self.upLetter[:] = 0
        self.letterIdx = 0

        self.box_vec[:] = 0
        self.box_vec[0][-1] =1

        self.EMS[0:self.NOEMS] = 0
        self.EMS[0] = self.firstEMS
        self.NOEMS  = 1

        self.boxes = []
        self.box_idx = 0
        self.serial_number = 0

        self.ZMAP = dict()
        self.ZMAP[0] = dict()

        r = self.ZMAP[0]
        r['x_up'] = [0]
        r['y_left'] = [0]
        r['x_bottom'] = [self.plain_size[0]]
        r['y_right'] = [self.plain_size[1]]

    def interSect2D(self, box):
        if self.box_idx == 0:
            return 0, [], []
        intersect = np.around(np.minimum(box, self.upLetter[0: self.box_idx]), 6)
        signal = (intersect[:, 0] + intersect[:, 2] > 0) * (intersect[:, 1] + intersect[:, 3] > 0) # 等于零的地方表示不相交
        index = np.where(signal)[0]
        if len(index) == 0:
            return 0, [], []
        else:
            return np.max(self.upLetter[index, 4]), index, intersect[index]

    def get_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def scale_down(self, bottom_whole_contact_area):
        centre2D = np.mean(bottom_whole_contact_area, axis=0)
        dirction2D = bottom_whole_contact_area - centre2D
        bottom_whole_contact_area -= dirction2D * 0.1
        return bottom_whole_contact_area.tolist()

    def drop_box(self, box_size, idx, flag, density, setting, **kwags):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx

        if lx + x - 1e-6 > self.plain_size[0] or ly + y - 1e-6 > self.plain_size[1]:
            return False
        if lx + 1e-6 < 0 or ly + 1e-6 < 0:
            return False
        box_info = np.array([-lx, -ly, lx + x, ly + y, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)
        if max_h + z - 1e-6 > self.height:
            return False
        box_info[-1] = max_h + z
        box_now = Box(x, y, z, lx, ly, max_h, density)

        if setting != 2:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.lz + tmp.z - max_h) < 1e-6:

                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1
                    newEdge = DownEdge(tmp)
                    newEdge.area = (x1, y1, x2, y2)
                    newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))
        sta_flag = self.check_box(max_h, box_now, setting)
        if sta_flag:
            self.boxes.append(box_now)  # record rotated box
            self.upLetter[self.box_idx] = box_info
            self.box_vec[self.box_idx] = np.array(
                        [lx, ly, max_h, lx + x, ly + y, max_h + z, 0, 0, 1])
            self.box_idx += 1
            return True
        return False

    # Virtually place an item into the bin,
    # this function is used to check whether the placement is feasible for the current item
    def drop_box_virtual(self, box_size, idx, flag, density, setting, returnH = False, **kwargs):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        checkResult = True
        if lx + x - 1e-6 > self.plain_size[0] or ly + y - 1e-6 > self.plain_size[1]:
            checkResult = False
        if lx + 1e-6 < 0 or ly + 1e-6 < 0:
            checkResult = False

        box_info = np.array([-lx, -ly, lx + x, ly + y, 0])
        max_h, interIdx, interArea = self.interSect2D(box_info)

        if max_h + z - 1e-6 > self.height:
            checkResult = False

        box_now = Box(x, y, z, lx, ly, max_h, density, True)

        if setting != 2 and checkResult:
            combine_contact_points = []
            for inner in range(len(interIdx)):
                idx = interIdx[inner]
                tmp = self.boxes[idx]
                if abs(tmp.lz + tmp.z - max_h) < 1e-6:
                    x1, y1, x2, y2, _ = interArea[inner]
                    x1, y1 = -x1, -y1

                    newEdge = DownEdge(tmp)
                    newEdge.area = (x1, y1, x2, y2)
                    newEdge.centre2D = np.array([x1 + x2, y1 + y2]) / 2
                    box_now.bottom_edges.append(newEdge)
                    combine_contact_points.append([x1, y1])
                    combine_contact_points.append([x1, y2])
                    combine_contact_points.append([x2, y1])
                    combine_contact_points.append([x2, y2])

            if len(combine_contact_points) > 0:
                box_now.bottom_whole_contact_area = self.scale_down(ConvexHull(combine_contact_points))

        if returnH:
            return checkResult and self.check_box(max_h, box_now, setting, True), max_h
        else:
            return checkResult and self.check_box(max_h, box_now, setting, True)

    # Check if the placement is feasible
    def check_box(self, max_h, box_now, setting, virtual=False):
        assert isinstance(setting, int), 'The environment setting should be integer.'
        if setting == 2:
            return True
        else:
            if abs(max_h) < 1e-6:
                return True
            if not virtual:
                result = box_now.calculated_impact()
                return result
            else:
                return box_now.calculated_impact_virtual(True)

    def interSectEMS3D(self, itemLocation):
        itemLocation[0:3] *= -1

        EMS = self.EMS[0:self.NOEMS].copy()
        EMS[:, 0:3] *= -1

        if self.box_idx == 0:
            return 0, [], []

        intersect = np.around(np.minimum(itemLocation, EMS), 6)
        signal = (intersect[:, 0] + intersect[:, 3] > 0) * (intersect[:, 1] + intersect[:, 4] > 0) * (intersect[:, 2] + intersect[:, 5] > 0)
        delindex = np.where(signal)[0] 
        saveindex = np.where(signal == False)[0]
        intersect = intersect[delindex]
        intersect[:, 0:3] *= -1
        return delindex, saveindex, intersect

    # Calculate the incrementally generated empty maximal spaces during the packing.
    def GENEMS(self, itemLocation):
        originemss = self.NOEMS
        delflag, validflag, intersect = self.interSectEMS3D(np.array(itemLocation))

        for idx in range(len(delflag)):
            emsIdx = delflag[idx]
            inter = intersect[idx]
            self.Difference(emsIdx, inter)

        if len(delflag) != 0:
            validflag = [*validflag, *range(originemss, self.NOEMS)]
            validLength = len(validflag)
            self.EMS[0:validLength,:] = self.EMS[validflag, :]
            self.EMS[validLength:self.NOEMS,:] = 0
            self.NOEMS = validLength

        self.NOEMS, self.EMS = self.EliminateInscribedEMS(self.NOEMS, self.EMS)

        # maintain the event point by the way
        cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = itemLocation
        # bottom
        if cz_min < self.plain_size[2]:
            bottomRecorder = self.ZMAP[round(cz_min, 6)]
            cbox2d = [cx_min, cy_min, cx_max, cy_max]
            maintainEventBottom(cbox2d, bottomRecorder['x_up'], bottomRecorder['y_left'], bottomRecorder['x_bottom'],
                                bottomRecorder['y_right'], self.plain_size)

        if cz_max < self.plain_size[2]:
            AddNewEMSZ(itemLocation, self)

    # Split an EMS when it intersects a placed item
    def Difference(self, emsID, intersection):
        x1, y1, z1, x2, y2, z2 = self.EMS[emsID]
        x3, y3, z3, x4, y4, z4, = intersection
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x3, y2, z2):
            self.AddNewEMS(x1, y1, z1, x3, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x4, y1, z1, x2, y2, z2):
            self.AddNewEMS(x4, y1, z1, x2, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z1, x2, y3, z2):
            self.AddNewEMS(x1, y1, z1, x2, y3, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y4, z1, x2, y2, z2):
            self.AddNewEMS(x1, y4, z1, x2, y2, z2)
        if IsUsableEMS(self.low_bound, self.low_bound, self.low_bound, x1, y1, z4, x2, y2, z2):
            self.AddNewEMS(x1, y1, z4, x2, y2, z2)

    def AddNewEMS(self, a, b, c, x, y, z):
        self.EMS[self.NOEMS] = np.array([a, b, c, x, y, z])
        self.NOEMS += 1

    @staticmethod
    # Eliminate redundant ems
    def EliminateInscribedEMS(NOEMS, EMS):
        delflags = np.zeros(NOEMS)
        for i in range(NOEMS):
            for j in range(NOEMS):
                if i == j:
                    continue
                if (EMS[i][0] >= EMS[j][0] and EMS[i][1] >= EMS[j][1]
                    and EMS[i][2] >= EMS[j][2] and EMS[i][3] <= EMS[j][3]
                    and EMS[i][4] <= EMS[j][4] and EMS[i][5] <= EMS[j][5]):
                    delflags[i] = 1
                    break
        saveIdx = np.where(delflags == 0)[0]
        validLength = len(saveIdx)

        if validLength!= 0:
            EMS[0:validLength] = EMS[saveIdx]
        EMS[validLength:NOEMS] = 0
        NOEMS = validLength
        return NOEMS, EMS

    # Convert EMS to placement (leaf node) for the current item.
    def EMSPoint(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for ems in self.EMS:
            for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                if rot == 0:
                    sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                elif rot == 1:
                    sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if abs(sizex - sizey) < 1e-6:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if abs(sizex - sizey) < 1e-6:
                        continue

                if ems[3] - ems[0] + 1e-6 >= sizex and ems[4] - ems[1] + 1e-6 >= sizey and ems[5] - ems[
                    2] + 1e-6 >= sizez:
                    posVec.add((ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    def EventPoint(self, next_box, setting):
            allPostion = []
            if setting == 2: orientation = 6
            else: orientation = 2
            for k in self.ZMAP.keys():
                posVec = set()
                validEms = []

                for emsIdx in range(self.NOEMS):
                    ems = self.EMS[emsIdx]
                    if abs(ems[2] - k) < 1e-6:
                        validEms.append([ems[0], ems[1], -1, ems[3], ems[4], -1])

                if len(validEms) == 0:
                    continue
                validEms = np.array(validEms)
                r = self.ZMAP[k]

                for rot in range(orientation): # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
                    if rot == 0:
                        sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
                    elif rot == 1:
                        sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                        if abs(sizex - sizey) < 1e-6:
                            continue
                    elif rot == 2:
                        sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                        if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                            continue
                    elif rot == 3:
                        sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                        if abs(sizex - sizey) < 1e-6 and abs(sizey - sizez) < 1e-6:
                            continue
                    elif rot == 4:
                        sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                        if abs(sizex - sizey) < 1e-6:
                            continue
                    elif rot == 5:
                        sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                        if abs(sizex - sizey) < 1e-6:
                            continue

                    for xs in r['x_up']:
                        for ys in r['y_left']:
                            xe = xs + sizex
                            ye = ys + sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                        for ye in r['y_right']:
                            ys = ye - sizey
                            xe = xs + sizex
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                    for xe in r['x_bottom']:
                        xs = xe - sizex
                        for ys in r['y_left']:
                            ye = ys + sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))

                        for ye in r['y_right']:
                            ys = ye - sizey
                            posVec.add((xs, ys, k, xe, ye, k + sizez))
                posVec = np.array(list(posVec))
                emsSize = validEms.shape[0]

                cmpPos = posVec.repeat(emsSize, axis=0)

                cmpPos = cmpPos.reshape((-1, *validEms.shape))
                cmpPos = cmpPos - validEms

                cmpPos[:, :, 3] *= -1
                cmpPos[:, :, 4] *= -1
                cmpPos = np.where(cmpPos + 1e-6 > 0, 1, 0)

                cmpPos = cmpPos.cumprod(axis=2)
                cmpPos = cmpPos[:, :, -1]
                cmpPos = np.sum(cmpPos, axis=1)
                validIdx = np.argwhere(cmpPos > 0)
                tmpVec = np.around(posVec[validIdx, :].squeeze(axis=1), 6)
                if len(tmpVec) != 0:
                    tmpVec = np.unique(tmpVec, axis=0)
                allPostion.extend(tmpVec.tolist())
            return allPostion  #
