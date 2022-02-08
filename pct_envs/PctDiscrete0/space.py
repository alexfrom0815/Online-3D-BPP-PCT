import numpy as np
from functools import reduce
import copy
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

    xd = x2 - x1
    yd = y2 - y1
    zd = z2 - z1
    if ((xd >= xlow) and (yd >= ylow) and (zd >= zlow)):
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

    def printInfo(self):
        print('{}, {}, {}, {}, {}, {}'.format(self.x, self.y, self.z, self.vertex_low, self.thisVirtualStack.centre,
                                              self.thisVirtualStack.mass))
        for e in self.bottom_edges:
            e.box.printInfo()

    def standardize(self):
        return tuple([self.x, self.y, self.z, self.lx, self.ly, self.lz])

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
                    if self.thisStack.centre[0] > e.area[0] and self.thisStack.centre[0] < e.area[2] \
                            and self.thisStack.centre[1] > e.area[1] and self.thisStack.centre[1] < e.area[3]:
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
                    if self.thisVirtualStack.centre[0] > e.area[0] and self.thisVirtualStack.centre[0] < e.area[2] \
                            and self.thisVirtualStack.centre[1] > e.area[1] and self.thisVirtualStack.centre[1] < \
                            e.area[3]:
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
        # fixed
        self.plain_size = np.array([width, length, height])
        self.max_axis = max(width, length)
        self.height = height
        self.low_bound = size_minimum

        # init needed
        self.plain = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.space_mask = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.left_space = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        self.box_vec = np.zeros((holder, 9))
        self.box_vec[0][-1] = 1

        self.reset()
        self.alleps = []

        self.EMS3D = dict()
        self.EMS3D[0] = np.array([0, 0, 0, width, length, height, self.serial_number])

    def reset(self):
        self.plain[:] = 0
        self.space_mask[:] = 0
        self.left_space[:] = 0
        self.box_vec[:] = 0
        self.box_vec[0][-1] =1

        self.NOEMS = 1
        self.EMS = [np.array([0, 0, 0, *self.plain_size])]

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

        self.EMS3D = dict()
        self.EMS3D[0] = np.array([0, 0, 0, self.plain_size[0], self.plain_size[1], self.plain_size[2], self.serial_number])

    def print_height_graph(self):
        print(self.plain)

    def get_height_graph(self):
        plain = np.zeros(shape=(self.max_axis, self.max_axis), dtype=np.int32)
        for box in self.boxes:
            plain = self.update_height_graph(plain, box)
        return plain

    @staticmethod
    def update_height_graph(plain, box):
        plain = copy.deepcopy(plain)
        le = box.lx
        ri = box.lx + box.x
        up = box.ly
        do = box.ly + box.y
        max_h = np.max(plain[le:ri, up:do])
        max_h = max(max_h, box.lz + box.z)
        plain[le:ri, up:do] = max_h
        return plain

    def get_box_list(self):
        vec = list()
        for box in self.boxes:
            vec += box.standardize()
        return vec

    def get_plain(self):
        return copy.deepcopy(self.plain)

    def get_action_space(self):
        return self.plain_size[0] * self.plain_size[1]

    def get_ratio(self):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * self.plain_size[2]
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def get_ratio_vary(self, h):
        vo = reduce(lambda x, y: x + y, [box.x * box.y * box.z for box in self.boxes], 0.0)
        mx = self.plain_size[0] * self.plain_size[1] * h
        ratio = vo / mx
        assert ratio <= 1.0
        return ratio

    def idx_to_position(self, idx):
        lx = idx // self.plain_size[0]
        ly = idx % self.plain_size[0]
        return lx, ly

    def position_to_index(self, position):
        assert len(position) == 2
        assert position[0] >= 0 and position[1] >= 0
        assert position[0] < self.plain_size[0] and position[1] < self.plain_size[1]
        return position[0] * self.plain_size[1] + position[1]

    def scale_down(self, bottom_whole_contact_area):
        centre2D = np.mean(bottom_whole_contact_area, axis=0)
        dirction2D = bottom_whole_contact_area - centre2D
        bottom_whole_contact_area -= dirction2D * 0.1
        return bottom_whole_contact_area.tolist()

    def drop_box(self, box_size, idx, flag, density, setting):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        rec = self.plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)
        box_now = Box(x, y, z, lx, ly, max_h, density)

        if setting != 2:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.lz + tmp.z == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
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

        sta_flag = self.check_box(x, y, lx, ly, z, max_h, box_now, setting)
        if sta_flag:
            self.boxes.append(box_now)  # record rotated box
            self.plain = self.update_height_graph(self.plain, self.boxes[-1])
            self.height = max(self.height, max_h + z)
            self.box_vec[self.box_idx] = np.array(
                        [lx, ly, max_h, lx + x, ly + y, max_h + z, density, 0, 1])
            self.box_idx += 1
            return True
        return False

    def drop_box_virtual(self, box_size, idx, flag, density, setting):
        if not flag:
            x, y, z = box_size
        else:
            y, x, z = box_size

        lx, ly = idx
        rec = self.plain[lx:lx + x, ly:ly + y]
        max_h = np.max(rec)

        box_now = Box(x, y, z, lx, ly, max_h, density, True)

        if setting != 2:
            combine_contact_points = []
            for tmp in self.boxes:
                if tmp.lz + tmp.z == max_h:
                    x1 = max(box_now.vertex_low[0], tmp.vertex_low[0])
                    y1 = max(box_now.vertex_low[1], tmp.vertex_low[1])
                    x2 = min(box_now.vertex_high[0], tmp.vertex_high[0])
                    y2 = min(box_now.vertex_high[1], tmp.vertex_high[1])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    else:
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

        return self.check_box(x, y, lx, ly, z, max_h, box_now, setting, True)

    def check_box(self, x, y, lx, ly, z, max_h, box_now, setting, virtual=False):
        if lx + x > self.plain_size[0] or ly + y > self.plain_size[1]:
            return False
        if lx < 0 or ly < 0:
            return False
        if max_h + z > self.height:
            return False

        if setting == 2:
            return True
        else:
            if max_h == 0:
                return True
            if not virtual:
                result = box_now.calculated_impact()
                return result
            else:
                return box_now.calculated_impact_virtual(True)

    def GENEMS(self, itemLocation):
        numofemss = len(self.EMS)
        delflag = []
        for emsIdx in range(numofemss):
            xems1, yems1, zems1, xems2, yems2, zems2 = self.EMS[emsIdx]
            xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2 = itemLocation

            if (xems1 > xtmp1): xtmp1 = xems1
            if (yems1 > ytmp1): ytmp1 = yems1
            if (zems1 > ztmp1): ztmp1 = zems1
            if (xems2 < xtmp2): xtmp2 = xems2
            if (yems2 < ytmp2): ytmp2 = yems2
            if (zems2 < ztmp2): ztmp2 = zems2

            if (xtmp1 > xtmp2): xtmp1 = xtmp2
            if (ytmp1 > ytmp2): ytmp1 = ytmp2
            if (ztmp1 > ztmp2): ztmp1 = ztmp2
            if (xtmp1 == xtmp2 or ytmp1 == ytmp2 or ztmp1 == ztmp2):
                continue

            self.Difference(emsIdx, (xtmp1, ytmp1, ztmp1, xtmp2, ytmp2, ztmp2))
            delflag.append(emsIdx)

        if len(delflag) != 0:
            NOEMS = len(self.EMS)
            self.EMS = [self.EMS[i] for i in range(NOEMS) if i not in delflag]
        self.EliminateInscribedEMS()

        # maintain the event point by the way
        cx_min, cy_min, cz_min, cx_max, cy_max, cz_max = itemLocation
        # bottom
        if cz_min < self.plain_size[2]:
            bottomRecorder = self.ZMAP[cz_min]
            cbox2d = [cx_min, cy_min, cx_max, cy_max]
            maintainEventBottom(cbox2d, bottomRecorder['x_up'], bottomRecorder['y_left'], bottomRecorder['x_bottom'],
                                bottomRecorder['y_right'], self.plain_size)

        if cz_max < self.plain_size[2]:
            AddNewEMSZ(itemLocation, self)

    def Difference(self, emsID, intersection):
        x1, y1, z1, x2, y2, z2 = self.EMS[emsID]
        x3, y3, z3, x4, y4, z4, = intersection
        if self.low_bound == 0:
            self.low_bound = 0.1
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
        self.EMS.append(np.array([a, b, c, x, y, z]))

    def EliminateInscribedEMS(self):
        NOEMS = len(self.EMS)
        delflags = np.zeros(NOEMS)
        for i in range(NOEMS):
            for j in range(NOEMS):
                if i == j:
                    continue
                if (self.EMS[i][0] >= self.EMS[j][0] and self.EMS[i][1] >= self.EMS[j][1]
                        and self.EMS[i][2] >= self.EMS[j][2] and self.EMS[i][3] <= self.EMS[j][3]
                        and self.EMS[i][4] <= self.EMS[j][4] and self.EMS[i][5] <= self.EMS[j][5]):
                    delflags[i] = 1
                    break
        self.EMS = [self.EMS[i] for i in range(NOEMS) if delflags[i] != 1]
        return len(self.EMS)

    #######################################################################################
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
                    if sizex == sizey:
                        continue
                elif rot == 2:
                    sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 3:
                    sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                    if sizex == sizey and sizey == sizez:
                        continue
                elif rot == 4:
                    sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                    if sizex == sizey:
                        continue
                elif rot == 5:
                    sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                    if sizex == sizey:
                        continue

                if ems[3] - ems[0] >= sizex and ems[4] - ems[1] >= sizey and ems[5] - ems[2] >= sizez:
                    posVec.add((ems[0], ems[1], ems[2], ems[0] + sizex, ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[1], ems[2], ems[3], ems[1] + sizey, ems[2] + sizez))
                    posVec.add((ems[0], ems[4] - sizey, ems[2], ems[0] + sizex, ems[4], ems[2] + sizez))
                    posVec.add((ems[3] - sizex, ems[4] - sizey, ems[2], ems[3], ems[4], ems[2] + sizez))
        posVec = np.array(list(posVec))
        return posVec

    #######################################################################################
    def FullCoord(self, next_box, setting):
        posVec = set()
        if setting == 2: orientation = 6
        else: orientation = 2

        for rot in range(orientation):  # 0 x y z, 1 y x z, 2 x z y,  3 y z x, 4 z x y, 5 z y x
            if rot == 0:
                sizex, sizey, sizez = next_box[0], next_box[1], next_box[2]
            elif rot == 1:
                sizex, sizey, sizez = next_box[1], next_box[0], next_box[2]
                if sizex == sizey:
                    continue
            elif rot == 2:
                sizex, sizey, sizez = next_box[0], next_box[2], next_box[1]
                if sizex == sizey and sizey == sizez:
                    continue
            elif rot == 3:
                sizex, sizey, sizez = next_box[1], next_box[2], next_box[0]
                if sizex == sizey and sizey == sizez:
                    continue
            elif rot == 4:
                sizex, sizey, sizez = next_box[2], next_box[0], next_box[1]
                if sizex == sizey:
                    continue
            elif rot == 5:
                sizex, sizey, sizez = next_box[2], next_box[1], next_box[0]
                if sizex == sizey:
                    continue

            for lx in range(self.plain_size[0]):
                for ly in range(self.plain_size[1]):
                    lz = self.plain[lx, ly]
                    if lx + sizex <= self.plain_size[0] and ly + sizey <= self.plain_size[1] \
                            and lz + sizez <= self.plain_size[2]:
                        posVec.add((lx, ly, lz, lx + sizex, ly + sizey, lz + sizez))

        posVec = np.array(list(posVec))
        return posVec