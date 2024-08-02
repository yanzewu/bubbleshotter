
import numpy as np

class BubbleMap:

    def __init__(self, ncols:int, colors:list[int], 
                 topbound:float, leftbound:float, hgap:float, vgap:float, maxrows:int=20) -> None:
        """ ncols: # of columns.
            colors: List of possible hue, indicating color.
            topbound, leftbound: The coordinate of the top,left bubble.
            hgap, vgap: Horizontal and vertical gap.
            maxlength: Max rows of map.
        """
        
        self.colors = colors
        self.topbound = topbound
        self.leftbound = leftbound
        self.hgap = hgap
        self.vgap = vgap

        self.local_topbound = 0
        self.local_leftbound = 0
        self.local_hgap = hgap
        self.local_vgap = vgap

        self.data = -np.ones((maxrows, ncols), dtype=int)
        self.size = 0
        self.poses = None

        self.unconnected_cache = {}
    
    def set_local_pos(self, topbound:float, leftbound:float, hgap:float, vgap:float):
        """ Set parameters at local coordinates.
        """
        self.local_topbound = topbound
        self.local_leftbound = leftbound
        self.local_hgap = hgap
        self.local_vgap = vgap

    def assign(self, vidx:np.ndarray, hidx:np.ndarray, cidx:np.ndarray):
        """ Assign the colors.
        """
        self.data.fill(-1)
        self.size = 0
        try:
            self.data[vidx, hidx] = cidx
        except ValueError:
            print(vidx, hidx, cidx)
            print('Assign failed')
            return False
        except IndexError:
            print(vidx, hidx, cidx)
            print('Assign failed')
            return False
        else:
            self.size = len(cidx)
            return True
        
    # ========= coord utils =============

    def get_bubble_pos(self, update:bool=True):
        """ Get positions of bubbles, in local coordinates. (x, y)
        """

        if not update and self.poses is not None:
            return self.poses
        
        bubbles = np.argwhere(self.data >= 0)
        poses = np.zeros((len(bubbles), 2))
        poses[:,1] = bubbles[:,0] * self.local_vgap + self.local_topbound
        poses[:,0] = bubbles[:,1] * self.local_hgap + self.local_leftbound + (bubbles[:,0] % 2 != 0) * self.local_hgap / 2
        self.poses = poses

        return poses

    def pos2grid(self, x:float, y:float):
        """ align a point to a bubble in a grid.
        """
        vidx = round((y - self.local_topbound)/self.local_vgap)
        hidx = round((x - self.local_leftbound - (self.local_hgap/2 if vidx % 2 == 1 else 0))/self.local_hgap)
        if hidx < 0:
            return 0, (vidx if self.data[0,vidx] < 0 else vidx+1)
        elif hidx == self.data.shape[1]:
            return self.data.shape[1]-1, (vidx if self.data[self.data.shape[1]-1,vidx] < 0 else vidx+1)

        return hidx, vidx
    
    def grid2pos(self, hidx:float, vidx:float):
        """ Get the position of a bubble.
        """
        y = vidx * self.local_vgap + self.local_topbound
        x = hidx * self.local_hgap + self.local_leftbound + (vidx % 2 != 0) * self.local_hgap / 2
        return x, y
    
    def color2cidx(self, color:int) -> int:
        """ Convert color (hue value) to color index.
        """
        return np.argmin(np.abs(self.colors - color))
    
    
    # ========== intersections ================

    def intersect(self, x0:float, y0:float, x:float, y:float, radius:float):
        """ For a bubble trajectory that passes (x0, y0) and (x, y), check which point it intersect at.
        Returns None,None if no intersection.
        """

        R = self.get_bubble_pos(update=False)
        d = np.abs((y-y0)*R[:,0] - (x-x0)*R[:,1] + x*y0-y*x0) / np.sqrt((y-y0)**2+(x-x0)**2)
        
        sel = d < radius
        if np.any(sel):
            Rsel = R[sel]
            l = np.inf
            for m_ in range(len(Rsel)):
                xc_, yc_ = point_circle_intersection(Rsel[m_,0], Rsel[m_,1], radius, x0, y0, x, y)
                l_ = (xc_ - x0)**2 + (yc_ - y0)**2
                if l_ < l:
                    xc, yc = xc_, yc_
                    l = l_
            return xc, yc

        else:
            return None, None
        
    def get_moving_lines_and_crossing(self, x:float, y:float, r:float, x0:float, y0:float, 
                                      leftwall:float, rightwall:float):
        """ Given a mouse position, Returns the lines to draw and crossing.
        
        x0, y0: The shooter position.
        x, y: The mouse position.
        r: The effective hitting radius (ideally = 2*bubble radius)
        leftwall, rightwall: The walls.

        Returns: lines: list((x1, y1, x2, y2)); crossing: (xc, yc) or None.
        """
        
        lines:list[tuple[int|float,int|float,int|float,int|float]] = []
        crossing = None

        if x == x0:
            _, yc = self.intersect(x0, y0, x, y, r)

            if yc is not None:
                crossing = (0, yc)
                lines.append((x0, y0, x0, yc))
            else:
                lines.append((x0, y0, x0, 0))
        
        else:
            if (x-x0) > 0: # right side
                x1, x2 = leftwall, rightwall
            else:
                x1, x2 = rightwall, leftwall

            k = (y-y0)/(x-x0)

            xc, ytop = self.intersect(x0, y0, x, y, r)

            if xc is not None:
                crossing = (xc, ytop)
                lines.append((x0, y0, xc, ytop))
            else:
                ytop = y0 + (x2-x0)*k
                lines.append((x0, y0, x2, ytop))
            
            # yshift = self.radius*(1+k**2)**0.5
            # lines.append((x0, y0-yshift, xt, ytop-yshift))
            # lines.append((x0, y0+yshift, xt, ytop+yshift))

            yl = self.local_topbound

            if crossing is None and ytop > yl:
                
                xc, ytop2 = self.intersect(x2, ytop, x2+1, ytop-k, r)

                if xc is not None:
                    crossing = (xc, ytop2)
                    lines.append((x2, ytop, xc, ytop2))
                else:
                    lines.append((x2, ytop, x1, ytop - (x1-x2)*k))

            # hit the ceiling
            if crossing is None and lines[-1][3] <= yl:
                xl = ((yl-lines[-1][3])*lines[-1][0]+(lines[-1][1]-yl)*lines[-1][2])/(lines[-1][1]-lines[-1][3])
                crossing = xl, yl
        
        return lines, crossing
    
    def shot_reward(self, crossing:tuple[float, float], grid:tuple[int, int], lines:list[tuple]):
        """ Get the rewarding function for different shot angles (angle does not appear explicitly).
        NOTE The rewarding function is OPPOSITE order: SMALLER means better.
        """
    
        gpos = self.grid2pos(*grid)

        # distance to the target grid
        dist = np.sqrt((crossing[0]-gpos[0])**2 + (crossing[1]-gpos[1])**2)/(self.local_hgap/2)
        reflected = len(lines)-1

        return dist + 0.5*reflected
        

    # ================ Graph operations ==========
    
    @staticmethod
    def neighbours(v, h, ncols):
        
        l = []
        if v > 0:
            l = [(v+1, h), (v-1, h)]
            if v % 2 == 0:
                if h < ncols-1:
                    l.append((v, h+1))
                if h > 0:
                    l += [(v,h-1), (v+1, h-1), (v-1, h-1)]
            else:
                if h < ncols-1:
                    l += [(v,h+1), (v+1, h+1), (v-1, h+1)]
                if h > 0:
                    l.append((v, h-1))
        else:
            l = [(v+1,h)]
            if v % 2 == 0:
                if h < ncols-1:
                    l.append((v, h+1))
                if h > 0:
                    l += [(v,h-1), (v+1, h-1)]
            else:
                if h < ncols-1:
                    l += [(v,h+1), (v+1, h+1)]
                if h > 0:
                    l.append((v, h-1))
        return l

    @staticmethod
    def get_unconnected_bubbles(data:np.ndarray):
        """ Returns a list of index of unconnected bubbles (vidx, hidx)
        """

        visited = np.zeros(data.shape, dtype=bool)
        
        # DFS
        stack = []
        for h in range(data.shape[1]):
            if data[0, h] >= 0 and not visited[0, h]:
                stack.append((0, h))
            while stack:
                v, h = stack.pop()
                if not visited[v, h]:
                    visited[v, h] = True
                    for n in BubbleMap.neighbours(v, h, data.shape[1]):
                        if data[n] >= 0 and not visited[n]:
                            stack.append(n)
        
        unconnected = np.argwhere(np.logical_and(data >= 0, np.logical_not(visited)))
        return unconnected
    
    def get_groups(self):
        """ Get color groups (will store them to cache)
        """

        self.unconnected_cache.clear()
        groups = []
        groupid = -np.ones_like(self.data)
        visited = np.zeros(self.data.shape, dtype=bool)

        # DFS

        stack = []
        gid = -1
        
        for v, h in np.ndindex(self.data.shape):
            if not visited[v, h] and self.data[v, h] >= 0: 
                stack.append((v, h))
                groups.append([])
                gid += 1
            else:
                continue

            while stack:
                v, h = stack.pop()
                if not visited[v, h]:
                    visited[v, h] = True
                    groupid[v, h] = gid
                    groups[gid].append((v, h))
                    for n in self.neighbours(v, h, self.data.shape[1]):
                        if self.data[n] >= 0 and not visited[n] and self.data[n] == self.data[v,h]:
                            stack.append(n)


        self.groups = groups
        self.groupid = groupid

        return groups, groupid

    def get_reward(self, hidx:int, vidx:int, cidx:int):
        """ The rewarding function. (hidx, vidx) is the attempt shooting location, cidx is the color idx.

        Returns: reward, new_map_data, groupids. groupids as a flag for identify unique shot.
        """
        
        groupids = set()
        redundancy = 0
        for n in self.neighbours(vidx, hidx, self.data.shape[1]):
            if self.data[n] == cidx:
                groupids.add(self.groupid[n])
                redundancy += 1
        redundancy -= len(groupids)

        group = []
        for gid in groupids:
            group += self.groups[gid]


        data_copy = self.data.copy()
        if len(group) >= 2:
            # print('group', group)
            # print('neighbours', self.neighbours(vidx, hidx, self.data.shape[1]))
            g = np.array(group)
            data_copy[g[:,0], g[:,1]] = -1

            uct_key = tuple(sorted(groupids))
            try:
                unconnected = self.unconnected_cache[uct_key]
            except KeyError:
                unconnected = self.get_unconnected_bubbles(data_copy)
                self.unconnected_cache[uct_key] = unconnected

            if len(unconnected) > 0:
                # print('Unconnected:', unconnected)
                data_copy[unconnected[:,0], unconnected[:,1]] = -1
                g = np.vstack((g, unconnected))

                # it's possible that there will be 2nd unconnected; but seems do not appear when color > 2.

            r_indv = 1 + 0.02 * g[:,0]
            reward = 10.0 + np.sum(r_indv)
            
            # additional boost 
            ymax = np.max(g[:,0])
            yave = np.mean(g[:,0])
            reward += ((2/3-yave/6)*(yave <=4) + 2/3*(ymax-6)**2*(ymax>6))
            x = np.mean(g[:,1])
            reward += 0.5*(1 - abs((x - self.data.shape[1]/2)/(self.data.shape[1]/2)))

            reward += 0.2*redundancy
            return reward, data_copy, frozenset(groupids)
        
        elif len(group) == 1:
            data_copy[vidx, hidx] = cidx
            reward = -vidx + 0.5*abs((hidx - self.data.shape[1]/2)/(self.data.shape[1]/2))
            return reward, data_copy, frozenset(groupids)
        
        else:
            reward = -5-vidx + abs((hidx - self.data.shape[1]/2)/(self.data.shape[1]/2))
            data_copy[vidx, hidx] = cidx
            return reward, data_copy, frozenset(groupids)
        
            

def create_bmap(image:np.ndarray, circles:np.ndarray, mingap:int=5, colorgap:int=5):
    """ image: P x Q array representing hue.
        circles: N x 2 array representing positions
        mingap: Minimum distance between two circles.
        colorgap: Minimum hue difference.
    """

    leftbound = np.min(circles[:,0])
    topbound = np.min(circles[:,1])
    leftbound = np.mean(circles[circles[:,0]-leftbound < mingap,0])
    topbound = np.mean(circles[circles[:,1]-topbound < mingap,1])

    hgap_half = np.min(circles[circles[:,0]-leftbound>mingap,0]) - leftbound
    vgap = np.min(circles[circles[:,1]-topbound>mingap,1]) - topbound
    hgap_half = np.mean(circles[np.logical_and(circles[:,0]-leftbound-hgap_half < mingap, circles[:,0]>=leftbound+hgap_half), 0]) - leftbound
    vgap = np.mean(circles[np.logical_and(circles[:,1]-topbound-vgap < mingap, circles[:,1]>=topbound+vgap), 1]) - topbound


    hgap_half = np.mean(circles[np.round((circles[:,0] - leftbound)/hgap_half) == 10, 0] - leftbound) / 10

    print(f'Map found: leftbound={leftbound}, topbound={topbound}, hgap={hgap_half}, vgap={vgap}')

    vidx = np.round((circles[:,1] - topbound)/vgap).astype(int)
    hidx = np.round((circles[:,0] - leftbound)/hgap_half).astype(int) // 2   # for odd rows

    # identify colors
    colors = image[np.round(circles[:,1]).astype(int), np.round(circles[:,0]).astype(int)].astype(int)
    coloridx = np.zeros(len(circles), dtype=int)
    colorlist = np.array([colors[0]], dtype=int)

    for j in range(1,len(colors)):
        i = np.argwhere(np.abs(colorlist - colors[j]) < colorgap)
        if len(i) == 0:
            colorlist = np.append(colorlist, colors[j])
            coloridx[j] = len(colorlist)-1
            continue
        elif len(i) > 1:
            print(f'Color identified error for bubbles {j}: color={colors[j]}, find={i}')
        
        coloridx[j] = i[0,0]

    print('colors=', colorlist)

    bmap = BubbleMap(np.max(hidx)+1, colorlist, topbound, leftbound, hgap_half*2, vgap)
    
    if bmap.assign(vidx, hidx, coloridx):
        print(bmap.data)

    return bmap


def reassign_bmap(bmap:BubbleMap, image:np.ndarray, circles:np.ndarray):
    """ Reassign maps from an exisiting image.
    image: 2D array of hue image.
    circles: N x 2 array representing the circle centers.

    Returns True/False if reassignment succeeded/failed.
    """

    vidx = np.round((circles[:,1] - bmap.topbound)/bmap.vgap).astype(int)
    hidx = np.round((circles[:,0] - bmap.leftbound)/(bmap.hgap/2)).astype(int) // 2   # for odd rows
    colors = image[np.round(circles[:,1]).astype(int), np.round(circles[:,0]).astype(int)].astype(int)
    cidx = np.argmin(np.abs(colors[:,None]-bmap.colors[None,:]), axis=1)
    
    if np.max(colors - bmap.colors[cidx]) > 5:
        print('Invalid color')
        print(colors)
        return False

    if bmap.assign(vidx, hidx, cidx):
        print(bmap.data)
        return True
    else:
        return False


def point_circle_intersection(x0:float, y0:float, radius:float, x1:float, y1:float, x2:float, y2:float):
    """ find the two points where a secant intersects a circle 
    (x0, y0): circle center.
    (x1, y1) - (x2, y2): the line.
    """

    dx, dy = x2 - x1, y2 - y1

    a = dx**2 + dy**2
    b = 2 * (dx * (x1 - x0) + dy * (y1 - y0))
    c = (x1 - x0)**2 + (y1 - y0)**2 - radius**2

    discriminant = b**2 - 4 * a * c
    assert (discriminant > 0), 'Not a secant!'

    t1:float = (-b + discriminant**0.5) / (2 * a)
    t2:float = (-b - discriminant**0.5) / (2 * a)

    if dy*t1 < dy*t2:
        return dx * t2 + x1, dy * t2 + y1
    else:
        return dx * t1 + x1, dy * t1 + y1
    


if __name__ == '__main__':
    bmap = BubbleMap(17, [], 0, 0, 0, 0)
    bmap.data = np.array([[1,4,3,0,4,4,0,5,3,2,2,2,3,4,2,0,0],
[4,4,3,4,5,2,4,2,2,0,1,2,0,5,2,0,2],
[4,0,3,5,2,5,0,4,4,5,1,3,1,5,4,5,4],
[4,4,5,2,2,5,1,3,0,0,5,3,0,3,3,2,4],
[2,1,0,5,0,4,3,1,2,0,2,1,1,5,1,3,1],
[1,3,5,1,4,5,2,5,2,0,1,3,0,2,0,2,4],
[0,2,1,5,0,5,-1,2,4,1,3,1,3,4,5,3,3],
[5,5,3,-1,-1,-1,-1,2,4,0,4,5,0,5,0,3,4],
[4,3,4,-1,-1,-1,-1,-1,5,4,2,3,2,1,2,4,5],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,4,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    bmap.get_groups()
    print(bmap.groupid)