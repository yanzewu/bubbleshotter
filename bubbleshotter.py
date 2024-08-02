
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt
import sys        
import win32api, win32con
import time
import gui_util
import cvgeom
import bubble
import numpy as np
from panel import Panel


class MainWindow(QtWidgets.QWidget):    # it's not a QMainWindow :D

    resize_level = 2    # <---- change this to match the OS settings (setting->display->scaled layout)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.setWindowFlags(Qt.WindowType.Popup|Qt.WindowType.WindowTransparentForInput|Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setWindowTitle("Bubble Shotter")
        self.resize(300, 200)
        self.move(0, self.pos().y())

        self.canvas = Panel()

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(20, 20, 20, 20)

        ## cached variables

        # shooter
        self.shooter:list[int] = [676.625, 759.125]    # x, y
        self.shooter_cidx:int = 10
        self.radius:float = 18.425
        self.click_distance = 150

        # map
        self.leftwall:float = 321.0   # x
        self.rightwall:float = 1037.5
        self.bmap:bubble.BubbleMap|None = None
        self.intersect_ratio:float = 0.97
        self.rewarded_shots:list = []
        self.is_auto_running = False

        # screenshots utils
        self.bmap_global_loc:tuple[int,int,int,int] = 0,0,1,1
        self.shooter_center_loc:tuple[int,int] = 0, 0
        self.shooter_global_loc:tuple[int,int,int,int] = 0,0,0,0

        # timer

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.auto_run_step)


        ## =========== GUI =============

        def inc_shooter(idx, amount):
            self.shooter[idx] += amount
            print(f'Set shooter position => {self.shooter}')
            self.redraw_shooter()

        def inc_var(name, amount):
            v = getattr(self, name) + amount
            setattr(self, name, v)
            print(f'Set {name} => {v}')

            if name == 'radius':
                self.redraw_shooter()
            elif name == 'leftwall' or name == 'rightwall':
                self.redraw_walls()

        def inc_bmap_gap(name, amount):
            if not self.bmap:
                return
            if name == 'hgap':
                self.bmap.local_hgap += amount
            elif name == 'vgap':
                self.bmap.local_vgap += amount

            print(f'Set bmap gap => horiz={self.bmap.local_hgap}, vert={self.bmap.local_vgap}')
            self.redraw_bmap()

        def make_sub_buttons(name_clicked:dict):
            buttons = []
            for name, clicked in name_clicked.items():
                buttons.append(QtWidgets.QPushButton(name, maximumWidth=30, clicked=clicked))
            return buttons
        
        def make_title_button(name, clicked):
            return QtWidgets.QPushButton(name, maximumWidth=100, clicked=clicked)

        buttons_shooter = [make_title_button("Identify Shooter", lambda: self.canvas.select_rectangle(self.identify_shooter))]
        buttons_shooter += make_sub_buttons({"<-":lambda: inc_shooter(0, -0.2),
                                             "->":lambda: inc_shooter(0, 0.2),
                                             "^":lambda: inc_shooter(1, -0.2),
                                             "v":lambda: inc_shooter(1, 0.2),
                                             "<->":lambda: inc_var('radius', 0.2),
                                             ">-<":lambda: inc_var('radius', -0.2)})
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_shooter))

        buttons_leftwall = [QtWidgets.QPushButton("Left Wall", maximumWidth=100,
                                                      clicked=lambda: self.canvas.select_rectangle(lambda *args: self.identify_wall(0, *args)))]
        buttons_leftwall += make_sub_buttons({"<-":lambda: inc_var('leftwall', -0.2),
                                              "->":lambda: inc_var('leftwall', 0.2)})
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_leftwall))

        buttons_rightwall = [QtWidgets.QPushButton("Right Wall", maximumWidth=100,
                                                       clicked=lambda: self.canvas.select_rectangle(lambda *args: self.identify_wall(1, *args)))]
        buttons_rightwall += make_sub_buttons({"<-":lambda: inc_var('rightwall', -0.2),
                                              "->":lambda: inc_var('rightwall', 0.2)})
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_rightwall))

        buttons_canvas = [make_title_button("Identify Canvas", lambda: self.canvas.select_rectangle(self.identify_bmap))]
        buttons_canvas += make_sub_buttons({"->":lambda: inc_bmap_gap('hgap', 0.05),
                                            "<-":lambda: inc_bmap_gap('hgap', -0.05),
                                            " +v":lambda: inc_bmap_gap('vgap', 0.05),
                                            " -v":lambda: inc_bmap_gap('vgap', -0.05)})
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_canvas))

        buttons_intersect = make_sub_buttons({"+hit":lambda: inc_var('intersect_ratio', 0.01),
                                            "-hit":lambda: inc_var('intersect_ratio', -0.01)})
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_intersect))

        buttons_run = [make_title_button("Run", self.set_run), 
                       make_title_button("Stop", self.canvas.finish_run),
                       make_title_button("Autorun", lambda: self.timer.start(2000)),
                       make_title_button("Stop", lambda: (self.timer.stop(), setattr(self, 'is_auto_running', False), self.canvas.setVisible(True)))]
        
        layout.addLayout(gui_util.create_and_set_layout(QtWidgets.QHBoxLayout, *buttons_run))

        self.canvas.show()

    def set_run(self):
        """ Set running 
        """
        self.canvas.start_run(self.on_mouse_movement, self.on_click)
        self.canvas.q_callback = self.reidentify_bmap_and_shooter
        self.canvas.w_callback = self.get_shots
        self.canvas.e_callback = self.auto_shoot


    # ========== canvas callbacks ===============

    def on_mouse_movement(self, x, y):
        lines, crossing = self.bmap.get_moving_lines_and_crossing(x, y,
                self.radius*2*self.intersect_ratio,
                self.shooter[0], self.shooter[1], 
                self.leftwall+self.radius, self.rightwall-self.radius)

        self.canvas.lines = [
            (self.leftwall, 0, self.leftwall, self.canvas.pixmap.height()),
            (self.rightwall, 0, self.rightwall, self.canvas.pixmap.height())] + lines
        
        if crossing is not None:
            xc, yc = self.bmap.grid2pos(*self.bmap.pos2grid(crossing[0], crossing[1]))
            self.canvas.clear_named_circles('attempt')
            self.canvas.draw_named_circle('attempt', xc, yc, self.radius, Qt.GlobalColor.green)

    def on_click(self, x, y):
        self.canvas.setVisible(False)
        self.send_click(x, y, 1.0)
        self.reidentify_bmap_and_shooter()

    # ========= auto run utils ==========
    
    def auto_run_step(self):
        """ Called when the timer ticks.
        """

        if self.is_auto_running:    # Lock to prevent repeated calls
            return
        
        self.is_auto_running = True

        if self.canvas.isVisible():
            time.sleep(0.5)
            self.canvas.setVisible(False)
        
        self.get_shots()
        if not self.auto_shoot():
            for i in range(10):
                print('Waiting for board...')
                time.sleep(0.25)
                if self.reidentify_bmap_and_shooter():
                    break
            else:
                print('Finished?')  # find error.
                self.timer.stop()
                self.canvas.setVisible(True)

        self.is_auto_running = False
    
    def get_shots(self):
        """ Calculate possible shots => write into self.rewarded_shots, and display them
        """

        shots:dict[tuple, list] = {}    # (grid) => (theta, dist, # of lines)
        rewarded_shots = {}
        
        for theta in np.linspace(np.pi/12, np.pi*11/12, 200):
            lines, crossing = self.bmap.get_moving_lines_and_crossing(
                self.shooter[0] + 10*np.cos(theta), 
                self.shooter[1] - 10*np.sin(theta), # <-- note it's - as y coord is flipped.
                self.radius*2*self.intersect_ratio,
                self.shooter[0], self.shooter[1], 
                self.leftwall+self.radius, self.rightwall-self.radius)
            
            if crossing is not None:
                grid = self.bmap.pos2grid(crossing[0], crossing[1])
                if grid not in shots:
                    shots[grid] = []

                shots[grid].append((theta, self.bmap.shot_reward(crossing, grid, lines)))
        
        self.bmap.get_groups()
        print('groupids=', self.bmap.groupid)

        self.canvas.clear_named_circles('good_shots')
        for grid, theta_u_n in shots.items():
            reward, newmap, gkey = self.bmap.get_reward(grid[0], grid[1], self.shooter_cidx)

            # we want: (1) close to center as possible; (2) no reflection
            best_theta, shot_reward = min(theta_u_n, key=lambda k:k[1])
            print(f"shot={grid}, reward={reward}, angle={best_theta}, shotreward={shot_reward}")
            reward -= 0.01*shot_reward

            if reward > 0:
                p = self.bmap.grid2pos(grid[0], grid[1])
                self.canvas.draw_named_circle('good_shots', p[0], p[1], self.radius, Qt.GlobalColor.red)

            # sometimes multiple shots points at same result! we should check it
            if gkey in rewarded_shots:
                prev_reward = rewarded_shots[gkey][1]
                if prev_reward < reward:
                    rewarded_shots[gkey] = (best_theta, reward, grid, newmap)
            else:
                rewarded_shots[gkey] = (best_theta, reward, grid, newmap)

        self.rewarded_shots = list(rewarded_shots.values())

        self.canvas.update_paint()
        
    def auto_shoot(self):
        """ Performs a one-time shooting.
        """

        if not self.rewarded_shots:
            return False
        
        hr, htheta, hs, hnewmap = -1000, None, None, None
        for theta, reward, s, newmap in self.rewarded_shots:
            if reward > hr:
                hr = reward
                htheta = theta
                hs = s
                hnewmap = newmap

        print(f'Highest Reward found at {hs}: reward={hr} angle={htheta}')
        
        self.canvas.clear_named_circles('good_shots')
        p = self.bmap.grid2pos(hs[0], hs[1])
        self.canvas.draw_named_circle('good_shots', p[0], p[1], self.radius, Qt.GlobalColor.green)

        self.canvas.setVisible(False)
        ncolors = len(np.unique(self.bmap.data))
        if ncolors == 3: # 2 color
            wait_time = 3
        elif ncolors == 4:
            wait_time = 1.25
        else:
            wait_time = 1
        
        self.send_click(self.shooter[0]+self.click_distance*np.cos(htheta), 
                        self.shooter[1]-self.click_distance*np.sin(htheta), wait_time)
        if not self.is_auto_running:
            self.canvas.setVisible(True)
        reidentify_succeeded = self.reidentify_bmap_and_shooter()
        
        if reidentify_succeeded and len(self.bmap.get_unconnected_bubbles(self.bmap.data)) > 0:
            print('Find unconnected components')
            reidentify_succeeded = False

        if reidentify_succeeded:
            mismatch_inc = np.sum(self.bmap.data[hnewmap < 0] >= 0)
            mismatch_dec = np.sum(self.bmap.data[hnewmap >= 0] < 0)
            if mismatch_inc != 0 or mismatch_dec != 0:
                print('Prediction:', hnewmap)
                print('Diff', (hnewmap != self.bmap.data).astype(int))
                print(f'Mismatches: +{mismatch_inc}, -{mismatch_dec}')
            self.redraw_bmap()
            return True
        else:
            return False

    # ========= CV utils ===========

    def identify_shooter(self, x1:int, y1:int, x2:int, y2:int):
        """ Load shooter parameters from CV.
        """
        self.shooter_global_loc = (x1, y1, x2, y2)
        
        pixmap = self.get_screenshot(x1, y1, x2, y2)

        circles = cvgeom.get_circles(cvgeom.pixmap2arr(pixmap))
        self.canvas.rects.clear()
        if len(circles) > 0:
            circle = circles[0]
            center = self.canvas.mapFromGlobal(QtCore.QPointF(x1+circle[0]/self.resize_level, 
                                                              y1+circle[1]/self.resize_level))
            self.radius = circle[2] / self.resize_level
            
            self.shooter = [center.x(), center.y()]
            self.shooter_center_loc = (round(circle[0]), round(circle[1]))
        
        self.redraw_shooter()

    def identify_wall(self, left_or_right:int, x1:int, y1:int, x2:int, y2:int):
        """ Load wall parameters from CV
        """
        pixmap = self.get_screenshot(x1, y1, x2, y2)
        
        vline = cvgeom.get_vline(cvgeom.pixmap2arr(pixmap))
        wall = self.canvas.mapFromGlobal(QtCore.QPointF(x1+vline/self.resize_level, 0)).x()

        if left_or_right == 0:  # left
            self.leftwall = wall
        else:
            self.rightwall = wall

        self.canvas.rects.clear()
        self.redraw_walls()

    def identify_bmap(self, x1:int, y1:int, x2:int, y2:int):
        """ Load board from CV.
        """

        pixmap = self.get_screenshot(x1, y1, x2, y2)
        self.bmap_global_loc = x1, y1, x2, y2

        # self.display_image(cvgeom.arr2pixmap(cvgeom.get_edges(cvgeom.pixmap2arr(pixmap))))
        # self.display_image(pixmap)

        circles = cvgeom.get_circles(cvgeom.pixmap2arr(pixmap), minRadius=round(self.radius*self.resize_level)-4, 
                                     maxRadius=round(self.radius*self.resize_level)+4)
        
        print('Total %s bubbles found' % len(circles))
        
        self.bmap = bubble.create_bmap(cvgeom.rgba2hue(cvgeom.pixmap2arr(pixmap)), circles)

        # convert to local coords
        topleft = self.canvas.mapFromGlobal(QtCore.QPointF(x1+self.bmap.leftbound/self.resize_level, 
                                                          y1+self.bmap.topbound/self.resize_level))
        self.bmap.set_local_pos(topleft.y(), topleft.x(), self.bmap.hgap/self.resize_level, self.bmap.vgap/self.resize_level)

        self.canvas.rects.clear()
        self.redraw_bmap()

        self.reidentify_shooter()
            

    def reidentify_shooter(self, hide_canvas=True) -> bool:
        """ Reidentify the shooter using screenshot at cached position.
        
        Returns: False => the color identification failed.
        """
        if not self.bmap:
            return False
        
        pixmap = self.get_screenshot(*self.shooter_global_loc, hide_canvas=hide_canvas)
        image = cvgeom.pixmap2arr(pixmap)
        circles = cvgeom.get_circles(image, minRadius=round(self.radius*self.resize_level)-4, 
                                     maxRadius=round(self.radius*self.resize_level)+4)
        if len(circles) == 0:
            print('Shooter identification failed')
            return False

        hueimage = cvgeom.rgba2hue(image)
        shooter_color = hueimage[self.shooter_center_loc[0], self.shooter_center_loc[1]]
        self.shooter_cidx = self.bmap.color2cidx(shooter_color)

        print(f'Using exisiting shooter at {self.shooter_center_loc} (global pos {self.shooter_global_loc})')
        print(f'Shooter color={shooter_color} cid={self.shooter_cidx} among all colors={self.bmap.colors}')

        return abs(shooter_color - self.bmap.colors[self.shooter_cidx]) < 5 and shooter_color != 0

    def reidentify_bmap_and_shooter(self):
        """ Reidentify the board using screenshot at cached position.
        
        Returns: False => Identification failed.
        """
        if not self.bmap:
            return False
        
        print('\n------------------------')
        pixmap = self.get_screenshot(*self.bmap_global_loc, show_canvas=False)
        circles = cvgeom.get_circles(cvgeom.pixmap2arr(pixmap), minRadius=round(self.radius*self.resize_level)-4, 
                                     maxRadius=round(self.radius*self.resize_level)+4)
        
        print('Total %s bubbles found' % len(circles))
        if len(circles) > 0 and bubble.reassign_bmap(self.bmap, cvgeom.rgba2hue(cvgeom.pixmap2arr(pixmap)), circles):
            self.redraw_bmap()
            return self.reidentify_shooter(hide_canvas=False)
        else:
            if not self.is_auto_running:
                self.canvas.setVisible(True)
            
        return False

    # ======== painting utils ==========

    def redraw_shooter(self):
        self.canvas.circles.clear()
        print(f'Shooter={self.shooter[0]}, {self.shooter[1]}, Radius={self.radius}')
        self.canvas.draw_circle(self.shooter[0], self.shooter[1], self.radius, Qt.GlobalColor.cyan)
        self.canvas.update_paint()

    def redraw_walls(self):
        self.canvas.lines.clear()
        if self.leftwall != 0:
            print(f'Leftwall={self.leftwall}')
            self.canvas.draw_line(self.leftwall, 0, self.leftwall, self.canvas.pixmap.height())
        
        if self.rightwall != 0:
            print(f'Rightwall={self.rightwall}')
            self.canvas.draw_line(self.rightwall, 0, self.rightwall, self.canvas.pixmap.height())

        self.canvas.update_paint()

    def redraw_bmap(self):
        bubbles = self.bmap.get_bubble_pos()
        print(f'#Bubbles={len(bubbles)}')
        
        self.canvas.clear_named_circles('bmap')
        for b in bubbles:
            self.canvas.draw_named_circle('bmap', b[0], b[1], self.radius, Qt.GlobalColor.red)
        self.canvas.update_paint()


    # ======== low level utilities =============

    def display_image(self, pixmap):
        self.label_image = QtWidgets.QLabel()
        self.label_image.setPixmap(pixmap)
        self.label_image.show()

    def send_click(self, x, y, wait_time:float=1.5):
        """ Actually send the click to the OS.
        """

        gp = self.canvas.mapToGlobal(QtCore.QPointF(x, y))
        gx, gy = round(gp.x()*self.resize_level), round(gp.y()*self.resize_level)
        print(f'Click sent at ({gx}, {gy})')

        old_pos = win32api.GetCursorPos()
        
        win32api.SetCursorPos((gx, gy))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, gx, gy)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, gx, gy)
        win32api.SetCursorPos(old_pos)
        time.sleep(wait_time)

    def get_screenshot(self, x1, y1, x2, y2, hide_canvas=True, show_canvas=True):
        """ Take a screen shot!
        
        For multiple identification, will not set canvas to visible in the end.
        """

        if hide_canvas and not self.is_auto_running:
            self.canvas.setVisible(False)
            time.sleep(0.5)

        screen = QtGui.QGuiApplication.primaryScreen()
        area = (round(x1), round(y1), round(x2-x1), round(y2-y1))

        print('Screenshot take at', area)
        pixmap = screen.grabWindow(0, *area)
        
        if show_canvas and not self.is_auto_running:
            time.sleep(0.1)
            self.canvas.setVisible(True)

        return pixmap
    
    def closeEvent(self, a0) -> None:
        """ Close me => exit
        """
        QtCore.QCoreApplication.instance().quit()


if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
