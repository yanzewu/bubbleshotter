
from typing import Callable, Optional, Any
import canvas
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt

class Panel(canvas.Canvas):
    """ The window used for selection
    """

    MODE_SEL = 1
    MODE_RUN = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(1200, 800, *args, **kwargs)

        # self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint) #|Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
        # self.setWindowOpacity(0.4)
        self.setStyleSheet('background-color: transparent')

        self.mode:int = 0   # 0/1/2
        self.pos1:Optional[QtCore.QPoint] = None    # on selection mode, the first corner of the rectangle

        # mouse
        self.sel_finish_callback:Optional[Callable[[int,int,int,int],Any]] = None
        self.mouse_movement_callback:Optional[Callable[[int,int],Any]] = None
        self.click_callback:Optional[Callable[[int,int],Any]] = None

        # key pressing
        self.q_callback = None
        self.w_callback = None
        self.e_callback = None

    def start_sel(self):
        """ Start selection mode.
        """
        self.finish_run()
        self.activateWindow()
        self.mode = self.MODE_SEL

    def start_run(self, mouse_movement_callback:Callable[[int,int],Any], click_callback:Callable[[int,int],Any]):
        """ Start running mode.
        
        mouse_movement_callback(int, int): Called when the mouse is moving when MODE==2. Will provide LOCAL position x and y.
        click_callback(int, int): Called when mouse released when MODE==2. Will provide LOCAL position x and y.
        """
        self.mouse_movement_callback = mouse_movement_callback
        self.click_callback = click_callback

        self.setMouseTracking(True)
        self.activateWindow()
        self.mode = self.MODE_RUN

    def finish_run(self):
        """ End the running mode.
        """
        self.setMouseTracking(False)
        self.lines.clear()
        self.update_paint()
        self.mode = 0

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.mode == self.MODE_SEL:
            self.pos1 = ev.pos()

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.mode == self.MODE_SEL and self.pos1:
            self.rects.clear()
            self.draw_rectangle(self.pos1.x(), self.pos1.y(), a0.pos().x(), a0.pos().y())
            self.update_paint()
        
        elif self.mode == self.MODE_RUN and self.mouse_movement_callback:
            self.mouse_movement_callback(a0.pos().x(), a0.pos().y())
            self.update_paint()

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        if self.mode == self.MODE_SEL:
            self.mode = 0
            if self.sel_finish_callback and self.pos1:
                g1 = self.mapToGlobal(self.pos1)
                g2 = self.mapToGlobal(a0.pos())
                
                # reorder -- otherwise screenshot will panic
                x1, x2 = sorted((g1.x(), g2.x()))
                y1, y2 = sorted((g1.y(), g2.y()))

                # execute the callback
                self.sel_finish_callback(x1, y1, x2, y2)
            
            self.pos1 = None
            self.sel_finish_callback = None
        
        elif self.mode == self.MODE_RUN and self.click_callback:
            self.click_callback(a0.pos().x(), a0.pos().y())

    def keyPressEvent(self, a0:QtGui.QKeyEvent) -> None:
        if self.mode == self.MODE_RUN:
            if a0.key() == Qt.Key.Key_Q and self.q_callback:
                self.q_callback()
            elif a0.key() == Qt.Key.Key_W and self.w_callback:
                self.w_callback()
            elif a0.key() == Qt.Key.Key_E and self.e_callback:
                self.e_callback()

    def select_rectangle(self, sel_finish_callback:Callable[[int,int,int,int],Any]):
        """ Start the selection of a rectangle; and then call 
            sel_finish_callback(x1, y1, x2, y2), where x and y are GLOBAL positions.
        """

        self.sel_finish_callback = sel_finish_callback
        self.start_sel()