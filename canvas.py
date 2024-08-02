
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt


class Canvas(QtWidgets.QWidget):
    """ A widget for drawing.
    """

    pixmap = None
    _sizeHint = QtCore.QSize()
    ratio = Qt.AspectRatioMode.KeepAspectRatio
    transformation = Qt.TransformationMode.SmoothTransformation

    def __init__(self, width, height, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.pixmap = QtGui.QPixmap(width, height)
        self.resize(width, height)

        self.lines = []
        self.rects = []
        self.circles = []
        self.named_circles:dict[str,list] = {}

    def setPixmap(self, pixmap):
        if self.pixmap != pixmap:
            self.pixmap = pixmap
            if isinstance(pixmap, QtGui.QPixmap):
                self._sizeHint = pixmap.size()
            else:
                self._sizeHint = QtCore.QSize()
            self.updateGeometry()
            self.updateScaled()

    def setAspectRatio(self, ratio):
        if self.ratio != ratio:
            self.ratio = ratio
            self.updateScaled()

    def setTransformation(self, transformation):
        if self.transformation != transformation:
            self.transformation = transformation
            self.updateScaled()

    def updateScaled(self):
        if self.pixmap:
            self.scaled = self.pixmap.scaled(self.size(), self.ratio, self.transformation)
        self.update()

    def sizeHint(self):
        return self._sizeHint

    def resizeEvent(self, event):
        self.updateScaled()

    def paintEvent(self, event):
        if not self.pixmap:
            return

        qp = QtGui.QPainter(self)
        r = self.scaled.rect()
        r.moveCenter(self.rect().center())
        qp.drawPixmap(r, self.scaled)
        qp.end()

    # ========= you only need to care below ==========

    def update_paint(self):
        """ Flush everything to the screen
        """

        if not self.pixmap:
            return
        
        self.pixmap.fill(QtGui.QColor(0,0,0,10))
        painter = QtGui.QPainter(self.pixmap)
        pen = QtGui.QPen(Qt.GlobalColor.white, 1)
        painter.setPen(pen)
        for x1, y1, x2, y2 in self.lines:
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        painter.setPen(QtGui.QPen(Qt.GlobalColor.red, 2))
        for x1, y1, x2, y2 in self.rects:
            painter.drawRect(x1, y1, x2-x1, y2-y1)

        for x1, y1, r, color in self.circles:
            painter.setPen(QtGui.QPen(color, 1))
            painter.drawEllipse(QtCore.QPointF(x1, y1),r,r)

        for circles in self.named_circles.values():
            for x1, y1, r, color in circles:
                painter.setPen(QtGui.QPen(color, 1))
                painter.drawEllipse(QtCore.QPointF(x1, y1),r,r)

        painter.end()
        self.updateScaled()

    def draw_line(self, x1:float, y1:float, x2:float, y2:float):
        """ Draw a line (x1, y1) -- (x2, y2)
        """
        self.lines.append((x1, y1, x2, y2))

    def draw_rectangle(self, x1:float, y1:float, x2:float, y2:float):
        """ Draw a rect at (x1, y1) -- (x2, y2)
        """
        self.rects.append((x1, y1, x2, y2))

    def draw_circle(self, x1:float, y1:float, r:float, color=Qt.GlobalColor.white):
        """ Draw a circle at (x1, y1) with radius=r.
        """
        self.circles.append((x1, y1, r, color))

    def draw_named_circle(self, name:str, x1:float, y1:float, r:float, color=Qt.GlobalColor.white):
        """ Draw a circle at (x1, y1) with radius=r, with name tag = name.
        """
        if name not in self.named_circles:
            self.named_circles[name] = []
        self.named_circles[name].append((x1, y1, r, color))

    def clear_named_circles(self, name:str):
        """ Clear all circles with name tag = name.
        """
        try:
            self.named_circles[name].clear()
        except KeyError:
            pass