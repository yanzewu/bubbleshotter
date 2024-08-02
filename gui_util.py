
from PyQt6.QtWidgets import QLayout, QWidget

def add_widgets(layout:QLayout, *widgets:QWidget|QLayout):
    """ Fill a layout with widgets or layouts.
    """
    for w in widgets:
        if isinstance(w, QLayout):
            layout.addLayout(w)
        else:
            layout.addWidget(w)



def create_and_set_layout(layout_type, *widgets):
    """ Create a layout with widgets filled.
    """

    layout = layout_type()
    add_widgets(layout, *widgets)
    return layout
