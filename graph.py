from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


class Graph:
    """
    Класс для объектов графика
    """

    def __init__(self, layout, widget, layout_toolbar=None, name_graphics=None,
                 horizontal_axis_name_data=None, vertical_axis_name_data=None):
        # Объекты графика
        self.axis = None
        self.figure = None
        self.canvas = None
        self.toolbar = None
        # Слой - для отрисовки графика
        self.layout = layout
        # Виджет - для отрисовки графика
        self.widget = widget
        self.colorbar = None
        # Название графика
        self.name_graphics = name_graphics
        # Подпись осей
        self.horizontal_axis_name_data = horizontal_axis_name_data
        self.vertical_axis_name_data = vertical_axis_name_data
        # Если передали отдельный слой для toolbar, помещаем его туда
        if layout_toolbar is None:
            self.layout_toolbar = layout
        else:
            self.layout_toolbar = layout_toolbar
        # Вызываем инициализацию
        self.initialize()

    def initialize(self, draw=False):
        """
        Инициализирует фигуру matplotlib внутри контейнера GUI.
        Вызываем только один раз при инициализации
        """
        # Создание фигуры (self.fig и self.ax)
        self.figure = Figure()
        self.axis = self.figure.add_subplot(111)
        # Создание холста
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        if draw:
            self.canvas.draw()

        # Создание Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self.widget, coordinates=True)
        self.layout_toolbar.addWidget(self.toolbar)

    def zoom_area(self, x_min, x_max, y_min, y_max):
        """
        Приближает указанную область
        """
        # На графике задаем область.
        # Сохраняем текущий статус zoom как домашний
        self.toolbar.push_current()

        self.axis.set_xlim([x_min, x_max])
        self.axis.set_ylim([y_min, y_max])

        # Перерисовываем
        self.canvas.draw()
