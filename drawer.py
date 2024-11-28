import itertools
import numpy as np
from graph import Graph
import matplotlib.pyplot as plt


# ШАБЛОНЫ ОТРИСОВКИ ГРАФИКОВ
def cleaning_and_chart_graph(graph: Graph, x_label: str, y_label: str, title: str):
    """
    Очистка и подпись графика (вызывается в начале)
    """
    # Возвращаем зум в домашнюю позицию
    graph.toolbar.home()
    # Очищаем стек осей (от старых x, y lim)
    graph.toolbar.update()
    # Очищаем график
    graph.axis.clear()
    # Задаем название осей
    graph.axis.set_xlabel(x_label)
    graph.axis.set_ylabel(y_label)
    # Задаем название графика
    graph.axis.set_title(title)


def clear_vertical_lines(graph: Graph):
    """
    Очистка вертикальных линий
    """
    # Проверяем, содержит ли объект (graph) атрибут (vertical_lines)
    if hasattr(graph, 'vertical_lines'):
        # Если да, то очищаем его
        for line in graph.vertical_lines:
            line.remove()
        graph.vertical_lines.clear()
    # Иначе инициализируем атрибут vertical_lines
    else:
        graph.vertical_lines = []


def draw_graph(graph: Graph):
    """
    Отрисовка (вызывается в конце)
    """
    # Убеждаемся, что все помещается внутри холста
    graph.figure.tight_layout()
    # Показываем новую фигуру в интерфейсе
    graph.canvas.draw()


class Drawer:
    """
    Класс художник. Имя холст (graph), рисует на нем данные
    """

    # Цвет графиков
    SIGNAL_COLOR = "#ff0000"
    BIT_COLOR = "#4682B4"

    # ОТРИСОВКИ
    @staticmethod
    def graph_signal(graph: Graph, data_x: np.array, data_y: np.array):
        """
        Отрисовка сигнала
        """
        # Очистка, подпись графика и осей (вызывается в начале)
        cleaning_and_chart_graph(
            # Объект графика
            graph=graph,
            # Название графика
            title=graph.name_graphics,
            # Подпись осей
            x_label=graph.horizontal_axis_name_data, y_label=graph.vertical_axis_name_data
        )

        # Рисуем график
        graph.axis.plot(data_x, data_y, color=Drawer.SIGNAL_COLOR)
        # Отрисовка (вызывается в конце)
        draw_graph(graph)

    @staticmethod
    def graph_bit(graph: Graph, data_x: np.array, data_y: np.array):
        """
        Отрисовка битовой последовательности
        """
        # Очистка, подпись графика и осей
        cleaning_and_chart_graph(
            graph=graph,
            title=graph.name_graphics,
            x_label=graph.horizontal_axis_name_data,
            y_label=graph.vertical_axis_name_data
        )

        graph.axis.step(data_x, data_y, where='post', color=Drawer.BIT_COLOR)
        draw_graph(graph)

    @staticmethod
    def graph_convolution(graph: Graph, convolutions: dict[str, list[tuple[float, float]]]):
        """
        Отрисовка сверток для каждого кода Голда
        """
        # Цвета для графиков
        colors = itertools.cycle(['blue', 'green', 'red', 'orange'])

        # Очищаем и подготавливаем график
        cleaning_and_chart_graph(
            graph=graph,
            title=graph.name_graphics,
            x_label=graph.horizontal_axis_name_data,
            y_label=graph.vertical_axis_name_data
        )

        # Создаем пустой список для отслеживания линий на графике
        lines = []

        # Для каждого кода Голда строим отдельный график
        for key, conv_data in convolutions.items():
            time_values = [t * 100 for t, _ in conv_data]
            conv_values = [v for _, v in conv_data]
            color = next(colors)
            line, = graph.axis.plot(time_values, conv_values, label=f"С/о код Голда: {key}", color=color)
            # Добавляем линию в список для отслеживания
            lines.append(line)

        # Добавляем легенду
        legend = graph.axis.legend(loc='lower right')

        # Добавляем обработчик кликов для скрытия/показа графиков
        def on_legend_click(event):
            for legend_line, line in zip(legend.get_lines(), lines):
                if legend_line == event.artist:
                    visible = not line.get_visible()  # Меняем видимость линии
                    line.set_visible(visible)  # Применяем новое состояние видимости
                    legend_line.set_alpha(1.0 if visible else 0.2)  # Изменяем прозрачность элемента легенды
            graph.canvas.draw()  # Обновляем график

        # Привязываем обработчик событий к клику по элементам легенды
        graph.canvas.mpl_connect('pick_event', on_legend_click)

        # Включаем интерактивное выделение элементов легенды
        for legend_line in legend.get_lines():
            legend_line.set_picker(True)  # Делаем элементы легенды кликабельными

        # Отрисовка
        draw_graph(graph)

    @staticmethod
    def building_probability_error(noise_range: np.array, error_probabilities: list):
        """
        Построение графика зависимости вероятности ошибки определения символа от отношения сигнал/шум (SNR)
        """
        plt.figure()
        plt.plot(noise_range, error_probabilities, color='#412C84', marker='o')
        plt.title('Зависимость вероятности ошибки от SNR')
        plt.xlabel('SNR (дБ)')
        plt.ylabel('Вероятность ошибки')
        plt.grid(True)
        plt.ylim(0, 60)
        plt.show()
