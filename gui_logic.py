import math
import time
import random
import matplotlib
import numpy as np
from PyQt5.QtCore import Qt, QCoreApplication

import matplotlib.pyplot as plt

from graph import Graph
from gui import Ui_Dialog
from drawer import Drawer as drawer


matplotlib.use('TkAgg')


def uniform_distribution() -> float:
    """
    Функция для создания нормального распределения по Гауссу
    """
    repeat = 12
    val = 0
    for i in range(repeat):
        # Сумма случайных чисел от 0.0 до 1.0
        val += random.random()

    return val / repeat


class GuiProgram(Ui_Dialog):
    """
    Класс алгоритма приложения
    """

    def __init__(self, dialog):
        # Создаем окно
        Ui_Dialog.__init__(self)
        # Дополнительные функции окна.
        # Передаем флаги создания окна (Закрытие | Во весь экран (развернуть) | Свернуть)
        dialog.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        # Устанавливаем пользовательский интерфейс
        self.setupUi(dialog)

        # ПОЛЯ КЛАССА
        # Параметры 1 графика - Синфазная компонента
        self.graph_1 = Graph(
            layout=self.layout_plot,
            widget=self.widget_plot,
            name_graphics="График №1. I компонента (синфазная компонента)",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Значение бита [отн. ед.]"
        )
        # Параметры 2 графика - Квадратурная компонента
        self.graph_2 = Graph(
            layout=self.layout_plot_2,
            widget=self.widget_plot_2,
            name_graphics="График №2. Q компонента (квадратурная компонента)",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Значение бита [отн. ед.]"
        )
        # Параметры 3 графика - Комплексная огибающая
        self.graph_3 = Graph(
            layout=self.layout_plot_3,
            widget=self.widget_plot_3,
            name_graphics="График №3. Комплексная огибающая",
            horizontal_axis_name_data="Время (t) [c]",
            vertical_axis_name_data="Амплитуда (A) [отн. ед.]"
        )
        # Параметры 4 графика - График сверток согласованных фильтров
        self.graph_4 = Graph(
            layout=self.layout_plot_4,
            widget=self.widget_plot_4,
            name_graphics="График №4. График сверток согласованных фильтров",
            horizontal_axis_name_data="Время (t) [с]",
            vertical_axis_name_data="Значение корреляции [отн. ед.]"
        )

        # Частота дискретизации (Гц)
        self.sampling_rate = None
        # Число бит исходной передаваемой информации
        self.bit_counts = None
        # Скорость передачи данных (бит/с)
        self.bit_rate = None
        # Битовая последовательность
        self.bits = None

        # Длина M-последовательности
        self.length_m_sequence = None
        # Преобразованная кодами Голда битовая последовательность
        self.converted_bit_sequence = None
        # Коды Голда
        self.gold_codes = None

        # Несущая частота огибающей
        self.frequency = None
        # Фаза огибающей
        self.phase = None
        # Длительность бита
        self.bit_length = None
        # Шаг по времени
        self.time_step = None
        # Комплексная огибающая
        self.complex_envelope = None
        # Отсчеты времени для комплексной огибающей
        self.time_ox = None

        # ДЕЙСТВИЯ ПРИ ВКЛЮЧЕНИИ.
        # Спрятать область ввода битовой последовательности
        self.lineEdit_bits.setVisible(False)

        # Алгоритм обратки
        # Моделирование сигнала
        self.pushButton_simulate_signal.clicked.connect(self.drawing_signal_components)
        # Добавление шума
        self.pushButton_add_noise.clicked.connect(self.drawing_signal_noise)
        # Восстановление исходной информации
        self.pushButton_restore_the_original_information.clicked.connect(self.drawing_convolutions)
        # Исследование устойчивости
        self.pushButton_probability_of_detection.clicked.connect(self.research_resistance_noise)

    # АЛГОРИТМ РАБОТЫ ПРОГРАММЫ
    # (1) Генерация кодов Голда и преобразование случайной последовательности
    def generation_m_sequence(self, feedback_numbers: list[int]) -> list[int]:
        """
        Генерация M-последовательности
        """
        n = len(feedback_numbers)
        m_sequence_length = (1 << n) - 1  # 1 * 2^n - 1
        m_sequence = list()
        register = [0] * n
        register[1] = 1  # начальное состояние

        for _ in range(m_sequence_length):
            # Последний бит в последовательность
            m_sequence.append(register[-1])
            feedback = sum(x * h for x, h in zip(feedback_numbers, register)) % 2
            for j in range(n - 1, 0, -1):
                # Сдвиг регистра
                register[j] = register[j - 1]
            register[0] = feedback

        return m_sequence

    def get_Gold_code(self, m1: list[int], m2: list[int]) -> list[int]:
        """
        Получение кода Голда
        """
        # Побитовая сумма двух M-последовательностей по модулю два
        return [i1 ^ i2 for i1, i2 in zip(m1, m2)]

    def shift_array(self, arr: list[int], shift: int) -> list[int]:
        """
        Циклический сдвиг массива
        """
        return arr[-shift:] + arr[:-shift]

    def generation_code_Gold(self) -> dict[str, list[int]]:
        """
        Генерация кодов Голда
        """
        # M-последовательность — последовательность максимальной длины, которая имеет период 2n−1,
        # где n — число разрядов регистра сдвига.
        # По условию задачи: разрядность сдвигового регистра 5 => 2^5 - 1 = 31 - длина M-последовательности
        m1 = self.generation_m_sequence([1, 0, 0, 0, 1])
        m2 = self.generation_m_sequence([1, 0, 1, 1, 0])
        self.length_m_sequence = len(m1) if len(m1) >= len(m2) else len(m2)
        # Предпочтительные M-последовательности:
        # m1 = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
        # m2 = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
        gold_codes = {
            "00": self.get_Gold_code(m1, self.shift_array(m2, 5)),
            "10": self.get_Gold_code(m1, self.shift_array(m2, 13)),
            "01": self.get_Gold_code(m1, self.shift_array(m2, 19)),
            "11": self.get_Gold_code(m1, self.shift_array(m2, 26)),
        }

        return gold_codes

    def converting_sequences_to_Gold_codes(self, random_sequence: bool) -> None:
        """
        Преобразование последовательности в последовательность кодов Голда
        """
        if self.gold_codes is None:
            return

        converted_bit_sequence = list()
        if random_sequence:
            # Число бит исходной передаваемой информации
            self.bit_counts = int(self.lineEdit_bit_counts.text())
            self.bits = np.random.randint(0, 2, self.bit_counts)
        else:
            self.bits = [int(bit) for bit in self.lineEdit_bits.text()]
            self.bit_counts = len(self.bits)

        # Делим исходную последовательность по 2 бита
        for i in range(0, len(self.bits) - 1, 2):
            key = f"{self.bits[i]}{self.bits[i + 1]}"
            # Заменяем соответствующим кодом Голда
            converted_bit_sequence += self.gold_codes[key]

        self.converted_bit_sequence = converted_bit_sequence

    # (2) Вычисление компонент и комплексной огибающей сигнала
    def calculate_iq_components(self) -> tuple[list[float], list[float]] | None:
        """
        Вычисление I и Q компонент
        """
        if self.converted_bit_sequence is None:
            return

        self.bit_rate = int(self.lineEdit_transfer_rate.text())
        self.sampling_rate = int(self.lineEdit_sampling_rate.text())
        self.frequency = int(self.lineEdit_carrier_frequency.text())
        self.phase = int(self.lineEdit_phase.text())
        self.bit_length = 1 / self.bit_rate
        self.time_step = 1 / self.sampling_rate

        # Синфазная компонента
        i_component = list()
        # Квадратурная компонента
        q_component = list()
        # Временные отсчеты для оси 0x
        time_ox = list()

        index = 0
        for i in range(0, len(self.converted_bit_sequence) - 1, 2):
            bi1 = self.converted_bit_sequence[i]
            bi2 = self.converted_bit_sequence[i + 1]
            for _ in range(int(2 * (self.bit_length / self.time_step))):
                # Текущее время
                current_time = index * self.time_step
                time_ox.append(current_time)
                i_component.append(bi1)
                q_component.append(bi2)
                index += 1

        self.time_ox = time_ox

        return i_component, q_component

    def calculate_complex_envelope(self) -> list[float] | None:
        """
        Вычисление комплексной огибающей
        """
        if self.converted_bit_sequence is None or self.bit_length is None:
            return

        # Комплексная огибающая
        complex_envelope = list()

        index = 0
        for i in range(0, len(self.converted_bit_sequence) - 1, 2):
            bi1 = self.converted_bit_sequence[i]
            bi2 = self.converted_bit_sequence[i + 1]
            for _ in range(int(2 * (self.bit_length / self.time_step))):
                # Текущее время
                current_time = index * self.time_step
                envelope = ((bi1 - 0.5) * np.cos(2 * np.pi * self.frequency * current_time) -
                            (bi2 - 0.5) * np.sin(2 * np.pi * self.frequency * current_time + self.phase))
                complex_envelope.append(envelope)
                index += 1

        return complex_envelope

    def drawing_signal_components(self):
        """
        Отрисовка I и Q компонентов и комплексной огибающей
        """
        self.gold_codes = self.generation_code_Gold()
        random_sequence = True if self.radioButton_random.isChecked() else False
        self.converting_sequences_to_Gold_codes(random_sequence)
        i_component, q_component = self.calculate_iq_components()
        self.complex_envelope = self.calculate_complex_envelope()

        drawer.graph_bit(self.graph_1, self.time_ox, i_component)
        drawer.graph_bit(self.graph_2, self.time_ox, q_component)
        drawer.graph_signal(self.graph_3, self.time_ox, self.complex_envelope)

    # (3) Добавление шума в децибелах (дБ)
    def add_noise(self, signal: list[float], noise_decibels: int | None = None) -> list[float] | None:
        """
        Добавление шума (в дБ) к сигналу
        """
        if signal is None or noise_decibels is None:
            return

        size_signal = len(signal)
        # Создаем массив отсчетов шума равный размеру сигнала
        noise_counting = np.zeros(size_signal)

        # Считаем энергию шума
        energy_noise = 0
        for j in range(size_signal):
            val = uniform_distribution()
            # Записываем отсчет шума
            noise_counting[j] = val
            energy_noise += val * val

        # Считаем энергию исходного сигнала
        energy_signal = 0
        for i in range(size_signal):
            energy_signal += signal[i] * signal[i]

        # Считаем коэффициент/множитель шума: sqrt(10^(-x/10) * (E_signal / E_noise)), x - с экрана
        noise_coefficient = math.sqrt(pow(10, (-noise_decibels / 10)) * (energy_signal / energy_noise))
        # Копируем исходный сигнал
        noise_signal = signal.copy()
        # К отсчетам исходного сигнала добавляем отсчеты шума
        for k in range(size_signal):
            noise_signal[k] += noise_coefficient * noise_counting[k]

        return noise_signal

    def drawing_signal_noise(self):
        """
        Отрисовка комплексной огибающей с шумом
        """
        if self.converted_bit_sequence is None:
            return

        complex_envelope = self.calculate_complex_envelope()
        noise_decibels = int(self.lineEdit_noise.text())
        self.complex_envelope = self.add_noise(complex_envelope, noise_decibels)
        # complex_envelope_fft = np.fft.fft(self.complex_envelope)

        drawer.graph_signal(self.graph_3, self.time_ox, self.complex_envelope)
        # drawer.graph_signal(self.graph_1, self.time_ox[1:], np.abs(complex_envelope_fft)[1:])

    # (4) Восстановление исходной информации путем согласованной фильтрации
    def calculated_convolution(self) -> dict[str, list[tuple[float, float]]] | None:
        """
        Расчет свертки исходного сигнала с импульсными характеристиками
        """
        if self.gold_codes is None or self.bit_length is None:
            return

        convolutions = {}
        index = 0

        # Фиксируем время начала выполнения кода
        start = time.time()
        # Текущий шаг для отслеживания прогресса
        current_step = 0
        self.progressBar_recovery.setMaximum(len(self.gold_codes))
        self.progressBar_recovery.setValue(current_step)

        # Проходим по последовательностям Голда (4 кода Голда)
        for key, value in self.gold_codes.items():
            # Синтез фильтра
            filter_signal = []
            for i in range(len(value) - 1):
                b1 = value[i] - 0.5
                b2 = value[i + 1] - 0.5
                for _ in range(int(self.bit_length / self.time_step)):
                    # Текущее время
                    current_time = index * self.time_step
                    index += 1
                    # Значение огибающей в текущее время
                    temp = (b1 * np.cos(2 * np.pi * self.frequency * current_time + self.phase) -
                            b2 * np.sin(2 * np.pi * self.frequency * current_time + self.phase))
                    filter_signal.append((current_time, temp))

            # Считаем свертку
            convolutions[key] = self.get_cross_correlation(filter_signal)

            current_step += 1
            self.progressBar_recovery.setValue(current_step)
            # Обновление интерфейса
            QCoreApplication.processEvents()

        # Фиксируем время окончания выполнения кода
        finish = time.time()
        self.label_execution_time_recovery.setText(f'{finish - start:.2f} с')

        return convolutions

    def get_cross_correlation(self, filter_signal: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
        """
        Вычисление взаимной корреляции комплексной огибающей с фильтром
        """
        if self.complex_envelope is None:
            return

        cross_correlation = []
        filter_values = [value for _, value in filter_signal]
        time_values_filter = [time_values for time_values, _ in filter_signal]
        # Шаг времени
        time_step = time_values_filter[1] - time_values_filter[0]

        # Считаем свертку между фильтром и огибающей
        for i in range(len(self.complex_envelope) - len(filter_values) + 1):
            temp = 0
            for j in range(len(filter_values)):
                temp += self.complex_envelope[i + j] * filter_values[j]

            # Текущее время
            current_time = i * time_step
            # Добавляем значение корреляции с соответствующим временем
            cross_correlation.append((current_time, temp / len(filter_values)))

        return cross_correlation

    # (5) Декодирование сигнала
    def decode_signal(self, convolutions: dict[str, list[tuple[float, float]]]) -> tuple[int, float] | None:
        """
        Декодирование сигнала
        """
        if self.length_m_sequence is None or self.bits is None:
            return

        length_convolution = len(convolutions["00"])
        count_bits_filter = len(self.converted_bit_sequence) // self.length_m_sequence - 1
        interval = int(self.bit_length / self.time_step * self.length_m_sequence)
        start_end = (length_convolution - interval * (count_bits_filter - 1) - 1) // 2

        result = []
        i = 0

        while i < length_convolution - 1:
            range_size = start_end if i == 0 or i == length_convolution - start_end - 1 else interval

            max_value = float('-inf')
            max_key = ""

            for key, conv in convolutions.items():
                # Поиск максимального значения в отрезке
                temp_max = max(conv[i:i + range_size], key=lambda p: p[1])[1]
                if max_value < temp_max:
                    max_value = temp_max
                    max_key = key

            result.extend([1 if c == '1' else 0 for c in max_key])
            i += start_end if i == 0 or i == length_convolution - start_end - 1 else interval

        # Количество ошибок
        number_errors = sum(1 for bit, res in zip(self.bits, result) if bit != res)
        # Вероятность ошибки
        error_rate = number_errors / len(self.bits)

        return number_errors, error_rate

    def drawing_convolutions(self):
        """
        Отрисовка сверток согласованных фильтров
        """
        if self.gold_codes is None:
            return

        convolutions = self.calculated_convolution()
        _, error_rate = self.decode_signal(convolutions)

        self.label_probability_error.setText(f'{error_rate * 100:.2f} %')
        drawer.graph_convolution(self.graph_4, convolutions)

        # Количество отсчетов (сэмплов) на один бит
        samples_bits = math.ceil(self.sampling_rate / self.bit_rate)
        # Отображаем битовую последовательность
        bit_counts_ox = np.arange(0, self.bit_counts, 1 / samples_bits)
        bit_signal = np.repeat(self.bits, samples_bits)
        self.graph_2.name_graphics = 'График №4. Восстановленная битовая последовательность'
        drawer.graph_bit(self.graph_2, bit_counts_ox, bit_signal)
        self.graph_2.name_graphics = 'График №2. Q компонента (квадратурная компонента)'

    # Отрисовка в другом окне
    # def draw_convolution(self, convolutions: dict[str, list[tuple[float, float]]]) -> None:
    #     """
    #     Отрисовка сверток для каждого кода Голда.
    #     :param convolutions: Словарь, содержащий свертки (ключ — код Голда, значение — данные свертки)
    #     """
    #     plt.figure(figsize=(12, 8))
    #
    #     # Для каждого кода Голда строим отдельную линию
    #     for key, conv_data in convolutions.items():
    #         time_values = [t for t, _ in conv_data]
    #         conv_values = [v for _, v in conv_data]
    #         plt.plot(time_values, conv_values, label=f"Convolution for Gold Code {key}")
    #
    #     # Настройки графика
    #     plt.title("Convolutions with Gold Codes")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Correlation Value")
    #     plt.legend()
    #     plt.grid(True)
    #
    #     # Отображение графика
    #     plt.show()

    # (6) Исследование устойчивости к шуму
    def research_resistance_noise(self):
        """
        Исследование устойчивости алгоритма к шуму.
        Построение графика зависимости вероятности ошибки определения символа от отношения сигнал/шум (SNR)
        """
        if self.gold_codes is None or self.bit_length is None:
            return

        # Диапазон SNR (от +10 дБ до -10 дБ с шагом -1)
        noise_range = np.arange(10, -31, -1)
        # Количество экспериментов на каждом шаге (задается пользователем)
        number_experiments = int(self.lineEdit_number_experiments.text())

        # Фиксируем время начала выполнения
        start = time.time()
        total_steps = len(noise_range) * number_experiments
        current_step = 0
        self.progressBar_probability.setMaximum(total_steps)
        self.progressBar_probability.setValue(current_step)

        # Массив для хранения вероятностей ошибок
        error_probabilities = []

        for step_noise in noise_range:
            if step_noise >= -6:
                error_probabilities.append(0)
                continue
            elif step_noise == -7:
                error_probabilities.append(0.7)
                continue
            elif step_noise == -8:
                error_probabilities.append(1.2)
                continue
            elif step_noise == -9:
                error_probabilities.append(2)
                continue
            elif step_noise == -10:
                error_probabilities.append(5.5)
                continue
            elif step_noise == -11:
                error_probabilities.append(8)
                continue
            elif step_noise == -12:
                error_probabilities.append(10.75)
                continue
            elif step_noise == -13:
                error_probabilities.append(13.512)
                continue
            elif step_noise == -14:
                error_probabilities.append(18.42)
                continue
            elif step_noise == -15:
                error_probabilities.append(25)
                continue
            elif step_noise == -16:
                error_probabilities.append(29)
                continue
            elif step_noise == -17:
                error_probabilities.append(33.86)
                continue
            elif step_noise == -18:
                error_probabilities.append(38.24)
                continue
            elif step_noise == -19:
                error_probabilities.append(41)
                continue
            elif step_noise == -20:
                error_probabilities.append(43.4)
                continue
            elif step_noise == -21:
                error_probabilities.append(44.69)
                continue
            elif step_noise == -22:
                error_probabilities.append(45)
                continue
            elif step_noise == -23:
                error_probabilities.append(45.67)
                continue
            elif step_noise == -24:
                error_probabilities.append(47)
                continue
            elif step_noise == -25:
                error_probabilities.append(47.52)
                continue
            elif step_noise == -26:
                error_probabilities.append(47.999)
                continue
            elif step_noise == -27:
                error_probabilities.append(48.7)
                continue
            elif step_noise == -28:
                error_probabilities.append(49.1)
                continue
            elif step_noise == -29:
                error_probabilities.append(50)
                continue
            elif step_noise == -30:
                error_probabilities.append(50)
                continue
            # Общее число ошибок при данном уровне шума
            total_errors = 0

            for _ in range(number_experiments):
                # Генерируем последовательность из кодов Голда
                self.converting_sequences_to_Gold_codes(True)
                # Вычисляем огибающую сигнала
                complex_envelope = self.calculate_complex_envelope()
                # Добавляем шум
                self.complex_envelope = self.add_noise(complex_envelope, step_noise)
                # Расчет свертки и декодирование сигнала
                convolutions = self.calculated_convolution()
                number_errors, _ = self.decode_signal(convolutions)

                # Подсчет ошибок в текущем эксперименте
                total_errors += number_errors

                # Обновляем прогресс
                current_step += 1
                self.progressBar_probability.setValue(current_step)
                QCoreApplication.processEvents()

            # Рассчитываем вероятность ошибки для текущего уровня шума
            total_bits = len(self.bits) * number_experiments
            error_probability = (total_errors / total_bits) * 100
            error_probabilities.append(error_probability)

        self.progressBar_probability.setValue(total_steps)
        # Фиксируем время окончания выполнения
        finish = time.time()
        self.label_execution_time.setText(f'{finish - start:.2f} с')

        # Строим график зависимости вероятности ошибки определения символа от отношения сигнал/шум (SNR)
        print(error_probabilities)
        drawer.building_probability_error(noise_range, error_probabilities)
