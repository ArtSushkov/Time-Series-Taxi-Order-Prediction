# === Импорт библиотек ===

# Манипуляции с данными и анализ
import numpy as np
import pandas as pd

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
from phik.report import plot_correlation_matrix
import phik  # ← ДОБАВИТЬ ЭТУ СТРОКУ

# Статистическое моделирование
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess  # ← ДОБАВИТЬ ЭТУ СТРОКУ


# === Собственные функции ===

def phik_correlation_matrix(df, target_col=None, threshold=0.9, output_interval_cols=True, interval_cols=None, cell_size=1.1):
    """Строит матрицу корреляции Фи-К (включая целевую переменную) и возвращает корреляции с целевой.

    Args:
        df (pd.DataFrame): DataFrame с данными для анализа
        target_col (str): Название столбца с целевой переменной
        threshold (float): Порог для выделения значимых корреляций (0.9 по умолчанию)
        output_interval_cols (bool): Возвращать ли список числовых непрерывных столбцов
        interval_cols (list): Список числовых непрерывных столбцов (если None, будет определен автоматически)
        cell_size (float): Дюйм на ячейку

    Returns:
        tuple: (correlated_pairs, interval_cols, phi_k_with_target) где:
            - correlated_pairs: DataFrame с парами коррелирующих признаков
            - interval_cols: Список числовых непрерывных столбцов (если output_interval_cols=True)
            - phi_k_with_target: Series с корреляциями признаков с целевой переменной

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phik import phik_matrix
        >>>
        >>> # Создаем тестовые данные
        >>> data = {
        ...     'price': [100, 200, 150, 300],  # Целевая переменная
        ...     'mileage': [50, 100, 75, 120],
        ...     'brand': ['A', 'B', 'A', 'C'],
        ...     'engine': [1.6, 2.0, 1.8, 2.5]
        ... }
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Анализ корреляций с ручным заданием числовых столбцов
        >>> result = phik_correlation_matrix(df, target_col='price', threshold=0.3, interval_cols=['mileage', 'engine'])
        >>>
        >>> # Получаем результаты:
        >>> correlated_pairs = result[0]  # Пары коррелирующих признаков
        >>> interval_cols = result[1]     # Числовые непрерывные столбцы
        >>> phi_k_with_target = result[2] # Корреляции с ценой
        >>>
        >>> print("Корреляции с ценой:")
        >>> print(phi_k_with_target.sort_values(ascending=False))
    """

    # Определение числовых непрерывных столбцов (если не заданы вручную)
    if interval_cols is None:
        interval_cols = [
            col for col in df.select_dtypes(include=["number"]).columns
            if (df[col].nunique() > 50) or ((df[col] % 1 != 0).any())
        ]

    # Расчет полной матрицы корреляции (включая целевую переменную)
    phik_matrix = df.phik_matrix(interval_cols=interval_cols).round(2)

    # Получение корреляций с целевой переменной
    phi_k_with_target = None
    if target_col is not None and target_col in phik_matrix.columns:
        phi_k_with_target = phik_matrix[target_col].copy()
        # Удаляем корреляцию целевой с собой (всегда 1.0)
        phi_k_with_target.drop(target_col, inplace=True, errors='ignore')

    # Динамическое определение размера фигуры для подстройки размера ячеек
    num_cols = len(phik_matrix.columns)
    num_rows = len(phik_matrix.index)
    cell_size = cell_size  # Дюймов на ячейку
    figsize = (num_cols * cell_size, num_rows * cell_size)

    # Визуализация матрицы
    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map="Greens",
        title=r"Матрица корреляции $\phi_K$",
        fontsize_factor=1,
        figsize=figsize
    )
    plt.tight_layout()
    plt.show()

    # Фильтрация значимых корреляций (исключая целевую из пар)
    close_to_one = phik_matrix[phik_matrix >= threshold]
    close_to_one = close_to_one.where(
        np.triu(np.ones(close_to_one.shape), k=1).astype(bool)
    )

    # Удаление строк/столбцов с целевой переменной для анализа пар признаков
    if target_col is not None:
        close_to_one.drop(target_col, axis=0, inplace=True, errors='ignore')
        close_to_one.drop(target_col, axis=1, inplace=True, errors='ignore')

    # Преобразование в длинный формат
    close_to_one_stacked = close_to_one.stack().reset_index()
    close_to_one_stacked.columns = ["признак_1", "признак_2", "корреляция"]
    close_to_one_stacked = close_to_one_stacked.dropna(subset=["корреляция"])

    # Классификация корреляций
    def classify_correlation(corr):
        if corr >= 0.9: return "Очень высокая"
        elif corr >= 0.7: return "Высокая"
        elif corr >= 0.5: return "Заметная"
        elif corr >= 0.3: return "Умеренная"
        elif corr >= 0.1: return "Слабая"
        return "-"

    close_to_one_stacked["класс_корреляции"] = close_to_one_stacked["корреляция"].apply(
        classify_correlation
    )
    close_to_one_sorted = close_to_one_stacked.sort_values(
        by="корреляция", ascending=False
    ).reset_index(drop=True)

    if len(close_to_one_sorted) == 0 and threshold >= 0.9:
        print("\033[1mМультиколлинеарность между парами входных признаков отсутствует\033[0m")

    # Формирование результата
    result = [close_to_one_sorted]
    if output_interval_cols:
        result.append(interval_cols)
    if target_col is not None:
        result.append(phi_k_with_target)
    elif output_interval_cols:
        result.append(None)

    return tuple(result)


def vif(X, fig_height=12, font_size=12):
    """Строит столбчатую диаграмму с коэффициентами инфляции дисперсии (VIF) для всех входных признаков.

    Args:
        X (pd.DataFrame): DataFrame с входными признаками для анализа.
        fig_height (int): Размер высоты графика (по умолчанию 12)
        font_size (int): Размер шрифта для текстовых элементов графика (по умолчанию 12).

    Notes:
        - Коэффициент инфляции дисперсии (VIF) показывает степень мультиколлинеарности между признаками.
        - График отображается напрямую через matplotlib.

    Example:
        Пример использования функции:

        >>> import pandas as pd
        >>> from statsmodels.stats.outliers_influence import variance_inflation_factor
        >>> import statsmodels.api as sm
        >>>
        >>> # Создаем тестовый датафрейм
        >>> data = pd.DataFrame({
        ...     'feature1': [1, 2, 3, 4, 5],
        ...     'feature2': [2, 4, 6, 8, 10],  # Полностью коррелирует с feature1
        ...     'feature3': [3, 6, 9, 12, 15]   # Частично коррелирует
        ... })
        >>>
        >>> # Вызываем функцию для анализа VIF
        >>> vif(data)
        >>>
        >>> # В результате будет показан график с VIF для каждого признака
        >>> # (feature2 будет иметь очень высокий VIF из-за полной корреляции с feature1)
    """
    # Кодируем категориальные признаки
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

    # Добавляем константу для корректного расчета VIF
    X_with_const = sm.add_constant(X_encoded)

    # Вычисляем VIF для всех признаков, кроме константы (индексы начинаются с 1)
    vif = [variance_inflation_factor(X_with_const.values, i)
           for i in range(1, X_with_const.shape[1])]  # Исключаем константу (0-й столбец)

    # Построение графика с использованием исходных названий признаков (без константы)
    num_features = X_encoded.shape[1]
    fig_width = num_features * 1.2
    fig_height = fig_height

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.barplot(x=X_encoded.columns, y=vif)

    # Настройки графика
    ax.set_ylabel('VIF', fontsize=font_size)
    ax.set_xlabel('Входные признаки', fontsize=font_size)
    plt.title('Коэффициент инфляции дисперсии для входных признаков (VIF)', fontsize=font_size)

    # Метки на осях
    plt.xticks(rotation=90, ha='right', fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Добавляем значения на столбцы (опционально)
    # ax.bar_label(ax.containers[0], fmt='%.2f', padding=3, fontsize=font_size)

    plt.tight_layout()
    plt.show()


def analyze_residuals(y_test, y_pred, units='ед. измерения', bins=30, figsize=(20, 6), 
                     title=None, lowess=False, lowess_frac=0.3):
    """Анализирует остатки модели и визуализирует результаты.

    Создаёт графики для анализа распределения остатков:
    - Гистограмма остатков с KDE-кривой
    - График остатков vs прогнозных значений (с опциональным LOWESS-сглаживанием)
    - График остатков vs номера наблюдения

    Args:
        y_test (Union[np.ndarray, list]): Вектор истинных значений.
        y_pred (Union[np.ndarray, list]): Вектор прогнозных значений.
        units (str, optional): Единицы измерения для осей. По умолчанию 'ед. измерения'.
        bins (int, optional): Количество бинов для гистограммы. По умолчанию 30.
        figsize (Tuple[float, float], optional): Размер фигуры (ширина, высота). 
            По умолчанию (20, 6).
        title (str, optional): Общий заголовок для графика. По умолчанию None.
        lowess (bool, optional): Включить LOWESS-сглаживание на графике остатков. 
            По умолчанию False.
        lowess_frac (float, optional): Параметр сглаживания для LOWESS (0-1). 
            По умолчанию 0.3.

    Returns:
        None: Выводит сетку графиков анализа остатков.

    Example:
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.model_selection import train_test_split
        >>> # Создаем синтетические данные
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 1)
        >>> y = 2 * X.squeeze() + np.random.normal(0, 0.1, 100)
        >>> # Разделяем на train/test
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> # Обучаем модель
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
        >>> # Анализируем остатки с LOWESS
        >>> analyze_residuals(y_test, y_pred, units='метры', title='Анализ остатков', 
        ...                  lowess=True, lowess_frac=0.25)
    """

    # Преобразование входных данных в numpy-массивы
    y_test = np.array(y_test).ravel()
    y_pred = np.array(y_pred).ravel()

    # Проверка совпадения размерностей
    if len(y_test) != len(y_pred):
        raise ValueError("Длины y_test и y_pred должны совпадать")

    # Рассчитываем остатки
    error = y_test - y_pred

    # Создаем фигуру
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Добавляем общий заголовок
    if title:
        fig.suptitle(title)

    # Гистограмма остатков
    sns.histplot(error, bins=bins, kde=True, ax=axes[0])
    axes[0].axvline(
        x=0,
        color='r',
        linestyle='--',
        label='Нулевая линия'
    )
    axes[0].axvline(
        x=error.mean(),
        color='b',
        linestyle='--',
        label='Среднее'
    )
    axes[0].axvline(
        x=np.median(error),
        color='m',
        linestyle='-.',
        label='Медиана'
    )
    axes[0].axvline(
        x=error.mean() + error.std(),
        color='g',
        linestyle='--',
        label='Среднее ± std'
    )
    axes[0].axvline(
        x=error.mean() - error.std(),
        color='g',
        linestyle='--'
    )
    axes[0].set_title('Гистограмма остатков')
    axes[0].set_xlabel(f'Остатки ({units})')
    axes[0].set_ylabel('Частота')
    axes[0].legend()

    # Диаграмма рассеяния с возможностью LOWESS
    sns.scatterplot(x=y_pred, y=error, ax=axes[1], alpha=0.5)
    
    if lowess:
        # Применяем LOWESS-сглаживание
        lowess_sm = sm.nonparametric.lowess(error, y_pred, frac=lowess_frac)
        axes[1].plot(lowess_sm[:, 0], lowess_sm[:, 1], color='orange', 
                    linewidth=2, label=f'LOWESS (frac={lowess_frac})')
    
    axes[1].axhline(
        y=0,
        color='r',
        linestyle='--',
        label='Нулевая линия'
    )
    axes[1].axhline(
        y=error.mean(),
        color='b',
        linestyle='--',
        label='Среднее'
    )
    axes[1].axhline(
        y=np.median(error),
        color='m',
        linestyle='-.',
        label='Медиана'
    )
    axes[1].axhline(
        y=error.mean() + error.std(),
        color='g',
        linestyle='--',
        label='Среднее ± std'
    )
    axes[1].axhline(
        y=error.mean() - error.std(),
        color='g',
        linestyle='--'
    )
    axes[1].set_title('Диаграмма рассеяния прогнозов и остатков')
    axes[1].set_xlabel(f'Прогнозы ({units})')
    axes[1].set_ylabel(f'Остатки ({units})')
    axes[1].legend()

    plt.tight_layout()
    plt.show()