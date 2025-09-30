"""
Функция для расчета позиций стратегий на основе mask_df.

Этот модуль содержит функцию calculate_strategy_positions, которая позволяет
рассчитывать позиции для размещения стратегий с равномерным шагом, основанным
на диапазоне значений в колонке 'stroka' исходной маски.

Основная логика:
1. Анализируется диапазон значений в колонке 'stroka' маски (max - min)
2. Вычисляется шаг как (диапазон + 1)
3. Создается список позиций с заданным количеством элементов

Пример реального использования с проектными данными:
- mask.xlsx содержит stroka от 1000 до 1015 (диапазон = 15)
- шаг = 15 + 1 = 16
- для 5 стратегий с начальной позицией 1000: [1000, 1016, 1032, 1048, 1064]
"""

import pandas as pd
from typing import List


def calculate_strategy_positions(
    start_number: int,
    rows_between: int, 
    num_strategies: int,
    mask_df: pd.DataFrame
) -> List[int]:
    """
    Расчитывает список позиций для размещения стратегий.
    
    Параметры:
    ----------
    start_number : int
        Начальное число (первое число в результирующем списке)
    rows_between : int  
        Количество строк между стратегиями (не используется в текущей реализации)
    num_strategies : int
        Количество стратегий для которых нужно рассчитать позиции
    mask_df : pd.DataFrame
        DataFrame с колонкой 'stroka', содержащей номера строк
        
    Возвращает:
    -----------
    List[int]
        Список позиций, где первое число = start_number,
        шаг = (max(stroka) - min(stroka)) + 1,
        количество элементов = num_strategies
        
    Пример:
    -------
    >>> mask_df = pd.DataFrame({'stroka': [10, 15, 20]})
    >>> calculate_strategy_positions(100, 5, 3, mask_df)
    [100, 111, 122]
    
    Расчет:
    - max(stroka) = 20, min(stroka) = 10
    - диапазон = 20 - 10 = 10
    - шаг = 10 + 1 = 11
    - позиции: 100, 100+11=111, 111+11=122
    """
    if mask_df.empty:
        raise ValueError("mask_df не может быть пустым")
        
    if 'stroka' not in mask_df.columns:
        raise ValueError("mask_df должен содержать колонку 'stroka'")
        
    if num_strategies <= 0:
        raise ValueError("Количество стратегий должно быть больше 0")
    
    # Шаг 4: Расчет количества строк в финальной маске
    stroka_values = mask_df['stroka'].dropna()
    if stroka_values.empty:
        raise ValueError("Колонка 'stroka' не содержит валидных значений")
    
    max_stroka = int(stroka_values.max())
    min_stroka = int(stroka_values.min())
    mask_range = max_stroka - min_stroka
    
    # Шаг 5-7: Создание списка позиций
    step = mask_range + 1 + rows_between
    positions = []
    
    for i in range(num_strategies):
        position = start_number + (i * step)
        positions.append(position)
    
    return positions


# Примеры использования для тестирования
if __name__ == "__main__":
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ ФУНКЦИИ calculate_strategy_positions")
    print("=" * 70)
    
    # Пример 1: Тестовые данные
    print("\n1. ТЕСТОВЫЙ ПРИМЕР:")
    print("-" * 50)
    test_mask_df = pd.DataFrame({
        'stroka': [10, 15, 20, 12, 18],
        'row_alias': ['row1', 'row2', 'row3', 'row4', 'row5']
    })
    
    print("Тестовый mask_df:")
    print(test_mask_df)
    print(f"\nmin(stroka) = {test_mask_df['stroka'].min()}")
    print(f"max(stroka) = {test_mask_df['stroka'].max()}")
    print(f"диапазон = {test_mask_df['stroka'].max() - test_mask_df['stroka'].min()}")
    print(f"шаг = {test_mask_df['stroka'].max() - test_mask_df['stroka'].min() + 1}")
    
    result1 = calculate_strategy_positions(
        start_number=100,
        rows_between=5,  # этот параметр пока не используется в логике
        num_strategies=4,
        mask_df=test_mask_df
    )
    
    print(f"\nРезультат: {result1}")
    print("Проверка шагов:")
    for i in range(1, len(result1)):
        step = result1[i] - result1[i-1]
        print(f"  {result1[i-1]} -> {result1[i]} (шаг: {step})")
    
    # Пример 2: Данные похожие на реальные из проекта
    print("\n\n2. ПРИМЕР С РЕАЛЬНЫМИ ДАННЫМИ:")
    print("-" * 50)
    real_like_mask_df = pd.DataFrame({
        'row_alias': ['base_long', 'base_short', 'filter_1', 'filter_2', 'exit_long'],
        'stroka': [1000, 1001, 1003, 1004, 1015],
        'Sec 0': [1, 1, 1, 1, 1]
    })
    
    print("Маска похожая на реальную:")
    print(real_like_mask_df)
    print(f"\nmin(stroka) = {real_like_mask_df['stroka'].min()}")
    print(f"max(stroka) = {real_like_mask_df['stroka'].max()}")
    print(f"диапазон = {real_like_mask_df['stroka'].max() - real_like_mask_df['stroka'].min()}")
    print(f"шаг = {real_like_mask_df['stroka'].max() - real_like_mask_df['stroka'].min() + 1}")
    
    result2 = calculate_strategy_positions(
        start_number=2000,
        rows_between=0,
        num_strategies=6,
        mask_df=real_like_mask_df
    )
    
    print(f"\nРезультат для 6 стратегий начиная с 2000: {result2}")
    print("Проверка шагов:")
    for i in range(1, len(result2)):
        step = result2[i] - result2[i-1]
        print(f"  {result2[i-1]} -> {result2[i]} (шаг: {step})")
    
    # Пример 3: Различные сценарии
    print("\n\n3. ДОПОЛНИТЕЛЬНЫЕ СЦЕНАРИИ:")
    print("-" * 50)
    
    scenarios = [
        ("Одна стратегия", 5000, 1),
        ("Две стратегии", 3000, 2), 
        ("Много стратегий", 1, 10)
    ]
    
    for name, start, count in scenarios:
        result = calculate_strategy_positions(start, 0, count, real_like_mask_df)
        print(f"{name}: start={start}, count={count}")
        print(f"  Результат: {result}")
        if len(result) > 1:
            print(f"  Шаг: {result[1] - result[0]}")
        print()
