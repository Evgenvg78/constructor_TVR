# Исправление проблемы с относительными ссылками на фильтры

## Проблема

При парсинге стратегии из TVR файла, ссылки на фильтры в столбцах `InL1`, `InL2`, `OutL1`, `OutL2` парсятся как **относительные** (например, "begin+2", "begin+5").

Однако, при создании файла маски из спарсированной стратегии эти ссылки записывались как **абсолютные номера строк** (например, "4990, 4993"), взятые из исходного TVR файла.

В результате, при генерации нового TVR файла из маски, все ссылки на фильтры оказывались неправильными - они указывали на строки из исходного файла, а не на относительные позиции в новом файле.

## Решение

Решение состоит из трех компонентов:

### 1. Утилиты для конвертации ссылок

В `src/tvr_service/templates/strategy_template.py` добавлены функции:

- **`convert_absolute_to_relative_reference(value, base_stroka)`** - конвертирует абсолютные номера строк в относительный формат (begin+X)
- **`_resolve_relative_references(value, context)`** - конвертирует относительные ссылки обратно в абсолютные при генерации

### 2. Автоматическая конвертация при создании маски

В `src/tvr_service/templates/mask_template.py` модифицирована функция `build_template_from_mask_df`:

- Добавлен параметр `convert_references_to_relative` (по умолчанию `True`)
- Добавлен параметр `reference_columns` для указания столбцов с ссылками
- При чтении маски автоматически конвертирует абсолютные ссылки в относительные

### 3. Автоматическая конвертация при генерации

В `src/tvr_service/templates/strategy_template.py` модифицирована функция `_resolve_value`:

- При генерации TVR из шаблона автоматически распознает относительные ссылки
- Конвертирует их в абсолютные с учетом значения `start` для каждой стратегии

### 4. Вспомогательная функция для создания масок

В `src/tvr_service/generator/layout.py` добавлена функция:

- **`build_mask_dataframe_from_layout(layout, compiled, source_dataframe, ...)`** - создает маску из layout с автоматической конвертацией ссылок

## Использование

### Создание маски с относительными ссылками

```python
from src.tvr_service.generator import build_mask_dataframe_from_layout

# Создаем маску из layout
mask_df = build_mask_dataframe_from_layout(
    layout=layout,
    compiled=compiled,
    source_dataframe=tvr_full,
    target_base_stroka=1000,
)

# Ссылки автоматически конвертированы в формат "begin+2", "begin+5" и т.д.
```

### Создание шаблона из маски

```python
from src.tvr_service.templates import build_template_from_mask_file

# По умолчанию convert_references_to_relative=True
mask_result = build_template_from_mask_file(
    'mask.xlsx',
    row_alias_column='row_alias',
    marker='#',
)

# Шаблон готов к использованию
```

### Генерация TVR из шаблона

```python
from src.tvr_service.generator import StrategyGenerator

generator = StrategyGenerator(
    template=mask_result.template,
    strategy_column='strategy_id',
    start_column='start',
)

# Генерируем TVR
tvr_df = generator.generate(config_table)

# Относительные ссылки автоматически конвертированы в абсолютные
# с учетом значения 'start' для каждой стратегии
```

## Примеры конвертации

### До (абсолютные ссылки)
```
InL1: "4990, 4993"
InL2: "4991"
```

### После (относительные ссылки)
```
InL1: "begin+10, begin+13"
InL2: "begin+11"
```

### ✨ НОВОЕ: Поддержка пересборки layout

Когда вы пересобираете layout (перемещаете фильтры на другие offset), **ссылки автоматически обновляются**:

```
Исходный layout:
  - Фильтр на offset=10, ссылка "begin+10"
  
После пересборки (фильтр перемещен на offset=2):
  - Фильтр на offset=2, ссылка автоматически обновлена на "begin+2"
```

**Реализация:** `build_mask_dataframe_from_layout` строит маппинг `original_stroka -> new_offset` и использует его при конвертации ссылок.

### При генерации с start=1000
```
InL1: "1010, 1013"
InL2: "1011"
```

## Обратная совместимость

- Если маска содержит уже относительные ссылки, они обрабатываются корректно
- Если маска содержит абсолютные ссылки и `convert_references_to_relative=False`, они останутся абсолютными (старое поведение)
- По умолчанию включена автоматическая конвертация для новых проектов

## Изменения в коде

1. `src/tvr_service/templates/strategy_template.py` - добавлены функции конвертации
2. `src/tvr_service/templates/mask_template.py` - добавлена конвертация при создании шаблона
3. `src/tvr_service/generator/layout.py` - добавлена вспомогательная функция `build_mask_dataframe_from_layout` с поддержкой пересборки
4. `src/tvr_service/generator/layout.py` - добавлена функция `_convert_references_with_mapping` для обновления ссылок при пересборке
5. `docs/strategy_mask_demo_v2.ipynb` - обновлен для демонстрации новой функциональности

## Ключевые особенности

✅ **Автоматическая конвертация** - ссылки конвертируются автоматически  
✅ **Поддержка пересборки** - при изменении offset ссылки автоматически обновляются  
✅ **Обратная совместимость** - старые маски продолжают работать  
✅ **Умный fallback** - если ссылка указывает на строку вне layout, используется базовая конвертация
