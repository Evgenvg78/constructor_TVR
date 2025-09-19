from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Any
import pandas as pd
import numpy as np
import re
import csv
import io

@dataclass
class TVRParams:
    # Разделитель в первой строке (строка с названиями колонок).
    # Если None — берём ПЕРВЫЙ символ первой строки файла как разделитель.
    header_sep: Optional[str] = None            # пример: '♪'
    # Разделитель между 1-м и 2-м числом и началом значения в данных.
    triple_sep: str = ' '                       # в TVR — пробел
    # Десятичный разделитель в TVR (в текстовом файле).
    tvr_decimal: str = '.'                      # важно: в TVR — точка
    # Десятичный разделитель в Excel/CSV (на случай, если будете сохранять CSV).
    excel_decimal: str = ','                    # справочно; для .xlsx не критично
    encoding: str = 'utf-8'
    # Явное кол-во колонок (если хотите форсировать). Обычно не требуется.
    n_columns: Optional[int] = None
    meta_sheet: str = "TVR_META"
    data_sheet: str = "TVR"

_num_like_re = re.compile(r'^\s*[+-]?\d+(?:[.,]\d+)?\s*$')

def _detect_header_and_columns(header_line: str, params: TVRParams) -> Tuple[str, List[str]]:
    """Определяем разделитель и список имён колонок из первой строки TVR2."""
    if not header_line:
        raise ValueError("Пустой файл TVR2: не найдена первая строка.")

    sep = params.header_sep if params.header_sep is not None else header_line[0]

    # В формате TVR2 обычно две «пустые позиции» под stroka/stolbec:
    # пример: '♪♪Col1♪Col2♪Col3'
    parts = header_line.strip().split(sep)
    # Если строка начинается с разделителя, split даст первый элемент ''.
    # Практика TVR2: первые две позиции пустые → берём с индекса 2.
    if len(parts) >= 3 and parts[0] == '' and parts[1] == '':
        col_names = parts[2:]
    else:
        # Фоллбек: предполагаем, что первые два — служебные, берём с индекса 2.
        col_names = parts[2:] if len(parts) > 2 else []

    if params.n_columns is not None:
        # подрезаем/дополняем до нужной длины
        if len(col_names) < params.n_columns:
            col_names = col_names + [f"Col{i}" for i in range(len(col_names)+1, params.n_columns+1)]
        else:
            col_names = col_names[:params.n_columns]

    return sep, col_names

def _parse_tvr2_triplets(lines: List[str], triple_sep: str) -> pd.DataFrame:
    """Парсит строки вида: 'row col value...' (value может содержать пробелы)."""
    rows, cols, vals = [], [], []
    for raw in lines:
        s = raw.rstrip('\n\r')
        if not s:
            continue
        # split только на 3 части: row, col, остальное как value
        parts = s.split(triple_sep, 2)
        if len(parts) < 3:
            # попробуем «жёстко» распарсить по пробелам, если triple_sep не сработал
            parts = s.split(' ', 2)
            if len(parts) < 3:
                # пустая ячейка/некорректная строка — пропустим
                continue
        try:
            r = int(parts[0])
            c = int(parts[1])
        except ValueError:
            # некорректная строка — пропуск
            continue
        v = parts[2]
        rows.append(r); cols.append(c); vals.append(v)
    return pd.DataFrame({'stroka': rows, 'stolbec': cols, 'value': vals})

def _maybe_to_number(s: Any) -> Any:
    """Пытаемся превратить строку в число (int/float). Иначе возвращаем исходное."""
    if s is None:
        return s
    if isinstance(s, (int, float, np.integer, np.floating)):
        return s
    if isinstance(s, str):
        t = s.strip()
        if _num_like_re.match(t):
            # заменим запятую на точку для корректного float()
            t2 = t.replace(',', '.')
            try:
                f = float(t2)
                # целое?
                if f.is_integer():
                    return int(f)
                return f
            except Exception:
                return s
    return s

def tvr2_to_excel(
    tvr2_path: str,
    xlsx_path: str,
    *,
    params: Optional[TVRParams] = None,
    header_sep: str | None = None,
    wide_fill_missing: bool = True,
    convert_numbers: bool = True
) -> None:
    """
    Конвертер TVR2 → XLSX.
    - Читает шапку, детектит разделитель, формирует список колонок.
    - Парсит тройки (stroka, stolbec, value).
    - Пивот в «широкую» таблицу: stroka + имена колонок.
    - Пишет в Excel (лист params.data_sheet) + метаданные (лист params.meta_sheet).
    """
    params = params or TVRParams()
    if header_sep is not None:
        params.header_sep = header_sep

    with open(tvr2_path, 'r', encoding=params.encoding) as f:
        header_line = f.readline().rstrip('\n\r')
        sep, col_names = _detect_header_and_columns(header_line, params)
        raw_lines = f.readlines()

    trip = _parse_tvr2_triplets(raw_lines, params.triple_sep)

    # Гарантируем наличие всех столбцов (1..N)
    n_cols = params.n_columns or len(col_names)
    if n_cols == 0:
        # определим по данным
        n_cols = int(trip['stolbec'].max()) if not trip.empty else 0

    if wide_fill_missing and n_cols > 0 and not trip.empty:
        # добавим «паттерн», чтобы пивот создал все столбцы
        pattern = pd.DataFrame({
            'stroka': [10**9]*n_cols,                         # фиктивная строка
            'stolbec': list(range(1, n_cols+1)),
            'value': ['__pattern__']*n_cols
        })
        trip2 = pd.concat([trip, pattern], ignore_index=True)
    else:
        trip2 = trip

    if trip2.empty:
        wide = pd.DataFrame({'stroka': []})
        for i in range(n_cols):
            wide[col_names[i] if i < len(col_names) else f"Col{i+1}"] = []
    else:
        wide_pivot = trip2.pivot_table(index='stroka', columns='stolbec', values='value', aggfunc='last')
        # убираем фиктивную строку, если добавляли pattern
        if wide_fill_missing:
            wide_pivot = wide_pivot[wide_pivot.index != 10**9]

        # упорядочим столбцы и переименуем их по col_names
        full_cols = list(range(1, n_cols+1))
        wide_pivot = wide_pivot.reindex(columns=full_cols)
        rename_map = {i+1: (col_names[i] if i < len(col_names) else f"Col{i+1}") for i in range(n_cols)}
        wide_pivot = wide_pivot.rename(columns=rename_map).reset_index().rename(columns={'stroka': 'stroka'})
        wide = wide_pivot

    if convert_numbers and not wide.empty:
        for c in wide.columns:
            if c == 'stroka':
                # stroka — точно целые
                wide[c] = pd.to_numeric(wide[c], errors='ignore', downcast='integer')
                continue
            wide[c] = wide[c].map(_maybe_to_number)

    # Записываем в Excel
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        wide.to_excel(writer, sheet_name=params.data_sheet, index=False)
        # Метаданные (для обратной конвертации)
        meta = {
            'header_sep': sep,
            'triple_sep': params.triple_sep,
            'tvr_decimal': params.tvr_decimal,
            'excel_decimal': params.excel_decimal,
            'encoding': params.encoding,
            'n_columns': n_cols,
            'column_names': '|'.join([str(x) for x in col_names])  # положим через |
        }
        meta_df = pd.DataFrame(list(meta.items()), columns=['key', 'value'])
        meta_df.to_excel(writer, sheet_name=params.meta_sheet, index=False)

        # немного косметики: заморозим верхнюю строку и первый столбец
        ws = writer.sheets[params.data_sheet]
        ws.freeze_panes = "B2"

def _format_tvr_value(x: Any, decimal: str) -> str:
    """Привести значение к строке для TVR: число → с заданным десятичным разделителем, остальное — как есть."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ''):
        return ''
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        # Use positional formatting to avoid scientific notation and keep significant decimals
        s = np.format_float_positional(float(x), trim='-')
        if s == '-0':
            s = '0'
        if decimal != '.':
            s = s.replace('.', decimal)
        return s
    s = str(x).strip()
    if not s:
        return ''
    # Если это строковое число с запятой — нормализуем к нужному decimal
    if _num_like_re.match(s):
        s2 = s.replace(',', '.')
        if decimal != '.':
            s2 = s2.replace('.', decimal)
        return s2
    normalized = s.replace(',', '.')
    try:
        val = float(normalized)
    except ValueError:
        return s
    formatted = np.format_float_positional(val, trim='-')
    if formatted == '-0':
        formatted = '0'
    if decimal != '.':
        formatted = formatted.replace('.', decimal)
    return formatted

def excel_to_tvr(
    xlsx_path: str,
    tvr2_path: str,
    *,
    params: Optional[TVRParams] = None,
    header_sep: str | None = None,
    skip_empty: bool = True
) -> None:
    """
    Конвертер XLSX → TVR2.
    - Читает «широкую» таблицу (лист params.data_sheet).
    - Если есть лист метаданных, использует его (разделители, имена колонок).
    - Формирует первую строку формата TVR2 и тройки 'row col value'.
    - Нормализует десятичный разделитель под TVR (обычно '.').

    Формат первой строки (как в твоём коде): два начальных разделителя + названия колонок:
    '♪♪Col1♪Col2♪Col3'
    """
    params = params or TVRParams()

    # Попробуем прочитать метаданные
    meta = {}
    try:
        meta_df = pd.read_excel(xlsx_path, sheet_name=params.meta_sheet, dtype=str, engine='openpyxl')
        if {'key', 'value'}.issubset(set(meta_df.columns)):
            meta = dict(zip(meta_df['key'], meta_df['value']))
    except Exception:
        pass

    meta_header_sep = meta.get('header_sep')
    sep_candidates = [params.header_sep, header_sep, meta_header_sep, '♪']
    header_sep = next((str(sep) for sep in sep_candidates if isinstance(sep, str) and sep), '♪')

    triple_sep = params.triple_sep or meta.get('triple_sep', ' ')
    tvr_decimal = params.tvr_decimal or meta.get('tvr_decimal', '.')
    excel_decimal = params.excel_decimal or meta.get('excel_decimal', ',')

    # Данные
    df = pd.read_excel(xlsx_path, sheet_name=params.data_sheet, dtype=str, engine='openpyxl')

    # Определим столбец stroka
    if 'stroka' in df.columns:
        wide = df.copy()
    else:
        # Если столбца нет, создадим индексацию строк 1..N
        wide = df.copy()
        wide.insert(0, 'stroka', range(1, len(wide) + 1))

    # Колонки данных = всё, кроме 'stroka'
    data_cols = [c for c in wide.columns if c != 'stroka']

    # Если в метаданных сохранены исходные имена — используем их (сохраняем порядок)
    if 'column_names' in meta and isinstance(meta['column_names'], str):
        saved_cols = meta['column_names'].split('|')
        # отфильтруем по тем, что есть в файле; если не совпадает — берём как в файле
        if set(saved_cols).issubset(set(data_cols)):
            data_cols = saved_cols

    # Первая строка TVR2: два разделителя подряд + названия колонок через разделитель
    header_line = header_sep + header_sep + header_sep.join(data_cols) + '\n'

    # Сформируем тройки
    lines = [header_line]
    for _, row in wide.iterrows():
        try:
            r = int(str(row['stroka']).strip())
        except Exception:
            # пропустим странные «stroka»
            continue

        for j, col_name in enumerate(data_cols, start=1):
            val_raw = row[col_name]
            if skip_empty and (val_raw is None or (isinstance(val_raw, float) and pd.isna(val_raw)) or str(val_raw).strip() == ''):
                continue

            # нормализуем число под TVR-десятичный разделитель
            val_str = _format_tvr_value(val_raw, decimal=tvr_decimal)

            # Соберём строку: row<sp>col<sp>value
            lines.append(f"{r}{triple_sep}{j}{triple_sep}{val_str}\n")

    with open(tvr2_path, 'w', encoding=params.encoding, newline='\n') as f:
        f.writelines(lines)

# Backward compatibility
excel_to_tvr2 = excel_to_tvr
