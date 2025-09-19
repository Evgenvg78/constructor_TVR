
import pandas as pd
from moexalgo import session, Market, Ticker, CandlePeriod
import numpy as np
import csv


def TVR_transform(TVR, varyant_types, type='Go', sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'):
  if type == 'Go':
      col_count = 'INITIALMARGIN'
  else:
      col_count = 'full_price'

  # модуль получения данных с биржи
  # загрузка данных по инструментам !!!!!! не работает вне торговой сессии !!!!

  # forts_tickers = Market('FO').tickers()
  # sec_data_list = []
  # for i in range(len(forts_tickers)):
  #   # print(i)
  #   name = forts_tickers.iloc[i, 0]
  #   sec = Ticker(name).info()
  #   sec = sec.T
  #   sec_data_list.append(sec)
  # sec_data_file = '/content/drive/MyDrive/work_data/TVR/sec_tvr.csv'
  sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)
  # sec_data = pd.concat(sec_data_list, axis=0).reset_index(drop=True)
  sec_tvr = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE', 'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
  sec_tvr['full_price'] = sec_tvr['PREVSETTLEPRICE'] / sec_tvr['MINSTEP'] * sec_tvr['STEPPRICE']
  
  

  

  # функция преобразования 3-5 в лист [3, 4, 5]

  def range_num_to_list(range_num):

    result = []
    for element in range_num.split(','):
      split_index = element.find('-')
      if split_index == -1:
        result.append(int(element))
      else:
        start = int(element[:split_index])
        end = int(element[split_index+1:])
        result.extend(list(range(start, end+1)))

    return result
  
  # модуль преобразования типа скриптов

  varyant_types = pd.read_csv(varyant_types, sep='=', header=0)
  varyant_types['variant'] = varyant_types['variant'].apply(range_num_to_list)
  varyant_types = varyant_types.explode('variant')


  # основной модуль


  with open(TVR, encoding='utf8') as file:
      separator = file.readline()[0]  # Получаем элемент, который разделяет названия колонок
  with open(TVR, encoding='utf8') as file:
      column_names = file.readline().strip().split(separator)[2:]  # Получаем названия колонок начиная с третьего элемента
  data = []
  with open(TVR, encoding='utf8') as file:
      lines = file.readlines()[1:]  # Пропускаем первую строку с заголовками
      reader = csv.reader(lines, delimiter=' ')
      for row in reader:
          if len(row) > 2:
              row[2] = ' '.join(row[2:])  # Объединяем оставшиеся элементы в третий столбец
              data.append(row[:3])  # Добавляем только первые три элемента в список

  # Преобразуем список в DataFrame
  df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])

  # Изменяем тип данных на целочисленные
  df['stroka'] = df['stroka'].astype(int)
  df['stolbec'] = df['stolbec'].astype(int)

  # создаем паттерн, чтобы все стобцы были учтены
  stroka = [9999]*56
  stolbec = list(range(1,57))
  data = ['**']*56
  pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data':data}
  pattern_df = pd.DataFrame(pattern_dic)

  df = pd.concat([df,pattern_df])
  df = df.replace(r'^\s*$', np.nan, regex=True)

  tvr_table = df.pivot(index='stroka', columns='stolbec', values='data') # разворачиваем таблицу
  tvr_table.columns = column_names
  tvr_table.reset_index(inplace = True)
  tvr_with_types = tvr_table

  # tvr_with_types = tvr_table.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant') # присоединяем типы скриптов
  tvr_with_types['stroka'] = tvr_with_types['stroka'].astype(int)

  condition = (tvr_with_types['Start']=='True') & (tvr_with_types['Kill all']!='True') #& (tvr_with_types['Out only']!='True') # фильтр - оставляем только действующих роботов
  tvr_with_types = tvr_with_types[condition] # применяем предыдущий фильрт

  # вставляем новый столбец np.nan
  stolbec_ryadom = 'V 0'
  index_stolbec = tvr_with_types.columns.get_loc(stolbec_ryadom)
  tvr_with_types.insert(index_stolbec+1, 'W 0', np.nan)

  col_names = tvr_with_types.columns.tolist() # сохраняем старые названия колонок
  tvr_with_types = tvr_with_types.reset_index()

  # это для того, чтобы потом сделать unpivot
  col_n = list(range(1,len(tvr_with_types.columns))) # создаем нумерованный лист для названия колонок
  col_n.insert(0, 'stroka') # в первую позицию поставим имя 'stroka'
  tvr_with_types.columns = col_n

  tvr_with_types = pd.melt(tvr_with_types, id_vars=['stroka'], var_name='stolbec', value_name='data') # делаем "обратный процесс
  # анпивот, чтобы провести манипуляции с вставкой информации по инструменту - ГО


  tvr_with_types = tvr_with_types.sort_values(by=['stroka', 'stolbec'], ascending=True)
  

  tvr_with_types = tvr_with_types.merge(sec_tvr, how='left', left_on='data', right_on='SECID') # присоединяем таблицу с ГО
  tvr_with_types[col_count] = tvr_with_types[col_count].shift(2) # передвигаем цену ниже на нужную строку
  tvr_with_types['data'] = np.where(tvr_with_types[col_count] > 0, tvr_with_types[col_count], tvr_with_types['data']) # вставляем ГО в структуру

  tvr_with_types = tvr_with_types.iloc[:, 0:3] #оставляем только нужные колонки

  tvr_with_types = tvr_with_types.pivot(index='stroka', columns='stolbec', values='data')
  tvr_with_types.columns = col_names

  tvr_with_types.reset_index(drop = True, inplace = True)
  tvr_with_types = tvr_with_types.sort_values(by=['Mode', 'stroka'])
  tvr_with_types['indexxx'] = tvr_with_types.groupby('Mode').cumcount()
  tvr_with_types = tvr_with_types.sort_values(by=['indexxx', 'stroka'])

  tvr_with_types['varr1'] = np.where(tvr_with_types['Mode']=='1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr2'] = np.where(tvr_with_types['Mode']=='-1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr1'].ffill(inplace = True)
  tvr_with_types['varr2'].bfill(inplace = True)
  tvr_with_types['variant'] = tvr_with_types['varr1'].astype(str)+'-'+tvr_with_types['varr2'].astype(str)
  tvr_with_types.drop(columns=['indexxx', 'varr1', 'varr2'], inplace = True)

  # считаем ГО
  # Извлечение  имен  столбцов
  V_cols = [col for col in tvr_with_types.columns if col.startswith('V')]
  W_cols = [col for col in tvr_with_types.columns if col.startswith('W')]

  # Проверка, что количество столбцов "V" и "W" одинаковое
  if len(V_cols) != len(W_cols):
      raise ValueError("Количество столбцов, начинающихся с 'V',  и 'W' должно быть одинаковым!")

  # Создание `dtype_dict`  динамически
  dtype_dict = {col: int for col in V_cols}

  # Преобразование типов  столбцов
  for i in V_cols + W_cols:
      tvr_with_types[i] = tvr_with_types[i].fillna(0).astype(int)

  # Применение `dtype_dict`
  tvr_with_types = tvr_with_types.astype(dtype_dict)

  # Расчет 'total_GO'
  tvr_with_types['total_GO'] = sum(abs(tvr_with_types[V_cols[i]]) * tvr_with_types[W_cols[i]] for i in range(len(V_cols)))

  tvr_with_types = tvr_with_types.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant')
  tvr_with_types['PARAMS'] = tvr_with_types['C'].astype(str)+'_'+ tvr_with_types['N'].astype(str)+'_'+ tvr_with_types['P'].astype(str)+'_'+ tvr_with_types['E'].astype(str)+'_'+ tvr_with_types['FrId'].astype(str)+'_'+ tvr_with_types['MoveN'].astype(str)

  return tvr_with_types

def TVR_asis(TVR):
  

  

  # функция преобразования 3-5 в лист [3, 4, 5]

  def range_num_to_list(range_num):

    result = []
    for element in range_num.split(','):
      split_index = element.find('-')
      if split_index == -1:
        result.append(int(element))
      else:
        start = int(element[:split_index])
        end = int(element[split_index+1:])
        result.extend(list(range(start, end+1)))

    return result
  
  
  # основной модуль


  with open(TVR, encoding='utf8') as file:
      separator = file.readline()[0]  # Получаем элемент, который разделяет названия колонок
  with open(TVR, encoding='utf8') as file:
      column_names = file.readline().strip().split(separator)[2:]  # Получаем названия колонок начиная с третьего элемента
  data = []
  with open(TVR, encoding='utf8') as file:
      lines = file.readlines()[1:]  # Пропускаем первую строку с заголовками
      reader = csv.reader(lines, delimiter=' ')
      for row in reader:
          if len(row) > 2:
              row[2] = ' '.join(row[2:])  # Объединяем оставшиеся элементы в третий столбец
              data.append(row[:3])  # Добавляем только первые три элемента в список

  # Преобразуем список в DataFrame
  df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])

  # Изменяем тип данных на целочисленные
  df['stroka'] = df['stroka'].astype(int)
  df['stolbec'] = df['stolbec'].astype(int)

  # создаем паттерн, чтобы все стобцы были учтены
  stroka = [9999]*56
  stolbec = list(range(1,57))
  data = ['**']*56
  pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data':data}
  pattern_df = pd.DataFrame(pattern_dic)

  df = pd.concat([df,pattern_df])
  df = df.replace(r'^\s*$', np.nan, regex=True)

  tvr_table = df.pivot(index='stroka', columns='stolbec', values='data') # разворачиваем таблицу
  tvr_table.columns = column_names
  tvr_table.reset_index(inplace = True)
  tvr_table



  return tvr_table

def TVR_transform_local(TVR, varyant_types, type='Go'):
  if type == 'Go':
      col_count = 'INITIALMARGIN'
  else:
      col_count = 'full_price'

  
  sec_data_file = r'G:\MyDrive\work_data\TVR\sec_tvr.csv'
  sec_data = pd.read_csv(sec_data_file, sep=',', index_col=False)
  # sec_data = pd.concat(sec_data_list, axis=0).reset_index(drop=True)
  sec_tvr = sec_data[['SECID', 'MINSTEP', 'STEPPRICE', 'PREVSETTLEPRICE', 'INITIALMARGIN', 'BUYSELLFEE', 'SCALPERFEE']]
  sec_tvr['full_price'] = sec_tvr['PREVSETTLEPRICE'] / sec_tvr['MINSTEP'] * sec_tvr['STEPPRICE']
  
  

  

  # функция преобразования 3-5 в лист [3, 4, 5]

  def range_num_to_list(range_num):

    result = []
    for element in range_num.split(','):
      split_index = element.find('-')
      if split_index == -1:
        result.append(int(element))
      else:
        start = int(element[:split_index])
        end = int(element[split_index+1:])
        result.extend(list(range(start, end+1)))

    return result
  
  # модуль преобразования типа скриптов

  varyant_types = pd.read_csv(varyant_types, sep='=', header=0)
  varyant_types['variant'] = varyant_types['variant'].apply(range_num_to_list)
  varyant_types = varyant_types.explode('variant')


  # основной модуль


  with open(TVR, encoding='utf8') as file:
      separator = file.readline()[0]  # Получаем элемент, который разделяет названия колонок
  with open(TVR, encoding='utf8') as file:
      column_names = file.readline().strip().split(separator)[2:]  # Получаем названия колонок начиная с третьего элемента
  data = []
  with open(TVR, encoding='utf8') as file:
      lines = file.readlines()[1:]  # Пропускаем первую строку с заголовками
      reader = csv.reader(lines, delimiter=' ')
      for row in reader:
          if len(row) > 2:
              row[2] = ' '.join(row[2:])  # Объединяем оставшиеся элементы в третий столбец
              data.append(row[:3])  # Добавляем только первые три элемента в список

  # Преобразуем список в DataFrame
  df = pd.DataFrame(data, columns=['stroka', 'stolbec', 'data'])

  # Изменяем тип данных на целочисленные
  df['stroka'] = df['stroka'].astype(int)
  df['stolbec'] = df['stolbec'].astype(int)

  # создаем паттерн, чтобы все стобцы были учтены
  stroka = [9999]*56
  stolbec = list(range(1,57))
  data = ['**']*56
  pattern_dic = {'stroka': stroka, 'stolbec': stolbec, 'data':data}
  pattern_df = pd.DataFrame(pattern_dic)

  df = pd.concat([df,pattern_df])
  df = df.replace(r'^\s*$', np.nan, regex=True)

  tvr_table = df.pivot(index='stroka', columns='stolbec', values='data') # разворачиваем таблицу
  tvr_table.columns = column_names
  tvr_table.reset_index(inplace = True)
  tvr_with_types = tvr_table

  # tvr_with_types = tvr_table.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant') # присоединяем типы скриптов
  tvr_with_types['stroka'] = tvr_with_types['stroka'].astype(int)

  condition = (tvr_with_types['Start']=='True') & (tvr_with_types['Kill all']!='True') #& (tvr_with_types['Out only']!='True') # фильтр - оставляем только действующих роботов
  tvr_with_types = tvr_with_types[condition] # применяем предыдущий фильрт

  # вставляем новый столбец np.nan
  stolbec_ryadom = 'V 0'
  index_stolbec = tvr_with_types.columns.get_loc(stolbec_ryadom)
  tvr_with_types.insert(index_stolbec+1, 'W 0', np.nan)

  col_names = tvr_with_types.columns.tolist() # сохраняем старые названия колонок
  tvr_with_types = tvr_with_types.reset_index()

  # это для того, чтобы потом сделать unpivot
  col_n = list(range(1,len(tvr_with_types.columns))) # создаем нумерованный лист для названия колонок
  col_n.insert(0, 'stroka') # в первую позицию поставим имя 'stroka'
  tvr_with_types.columns = col_n

  tvr_with_types = pd.melt(tvr_with_types, id_vars=['stroka'], var_name='stolbec', value_name='data') # делаем "обратный процесс
  # анпивот, чтобы провести манипуляции с вставкой информации по инструменту - ГО


  tvr_with_types = tvr_with_types.sort_values(by=['stroka', 'stolbec'], ascending=True)
  

  tvr_with_types = tvr_with_types.merge(sec_tvr, how='left', left_on='data', right_on='SECID') # присоединяем таблицу с ГО
  tvr_with_types[col_count] = tvr_with_types[col_count].shift(2) # передвигаем цену ниже на нужную строку
  tvr_with_types['data'] = np.where(tvr_with_types[col_count] > 0, tvr_with_types[col_count], tvr_with_types['data']) # вставляем ГО в структуру

  tvr_with_types = tvr_with_types.iloc[:, 0:3] #оставляем только нужные колонки

  tvr_with_types = tvr_with_types.pivot(index='stroka', columns='stolbec', values='data')
  tvr_with_types.columns = col_names

  tvr_with_types.reset_index(drop = True, inplace = True)
  tvr_with_types = tvr_with_types.sort_values(by=['Mode', 'stroka'])
  tvr_with_types['indexxx'] = tvr_with_types.groupby('Mode').cumcount()
  tvr_with_types = tvr_with_types.sort_values(by=['indexxx', 'stroka'])

  tvr_with_types['varr1'] = np.where(tvr_with_types['Mode']=='1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr2'] = np.where(tvr_with_types['Mode']=='-1', tvr_with_types['stroka'].astype(str), np.nan)
  tvr_with_types['varr1'].ffill(inplace = True)
  tvr_with_types['varr2'].bfill(inplace = True)
  tvr_with_types['variant'] = tvr_with_types['varr1'].astype(str)+'-'+tvr_with_types['varr2'].astype(str)
  tvr_with_types.drop(columns=['indexxx', 'varr1', 'varr2'], inplace = True)

  # считаем ГО
  # Извлечение  имен  столбцов
  V_cols = [col for col in tvr_with_types.columns if col.startswith('V')]
  W_cols = [col for col in tvr_with_types.columns if col.startswith('W')]

  # Проверка, что количество столбцов "V" и "W" одинаковое
  if len(V_cols) != len(W_cols):
      raise ValueError("Количество столбцов, начинающихся с 'V',  и 'W' должно быть одинаковым!")

  # Создание `dtype_dict`  динамически
  dtype_dict = {col: int for col in V_cols}

  # Преобразование типов  столбцов
  for i in V_cols + W_cols:
      tvr_with_types[i] = tvr_with_types[i].fillna(0).astype(int)

  # Применение `dtype_dict`
  tvr_with_types = tvr_with_types.astype(dtype_dict)

  # Расчет 'total_GO'
  tvr_with_types['total_GO'] = sum(abs(tvr_with_types[V_cols[i]]) * tvr_with_types[W_cols[i]] for i in range(len(V_cols)))

  tvr_with_types = tvr_with_types.merge(varyant_types, how = 'left', left_on='stroka', right_on='variant')
  tvr_with_types['PARAMS'] = tvr_with_types['C'].astype(str)+'_'+ tvr_with_types['N'].astype(str)+'_'+ tvr_with_types['P'].astype(str)+'_'+ tvr_with_types['E'].astype(str)+'_'+ tvr_with_types['FrId'].astype(str)+'_'+ tvr_with_types['MoveN'].astype(str)

  return tvr_with_types






