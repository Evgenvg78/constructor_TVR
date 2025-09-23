import pandas as pd
from pathlib import Path
mask_df = pd.read_excel(Path('mask.xlsx'))
print(mask_df['stroka'])
print(mask_df['stroka'].duplicated())
print(mask_df['stroka'].duplicated().tolist())
