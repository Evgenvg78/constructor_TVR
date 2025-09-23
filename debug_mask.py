import pandas as pd
from pathlib import Path
from src.tvr_service.templates.mask_template import build_template_from_mask_df

mask_path = Path('mask.xlsx')
mask_df = pd.read_excel(mask_path)
print(mask_df)
try:
    result = build_template_from_mask_df(mask_df, name='mask_demo', row_alias_column='row_alias')
    print('ok')
except Exception as exc:
    print('error', exc)
