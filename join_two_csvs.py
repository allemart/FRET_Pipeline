from pathlib import Path

import pandas as pd


csv_1 = Path('/Volumes/AlleMicro/FACS/Exported/26-6-11 JAK2 vs HLB PI3K x1/export_B01 j2 382.3 veh_NoDebris.csv')
csv_2 = Path('/Volumes/AlleMicro/FACS/Exported/26-6-11 JAK2 vs HLB PI3K x1/export_C01 j2 382.3 veh cont_NoDebris.csv')
output_csv = Path('/Volumes/AlleMicro/FACS/Exported/26-6-11 JAK2 vs HLB PI3K x1/export_B02 j2 382.3 veh MERGE_NoDebris.csv')


df_1 = pd.read_csv(csv_1)
df_2 = pd.read_csv(csv_2)

df_2 = df_2.copy()
df_2["Time"] = df_2["Time"] + df_1["Time"].iloc[-1]

joined = pd.concat([df_1, df_2], ignore_index=True)
joined.to_csv(output_csv, index=False)

print(f"Saved joined CSV to: {output_csv}")
