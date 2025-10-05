k2pandc = {
    "disposition": "disposition",
    "pl_orbper": "orbit_period_days",
    "st_rad": "stellar_radius_sun",
    "st_teff": "stellar_effective_temp_kelvin",
    "pl_eqt": "planet_equilibrium_temp_kelvin",
    "st_logg": "stellar_surface_gravity_log10_cm_s2",
}
cumulative = {
    "koi_pdisposition": "disposition",
    "koi_period": "orbit_period_days",
    "koi_srad": "stellar_radius_sun",
    "koi_steff": "stellar_effective_temp_kelvin",
    "koi_teq": "planet_equilibrium_temp_kelvin",
    "koi_slogg": "stellar_surface_gravity_log10_cm_s2",
}
toi = {
    "tfopwg_disp": "disposition",
    "pl_orbper": "orbit_period_days",
    "st_rad": "stellar_radius_sun",
    "st_teff": "stellar_effective_temp_kelvin",
    "pl_eqt": "planet_equilibrium_temp_kelvin",
    "st_logg": "stellar_surface_gravity_log10_cm_s2",
}
candidate_labels = ["CANDIDATE","PC","CP","KP","APC","CONFIRMED"]
not_candidate_labels = ["FALSE POSITIVE","FP","FA","REFUTED"]

import pandas as pd

toi_df = pd.read_csv("TOI.csv")

k2pandc_df = pd.read_csv("k2pandc.csv")

cumulative_df = pd.read_csv("cumulative.csv")

# change column names equal to its dictionary to same value
toi_usable_cols_df = toi_df[list(toi.keys())]
toi_usable_cols_df = toi_usable_cols_df.rename(columns=toi)
k2pandc_usable_cols_df = k2pandc_df[list(k2pandc.keys())]
k2pandc_usable_cols_df = k2pandc_usable_cols_df.rename(columns=k2pandc)
cumulative_usable_cols_df = cumulative_df[list(cumulative.keys())]
cumulative_usable_cols_df = cumulative_usable_cols_df.rename(columns=cumulative)

export_df = pd.concat([toi_usable_cols_df, k2pandc_usable_cols_df, cumulative_usable_cols_df], ignore_index=True)
print(export_df['disposition'].value_counts())
# set canditates label to 1 ,not candidates to 0
for row in export_df.itertuples():
    if row.disposition in candidate_labels:
        export_df.at[row.Index, 'disposition'] = 1
    elif row.disposition in not_candidate_labels:
        export_df.at[row.Index, 'disposition'] = 0

print(export_df['disposition'].value_counts())
export_df.to_csv("merged_data.csv", index=False)