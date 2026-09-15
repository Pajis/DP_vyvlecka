
import arcpy
from arcpy.sa import FocalStatistics, NbrRectangle
import pandas as pd
import os
import numpy as np
import joblib
from osgeo import gdal
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
  
# === NASTAVENÍ PRACOVNÍHO PROSTŘEDÍ ===
arcpy.env.workspace = r"D:\biomasa\experiment_ll_1.gdb"
arcpy.env.overwriteOutput = True
  
# === 1. CESTY K DATŮM ===
input_points = r"D:\biomasa\experiment_ll_1.gdb\body_exp_novapredikce"
target_field = "BIOMASA_23"
  
raster_layers = [
      r"D:\biomasa\vstupy\exp_ARVI_new.tif",
      #r"D:\biomasa\vstupy\exp_GNDVI_new.tif",
      r"D:\biomasa\vstupy\exp_MSAVI_new.tif",
      r"D:\biomasa\vstupy\exp_NDI45_new.tif",
      r"D:\biomasa\vstupy\exp_ReNDVI_new.tif",
      r"D:\biomasa\vstupy\exp_B2_new.tif",
      r"D:\biomasa\vstupy\exp_B3_new.tif",
      r"D:\biomasa\vstupy\exp_B4_new.tif",
      r"D:\biomasa\vstupy\exp_B5_new.tif",
      r"D:\biomasa\vstupy\exp_B6_new.tif",
      r"D:\biomasa\vstupy\exp_B8_new.tif",
      r"D:\biomasa\vstupy\exp_B11_new.tif",
      r"D:\biomasa\vstupy\exp_B4_Entropy_new.tif",
      #r"D:\biomasa\vstupy\exp_B4_Contrast_new.tif",
      #r"D:\biomasa\vstupy\exp_B4_GLCMMean_new.tif",
      #r"D:\biomasa\vstupy\exp_B6_GLCMMean_new.tif",
      #r"D:\biomasa\vstupy\exp_B6_Contrast_new.tif",
      #r"D:\biomasa\vstupy\exp_B6_Entropy_new.tif",
      #r"D:\biomasa\vstupy\exp_B8_GLCMMean_new.tif",
     r"D:\biomasa\vstupy\exp_B8_Contrast_new.tif",
      r"D:\biomasa\vstupy\exp_B11_GLCMMean_new.tif",
      r"D:\biomasa\vstupy\exp_VV23v2new.tif",
      r"D:\biomasa\vstupy\exp_VH23v2new.tif",
      r"D:\biomasa\vstupy\exp_CH1m_v2.tif",
       #________________________________________________NOVE S DEM---pak smaz
      r"D:\biomasa\vstupy\exp_cr_aspect20m_tif.tif",
      r"D:\biomasa\vstupy\exp_cr_dem20m_tif.tif",
      r"D:\biomasa\vstupy\exp_cr_slope20m_tif.tif"
]
  
output_table = r"D:\vysledky_modelovani_AGB\AGB_table_LightGBM_all.csv"
model_path = r"D:\vysledky_modelovani_AGB\AGB_model_LightGBM_all.pkl"
output_raster_path = r"D:\vysledky_modelovani_AGB\AGB_model_LightGBM_all.tif"
feature_importance_path = r"D:\vysledky_modelovani_AGB\feature_importance_LightGBM_all.csv"

# === 1b. VYTVOŘENÍ 1 km MŘÍŽKY A PŘIŘAZENÍ GROUP_ID DO BODŮ ===
grid_size_m = 1000  # 1 km
group_field = "GRID1KM_ID"

# zkontroluj jednotky (musí být metry)
sr = arcpy.Describe(input_points).spatialReference
if not sr or sr.linearUnitName.lower() not in ["meter", "metre", "meters", "metres", "m"]:
    arcpy.AddWarning(
        f"Pozor: body jsou v jednotkách '{sr.linearUnitName if sr else 'UNKNOWN'}'. "
        "Fishnet 1 km bude korektní jen v metrech (S-JTSK/UTM)."
    )

# přidej group field do bodů, pokud chybí
existing_fields = [f.name for f in arcpy.ListFields(input_points)]
if group_field not in existing_fields:
    arcpy.management.AddField(input_points, group_field, "LONG")

# vytvoř fishnet do scratch GDB
scratch = arcpy.env.scratchGDB
fishnet_fc = os.path.join(scratch, "fishnet_1km")
sj_out = os.path.join(scratch, "pts_with_grid1km")

for fc in [fishnet_fc, sj_out]:
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)

ext = arcpy.Describe(input_points).extent

origin_coord = f"{ext.XMin} {ext.YMin}"
y_axis_coord = f"{ext.XMin} {ext.YMin + 10}"
corner_coord = f"{ext.XMax} {ext.YMax}"

arcpy.management.CreateFishnet(
    out_feature_class=fishnet_fc,
    origin_coord=origin_coord,
    y_axis_coord=y_axis_coord,
    cell_width=grid_size_m,
    cell_height=grid_size_m,
    number_rows=0,
    number_columns=0,
    corner_coord=corner_coord,
    labels="NO_LABELS",
    template=input_points,
    geometry_type="POLYGON"
)

# ID buňky (použijeme OID fishnetu)
grid_id_field = "GRID_ID"
if grid_id_field not in [f.name for f in arcpy.ListFields(fishnet_fc)]:
    arcpy.management.AddField(fishnet_fc, grid_id_field, "LONG")
arcpy.management.CalculateField(fishnet_fc, grid_id_field, f"!{arcpy.Describe(fishnet_fc).OIDFieldName}!", "PYTHON3")

# spatial join: pro každý bod najdi polygon fishnetu
arcpy.analysis.SpatialJoin(
    target_features=input_points,
    join_features=fishnet_fc,
    out_feature_class=sj_out,
    join_operation="JOIN_ONE_TO_ONE",
    join_type="KEEP_ALL",
    match_option="INTERSECT"
)

# přenes GRID_ID zpátky do originální vrstvy podle TARGET_FID
# TARGET_FID = OID původního bodu
target_fid_field = "TARGET_FID"
sj_fields = [target_fid_field, grid_id_field]

grid_map = {}
with arcpy.da.SearchCursor(sj_out, sj_fields) as cur:
    for tid, gid in cur:
        # gid může být None pokud bod mimo fishnet (nemělo by, ale ošetříme)
        grid_map[tid] = gid

oid_name = arcpy.Describe(input_points).OIDFieldName
with arcpy.da.UpdateCursor(input_points, [oid_name, group_field]) as cur:
    for oid_val, grp in cur:
        gid = grid_map.get(oid_val, None)
        cur.updateRow([oid_val, gid])

# kontrola: kolik bodů nemá group id
cnt_null = 0
with arcpy.da.SearchCursor(input_points, [group_field]) as cur:
    for (gid,) in cur:
        if gid is None:
            cnt_null += 1
if cnt_null > 0:
    arcpy.AddWarning(f"{cnt_null} bodů nemá přiřazený {group_field} (mimo fishnet / problém s projekcí).")
  
 # === 2. EXTRAKCE RASTROVÝCH HODNOT DO BODŮ (30×30 m pro les) + zápis do originálu ===
arcpy.CheckOutExtension("Spatial")

r_fields = [f"r_{i}" for i in range(len(raster_layers))]
raster_field_pairs_orig = [[r, f"r_{i}"] for i, r in enumerate(raster_layers)]

# 0) pomocný klíč SRC_OID (nepůjde do CSV)
src_oid_field = "SRC_OID"
existing = [f.name for f in arcpy.ListFields(input_points)]
if src_oid_field not in existing:
    arcpy.management.AddField(input_points, src_oid_field, "LONG")

oid_name = arcpy.Describe(input_points).OIDFieldName
with arcpy.da.UpdateCursor(input_points, [oid_name, src_oid_field]) as cur:
    for oid_val, src_val in cur:
        if src_val in (None, 0):
            cur.updateRow([oid_val, oid_val])

# 1) focal mean rastry 2x2 pixely
sr = arcpy.Describe(input_points).spatialReference
if not sr or sr.linearUnitName.lower() not in ["meter", "metre", "meters", "metres", "m"]:
    arcpy.AddWarning(
        f"Pozor: jednotky jsou '{sr.linearUnitName if sr else 'UNKNOWN'}'. "
        "Focal 2x2 bude korektní jen v pixelech."
    )

scratch = arcpy.env.scratchGDB
focal_rasters = []
nbr = NbrRectangle(2,2, "CELL")

for i, r_path in enumerate(raster_layers):
    out_r = os.path.join(scratch, f"rf_foc30m_{i}")
    if arcpy.Exists(out_r):
        arcpy.management.Delete(out_r)
    FocalStatistics(r_path, nbr, "MEAN", "DATA").save(out_r)
    focal_rasters.append(out_r)

raster_field_pairs_foc = [[r, f"r_{i}"] for i, r in enumerate(focal_rasters)]

# 2) dočasné kopie bodů
tmp_forest = os.path.join(arcpy.env.workspace, "tmp_rf_pts_forest30m")
tmp_nonfor = os.path.join(arcpy.env.workspace, "tmp_rf_pts_nonforest")
for fc in [tmp_forest, tmp_nonfor]:
    if arcpy.Exists(fc):
        arcpy.management.Delete(fc)

where_forest = f"{target_field} > 5"
where_nonfor = f"{target_field} <= 5 OR {target_field} IS NULL"

arcpy.management.MakeFeatureLayer(input_points, "lyr_pts_rf")

arcpy.management.SelectLayerByAttribute("lyr_pts_rf", "NEW_SELECTION", where_forest)
arcpy.management.CopyFeatures("lyr_pts_rf", tmp_forest)

arcpy.management.SelectLayerByAttribute("lyr_pts_rf", "NEW_SELECTION", where_nonfor)
arcpy.management.CopyFeatures("lyr_pts_rf", tmp_nonfor)

# 3) extrakce
if int(arcpy.management.GetCount(tmp_forest)[0]) > 0:
    arcpy.sa.ExtractMultiValuesToPoints(tmp_forest, raster_field_pairs_foc, "NONE")

if int(arcpy.management.GetCount(tmp_nonfor)[0]) > 0:
    arcpy.sa.ExtractMultiValuesToPoints(tmp_nonfor, raster_field_pairs_orig, "NONE")

# 4) zapiš zpět do originálu: smaž staré r_*
orig_fields = [f.name for f in arcpy.ListFields(input_points)]
for f in r_fields:
    if f in orig_fields:
        arcpy.management.DeleteField(input_points, f)

# forest -> JoinField
if int(arcpy.management.GetCount(tmp_forest)[0]) > 0:
    arcpy.management.JoinField(
        in_data=input_points,
        in_field=src_oid_field,
        join_table=tmp_forest,
        join_field=src_oid_field,
        fields=r_fields
    )

# non-forest -> doplnit NULL/NaN přes dict
def _is_missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))

if int(arcpy.management.GetCount(tmp_nonfor)[0]) > 0:
    nf_dict = {}
    with arcpy.da.SearchCursor(tmp_nonfor, [src_oid_field] + r_fields) as cur:
        for row in cur:
            nf_dict[row[0]] = row[1:]

    with arcpy.da.UpdateCursor(input_points, [src_oid_field] + r_fields) as cur:
        for row in cur:
            key = row[0]
            if key not in nf_dict:
                continue
            vals_nf = nf_dict[key]
            new_vals = list(row)
            changed = False
            for j in range(len(r_fields)):
                if _is_missing(new_vals[1 + j]) and not _is_missing(vals_nf[j]):
                    new_vals[1 + j] = vals_nf[j]
                    changed = True
            if changed:
                cur.updateRow(new_vals)

# 5) export CSV pouze BIOMASA_23 + r_*
fields = [target_field, group_field] + r_fields
data = pd.DataFrame([row for row in arcpy.da.SearchCursor(input_points, fields)], columns=fields)
data.to_csv(output_table, index=False)
  
# === 3. TRÉNINK A VALIDACE MODELU (NESTED GROUPKFold) ===
data = pd.read_csv(output_table)

# --- groups + X/y ---
groups = data[group_field].values
X = data.drop(columns=[target_field, group_field]).values
y = data[target_field].values

# --- param grid ---
param_grid = {
    'learning_rate': [0.03, 0.05],
    'num_leaves': [40,50, 60,70, 80,100],
    'max_depth': [6, 7, 8, 9],
    'min_data_in_leaf': [5, 10],
    'n_estimators': [300,400, 500],
    'boosting_type': ['gbdt'],  
    'bagging_freq': [1, 5],
    'feature_fraction': [0.65],
    'bagging_fraction': [0.8],
    'lambda_l1': [5],
    'lambda_l2': [10],
    'min_gain_to_split': [0.01],
    'max_bin': [255],
}
# --- váhy + scoring (musí být definované před GridSearch) ---
W_RMSE = 1.0
W_MAE  = 1.0
W_R2   = 1.0

scoring = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error"
}

# --- počet foldů ---
OUTER_SPLITS = 5
INNER_SPLITS = 5

# ošetření: musí být aspoň tolik groupů, kolik foldů
n_groups_total = len(np.unique(groups))
outer_splits = min(OUTER_SPLITS, n_groups_total)
if outer_splits < 2:
    raise ValueError(f"Nelze spustit outer GroupKFold: jen {n_groups_total} unikátních groupů.")

outer_cv = GroupKFold(n_splits=outer_splits)

outer_results = []
best_params_list = []

# volitelně: uložíme out-of-fold predikce pro mapu reziduí / kontrolu
data["OUTER_FOLD"] = -1
data["PRED_OOF"] = np.nan
data["RESID_OOF"] = np.nan

fold_id = 0
for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
    fold_id += 1
    print("\n" + "="*40)
    print(f"OUTER FOLD {fold_id}/{outer_splits}")
    print("="*40)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]
    groups_tr = groups[train_idx]

    # inner CV fold count nesmí být > počet groupů v outer-train
    n_groups_tr = len(np.unique(groups_tr))
    inner_splits = min(INNER_SPLITS, n_groups_tr)
    if inner_splits < 2:
        print(f" Outer fold {fold_id}: málo groupů v train ({n_groups_tr}), přeskočeno.")
        continue

    inner_cv = GroupKFold(n_splits=inner_splits)

    # GridSearch pro tento outer fold (tuning jen na outer-train)
    grid_search = GridSearchCV(
        LGBMRegressor(random_state=36),
        param_grid,
        cv=inner_cv,
        scoring=scoring,
        refit=False,  # vybereme ručně přes combo
        verbose=0,
        n_jobs=1,
        return_train_score=False
    )

    grid_search.fit(X_tr, y_tr, groups=groups_tr)

    # ruční výběr nejlepší kombinace podle combo (z INNER CV mean)
    cvres = grid_search.cv_results_
    best_i = None
    best_combo = float("inf")

    for i in range(len(cvres["params"])):
        mean_rmse = -cvres["mean_test_RMSE"][i]
        mean_mae  = -cvres["mean_test_MAE"][i]
        mean_r2   =  cvres["mean_test_R2"][i]

        combo = (W_RMSE * mean_rmse) + (W_MAE * mean_mae) + (W_R2 * (1.0 - mean_r2))
        if combo < best_combo:
            best_combo = combo
            best_i = i

    best_params = cvres["params"][best_i]
    best_params_list.append(best_params)

    print(" Nejlepší parametry (INNER CV combo):", best_params)
    print(f" INNER CV mean: RMSE={-cvres['mean_test_RMSE'][best_i]:.4f}, "
          f"MAE={-cvres['mean_test_MAE'][best_i]:.4f}, "
          f"R2={cvres['mean_test_R2'][best_i]:.4f} | combo={best_combo:.4f}")

    # Fit model na celém outer-train s best_params
    model = LGBMRegressor(random_state=36, **best_params)
    model.fit(X_tr, y_tr)

    # Vyhodnocení na outer-test (tohle je “čistý test” v rámci nested CV)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)

    print(f" OUTER TEST: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    outer_results.append({
        "outer_fold": fold_id,
        "inner_cv_combo": float(best_combo),
        "outer_R2": float(r2),
        "outer_RMSE": float(rmse),
        "outer_MAE": float(mae),
        "best_params": str(best_params)
    })

    # OOF predikce do tabulky (pro spatial residuals / kontrolu)
    data.loc[test_idx, "OUTER_FOLD"] = fold_id
    data.loc[test_idx, "PRED_OOF"] = y_pred
    data.loc[test_idx, "RESID_OOF"] = (y_te - y_pred)

# --- report nested CV ---
outer_df = pd.DataFrame(outer_results)

print("\n===== NESTED GROUPKFold SUMMARY (OUTER) =====")
print(outer_df[["outer_fold", "outer_R2", "outer_RMSE", "outer_MAE", "inner_cv_combo"]])

print("\nMEAN (outer):")
print(outer_df[["outer_R2", "outer_RMSE", "outer_MAE"]].mean())

print("\nSTD (outer):")
print(outer_df[["outer_R2", "outer_RMSE", "outer_MAE"]].std())

nested_summary_path = r"D:\vysledky_modelovani_AGB\LGBM_all_nested_summary.csv"
outer_df.to_csv(nested_summary_path, index=False)
print(f"\n Uloženo: {nested_summary_path}")

# --- výběr finálních hyperparametrů BEZ použití outer-test ---
# použijeme nejčastější best_params z outer foldů (mode)
from collections import Counter
params_mode = Counter([tuple(sorted(p.items())) for p in best_params_list]).most_common(1)[0][0]
final_params = dict(params_mode)

print("\n FINÁLNÍ PARAMETRY (mode z INNER-CV):", final_params)

# Finální model natrénuj na všech datech (X, y)
best_model = LGBMRegressor(random_state=36, **final_params)
best_model.fit(X, y)
joblib.dump(best_model, model_path)

# Predikce na všech bodech do CSV (pro kontrolu)
data["PREDIKCE"] = best_model.predict(X)

# Ulož tabulku (můžeš pak mapovat OUTER_FOLD + RESID_OOF)
data.to_csv(output_table, index=False)


# === 4. PREDIKCE DO RASTRU ===
print("Provádím predikci do rastrových dat...")

best_model = joblib.load(model_path)

sample_raster = gdal.Open(raster_layers[0])
nrows, ncols = sample_raster.RasterYSize, sample_raster.RasterXSize
geotransform = sample_raster.GetGeoTransform()
projection = sample_raster.GetProjection()
nodata_value = -3.4028234663852886e+38

driver = gdal.GetDriverByName("GTiff")
output_raster = driver.Create(output_raster_path, ncols, nrows, 1, gdal.GDT_Float32)
output_raster.SetGeoTransform(geotransform)
output_raster.SetProjection(projection)
output_band = output_raster.GetRasterBand(1)
output_band.SetNoDataValue(nodata_value)

chunk_size = 100
for start_row in range(0, nrows, chunk_size):
    end_row = min(start_row + chunk_size, nrows)

    raster_values = []
    for raster_path in raster_layers:
        raster = gdal.Open(raster_path)
        if raster is None:
            raise ValueError(f"Could not open raster: {raster_path}")

        raster_band = raster.GetRasterBand(1)
        arr = raster_band.ReadAsArray(0, start_row, ncols, end_row - start_row)
        if arr is None:
            raise ValueError(f"Failed to read data from raster: {raster_path}")

        arr = arr.astype(np.float32)
        arr[arr == nodata_value] = np.nan
        raster_values.append(arr)

    raster_stack = np.stack(raster_values, axis=-1)
    valid_mask = ~np.isnan(raster_stack).any(axis=2)

    output_chunk = np.full((end_row - start_row, ncols), nodata_value, dtype=np.float32)
    if valid_mask.any():
        X_pred = raster_stack[valid_mask]
        y_pred = best_model.predict(X_pred)
        output_chunk[valid_mask] = y_pred

    output_band.WriteArray(output_chunk, 0, start_row)

print(" Prediction complete! Raster saved.")
output_raster.FlushCache()
output_raster = None
print(f"Predikovaný raster uložen do: {output_raster_path}")

# === 5. FEATURE IMPORTANCE ===
feature_importance = pd.DataFrame({
    "Feature": [f"r_{i}" for i in range(len(raster_layers))],
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv(feature_importance_path, index=False)
print(f"Feature importance uloženo do: {feature_importance_path}")
print("Nejlepší parametry (combo):", best_params)
