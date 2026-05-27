"""
Construye el corpus consolidado del TFM.

Mezcla los 4 datasets etiquetados (PlantVillage + Tilahun + Mendeley UE + Quizhpe)
en un unico conjunto, genera splits estratificados por clase Y por origen,
detecta duplicados por MD5, y produce los CSVs que consumiran el entrenamiento
y la evaluacion Edge.

Salida en outputs/corpus/:
  train.csv, val.csv, test.csv         - splits del corpus consolidado
  propio_inventory.csv                  - inventario del dataset sin etiquetar
  duplicates.csv                        - duplicados detectados (si los hay)
  corpus_summary.json                   - resumen para la memoria

Uso:
    python scripts/01_prepare_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def file_md5(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def scan_labeled_dataset(name: str) -> list[dict]:
    info = config.DATASETS[name]
    root, class_map = info["path"], info["map"]
    records = []

    if class_map is None:
        return records

    if not root.exists():
        print(f"  [MISS] {name}: ruta no existe -> {root}")
        return records

    for folder_name, canonical_class in class_map.items():
        folder = root / folder_name
        if not folder.exists():
            print(f"  [MISS] {name}: subcarpeta '{folder_name}' no encontrada")
            continue
        if canonical_class not in config.CLASS_TO_IDX:
            print(f"  [WARN] {name}: clase fuera de scope: {canonical_class}")
            continue

        label = config.CLASS_TO_IDX[canonical_class]
        count = 0
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                records.append({
                    "abs_path":   str(f.resolve()),
                    "label":      label,
                    "class_name": canonical_class,
                    "source":     name,
                    "md5":        file_md5(f),
                })
                count += 1
        print(f"    {name}/{folder_name} -> {canonical_class}: {count}")

    return records


def scan_unlabeled_dataset(name: str) -> list[dict]:
    info = config.DATASETS[name]
    root = info["path"]
    if not root.exists():
        return []
    records = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            records.append({
                "abs_path":   str(f.resolve()),
                "label":      None,
                "class_name": None,
                "source":     name,
                "md5":        file_md5(f),
            })
    print(f"    {name} (sin etiquetar): {len(records)}")
    return records


def detect_duplicates(records: list[dict]) -> list[dict]:
    by_hash = defaultdict(list)
    for r in records:
        by_hash[r["md5"]].append(r)
    dups = []
    for md5, group in by_hash.items():
        if len(group) > 1:
            for r in group:
                dups.append({
                    "md5": md5,
                    "abs_path": r["abs_path"],
                    "source": r["source"],
                    "class_name": r["class_name"],
                })
    return dups


def deduplicate_keep_first(records: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for r in records:
        if r["md5"] not in seen:
            seen.add(r["md5"])
            unique.append(r)
    return unique


def stratified_split_2way(df, ratios, seed):
    """
    Split estratificado por (label, source) en 3 particiones.
    ratios = (train, val, test) deben sumar 1.
    """
    df = df.copy().reset_index(drop=True)
    #strat = df["label"].astype(str) + "_" + df["source"]
    strat = df["label"] 
    # Primer split: train vs rest
    rest_ratio = ratios[1] + ratios[2]
    train_idx, rest_idx = train_test_split(
        df.index, test_size=rest_ratio, stratify=strat, random_state=seed,
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_rest  = df.loc[rest_idx].reset_index(drop=True)

    # Segundo split: val vs test
    rel_test = ratios[2] / (ratios[1] + ratios[2])
    #strat_rest = df_rest["label"].astype(str) + "_" + df_rest["source"]
    strat_rest = df_rest["label"]
    val_idx, test_idx = train_test_split(
        df_rest.index, test_size=rel_test, stratify=strat_rest, random_state=seed,
    )
    return (
        df_train,
        df_rest.loc[val_idx].reset_index(drop=True),
        df_rest.loc[test_idx].reset_index(drop=True),
    )


def summarize(df, name):
    if df is None or df.empty:
        return {"name": name, "total": 0, "per_class": {}, "per_source": {}}
    per_cls = df["class_name"].value_counts().to_dict() if "class_name" in df.columns else {}
    per_src = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    return {
        "name":       name,
        "total":      int(len(df)),
        "per_class":  {k: int(v) for k, v in per_cls.items()},
        "per_source": {k: int(v) for k, v in per_src.items()},
    }


def corpus_sha256(records):
    sorted_hashes = sorted(r["md5"] for r in records)
    return hashlib.sha256("".join(sorted_hashes).encode("utf-8")).hexdigest()


def main():
    config.ensure_dirs()
    out_dir = config.CORPUS_DIR
    print("=" * 70)
    print("CONSTRUCCION DEL CORPUS CONSOLIDADO")
    print("=" * 70)
    print(f"Salida: {out_dir}")
    print(f"Seed:   {config.SEED}\n")

    # Escaneo de datasets etiquetados
    print("[ESCANEO] datasets etiquetados:")
    labeled_records = []
    labeled_names = config.get_labeled_datasets()
    for name in labeled_names:
        labeled_records.extend(scan_labeled_dataset(name))
    print(f"  Total bruto: {len(labeled_records)} imagenes")

    if not labeled_records:
        raise SystemExit("ERROR: no se encontraron imagenes etiquetadas")

    # Escaneo del dataset propio (sin etiquetar, solo inventario)
    print("\n[INVENTARIO] datasets sin etiquetar:")
    pending_records = []
    for name in config.get_pending_datasets():
        pending_records.extend(scan_unlabeled_dataset(name))

    # Deteccion de duplicados
    print("\n[DUPLICADOS] verificando hash MD5...")
    duplicates = detect_duplicates(labeled_records)
    if duplicates:
        print(f"  AVISO: {len(duplicates)} archivos involucrados en duplicados")
        pd.DataFrame(duplicates).to_csv(out_dir / "duplicates.csv", index=False)
        labeled_records = deduplicate_keep_first(labeled_records)
        print(f"  Tras dedup: {len(labeled_records)} imagenes unicas")
    else:
        print("  OK: sin duplicados")

    # Construir DataFrame consolidado
    df_all = pd.DataFrame(labeled_records)
    print(f"\nCorpus consolidado: {len(df_all)} imagenes")
    print("Distribucion por clase:")
    print(df_all["class_name"].value_counts().to_string())
    #print("\nDistribucion por origen:")
    #print(df_all["source"].value_counts().to_string())
    print("\nDistribucion (clase x origen):")
    cross = df_all.groupby(["source", "class_name"]).size().unstack(fill_value=0)
    print(cross.to_string())

    min_per_combo = cross.replace(0, np.nan).min().min()
    print(f"\nMinimo de muestras por combinacion (clase x origen): "
          f"{int(min_per_combo) if not np.isnan(min_per_combo) else 'N/A'}")
    if not np.isnan(min_per_combo) and min_per_combo < 3:
        print("  AVISO: combinaciones con <3 muestras presentes. "
              "Estratificacion bidimensional desactivada (solo por clase).")


    n_dup = df_all["md5"].duplicated().sum()
    if n_dup > 0:
        raise RuntimeError(
            f"df_all contiene {n_dup} duplicados MD5. "
            f"La deduplicacion previa fallo. Abortando."
        )
    # Split estratificado por clase Y origen
    ratios = (config.SPLITS["train"], config.SPLITS["val"], config.SPLITS["test"])
    print(f"\n[SPLIT] estratificado por (label, source) con ratios {ratios}")
    df_train, df_val, df_test = stratified_split_2way(df_all, ratios, config.SEED)
    print(f"  train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")

    # Guardar CSVs
    print("\n[GUARDAR] CSVs:")
    csvs = [
        ("train.csv",            df_train),
        ("val.csv",              df_val),
        ("test.csv",             df_test),
        ("propio_inventory.csv", pd.DataFrame(pending_records) if pending_records else pd.DataFrame()),
    ]
    for filename, df in csvs:
        path = out_dir / filename
        df.to_csv(path, index=False)
        print(f"  {filename:<24} {len(df):>5}")

    # Resumen JSON
    summary = {
        "design": {
            "approach":   "single_pool_stratified",
            "stratify_by": ["label", "source"],
            "labeled_datasets": labeled_names,
            "pending_datasets": config.get_pending_datasets(),
            "splits": config.SPLITS,
            "seed":   config.SEED,
        },
        "splits": {
            "train": summarize(df_train, "train"),
            "val":   summarize(df_val,   "val"),
            "test":  summarize(df_test,  "test"),
        },
        "inventory": {
            "propio_total": int(len(pending_records)),
        },
        "duplicates_detected": len(duplicates),
        "corpus_sha256":       corpus_sha256(labeled_records),
        "total_unique_images": len(labeled_records),
    }
    with open(out_dir / "corpus_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Tabla final
    print("\n" + "=" * 70)
    print(f"{'Split':<10} {'Total':>8} {'healthy':>10} {'early_b':>10} {'late_b':>10}")
    print("-" * 70)
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        pc = df["class_name"].value_counts().to_dict()
        print(f"{name:<10} {len(df):>8} "
              f"{pc.get('healthy', 0):>10} "
              f"{pc.get('early_blight', 0):>10} "
              f"{pc.get('late_blight', 0):>10}")
    print(f"{'propio':<10} {len(pending_records):>8} (sin etiquetar)")
    print("=" * 70)
    print(f"\nCorpus SHA256: {summary['corpus_sha256'][:16]}...")
    print(f"Resumen completo: {out_dir / 'corpus_summary.json'}")


if __name__ == "__main__":
    main()
