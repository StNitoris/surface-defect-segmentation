"""运行全量实验，或使用已锁定的参数处理一张未标注图片。"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean, median
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from segmentation import (Parameters, baseline, clean_mask, difference_steps,
                          metrics, read_gray, save_png, segment)


ROOT = Path(__file__).resolve().parent
GROUPS = ("In", "Pa", "Sc")
SEED = 20261123


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset(root):
    sources = {p.stem: p for p in (root / "Source Images").glob("*.bmp")}
    labels = {p.stem: p for p in (root / "Ground truth").glob("*.png")}
    if sources.keys() != labels.keys():
        raise ValueError(f"原图与标注无法配对 {sorted(sources.keys() ^ labels.keys())[:10]}")
    samples = []
    rng = random.Random(SEED)
    for group in GROUPS:
        names = sorted((n for n in sources if n.startswith(group + "_")),
                       key=lambda n: int(n.split("_")[1]))
        if len(names) != 300:
            raise ValueError(f"{group} 应有 300 对，找到 {len(names)} 对，请检查数据目录 {root}")
        development = set(rng.sample(names, 20))
        for name in names:
            image, truth = read_gray(sources[name]), read_gray(labels[name])
            if image.shape != truth.shape or np.any((truth != 0) & (truth != 255)):
                raise ValueError(f"{name} 的标注尺寸或取值异常")
            samples.append((name, group, "dev" if name in development else "test", image, truth))
    if len(samples) != len(sources):
        raise ValueError("数据目录含有未知类别")
    return samples


def choose_parameters(samples, out):
    candidates = [Parameters(k, closing, area)
                  for k, closing, area in product((31, 61), (False, True), (1, 5, 15))]
    scores = {params: [] for params in candidates}
    for _, _, split, image, truth in samples:
        if split != "dev":
            continue
        for size in (31, 61):
            binary = difference_steps(image, size)[-1]
            for params in candidates:
                if params.background_size == size:
                    mask = clean_mask(binary, params.closing, params.min_area)
                    scores[params].append(metrics(mask, truth)["iou"])
    rows = [{**asdict(p), "dev_count": len(values), "mean_iou": mean(values)}
            for p, values in scores.items()]
    write_csv(out / "parameter_search.csv", rows)
    # 并列时保留候选顺序靠前的组合。
    best = max(candidates, key=lambda p: mean(scores[p]))
    (out / "parameters.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    return best


def save_panel(path, stages, truth=None, base=None):
    panels = [
        ("1. Source", stages["source"]), ("2. Smoothed", stages["smooth"]),
        ("3. Background", stages["background"]),
        ("4. Difference (scaled)", cv2.normalize(stages["response"], None, 0, 255, cv2.NORM_MINMAX)),
        ("5. Difference + Otsu", stages["otsu"]), ("6. Final mask", stages["prediction"])]
    if truth is not None:
        panels += [("7. Ground truth", truth), ("8. Direct Otsu baseline", base)]
    cols = len(panels) // 2
    canvas = np.full((2 * 242, cols * 216), 245, dtype=np.uint8)
    for i, (label, image) in enumerate(panels):
        y, x = (i // cols) * 242, (i % cols) * 216
        height, width = image.shape
        scale = min(200 / height, 200 / width)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        preview = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        top, left = y + 30 + (200 - h) // 2, x + 8 + (200 - w) // 2
        canvas[top:top + h, left:left + w] = preview
        cv2.putText(canvas, label, (x + 8, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, 20, 1, cv2.LINE_AA)
    save_png(path, canvas)


def summary(rows):
    results = []
    for split, group, method in product(("dev", "test", "all"), (*GROUPS, "all"),
                                        ("baseline", "difference")):
        selected = [r for r in rows if (split == "all" or r["split"] == split)
                    and (group == "all" or r["group"] == group) and r["method"] == method]
        results.append({"split": split, "group": group, "method": method, "count": len(selected),
                        **{f"mean_{m}": mean(r[m] for r in selected)
                           for m in ("iou", "precision", "recall")}})
    return results


def save_cases(samples, rows, params, out):
    by_name = {name: (image, truth) for name, _, _, image, truth in samples}
    index = []
    cases_dir = out / "cases"
    cases_dir.mkdir()
    for group in GROUPS:
        scores = [r for r in rows if r["group"] == group and r["split"] == "test"
                  and r["method"] == "difference"]
        worst = min(scores, key=lambda r: r["iou"])
        middle = median(r["iou"] for r in scores)
        typical = min((r for r in scores if r is not worst), key=lambda r: abs(r["iou"] - middle))
        for role, row in (("typical", typical), ("lowest", worst)):
            name = row["name"]
            image, truth = by_name[name]
            filename = f"{group}_{role}_{name}.png"
            save_panel(cases_dir / filename, segment(image, params), truth, baseline(image))
            index.append({"case": role, **row, "file": filename})
    write_csv(cases_dir / "index.csv", index)


def run_batch(root, out):
    samples = dataset(root)
    params = choose_parameters(samples, out)
    print(f"60 张开发图选定参数 {params}", flush=True)
    for method in ("baseline", "difference"):
        (out / "masks" / method).mkdir(parents=True)
    rows = []
    for i, (name, group, split, image, truth) in enumerate(samples, 1):
        for method, pred in (("baseline", baseline(image)),
                             ("difference", segment(image, params)["prediction"])):
            save_png(out / "masks" / method / f"{name}.png", pred)
            rows.append({"name": name, "group": group, "split": split, "method": method,
                         **metrics(pred, truth)})
        if i % 300 == 0:
            print(f"已处理 {i}/{len(samples)}", flush=True)
    totals = summary(rows)
    write_csv(out / "per_image.csv", rows)
    write_csv(out / "summary.csv", totals)
    save_cases(samples, rows, params, out)
    for row in totals:
        if row["split"] == "test" and row["group"] == "all":
            print(f"{row['method']} 测试集平均 IoU = {row['mean_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "SD-saliency-900", help="数据集目录")
    parser.add_argument("--output", type=Path, default=ROOT / "results", help="结果父目录")
    parser.add_argument("--image", type=Path, help="单图模式的原图路径")
    parser.add_argument("--params", type=Path, help="单图模式所需的 parameters.json")
    args = parser.parse_args()
    if bool(args.image) != bool(args.params):
        parser.error("单图模式需同时指定 --image 和 --params")
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = ("single_" if args.image else "batch_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = args.output / run_id
    with TemporaryDirectory(dir=args.output) as work:
        out = Path(work) / "run"
        out.mkdir()
        if args.image:
            params = Parameters(**json.loads(args.params.read_text(encoding="utf-8")))
            stages = segment(read_gray(args.image), params)
            save_png(out / "mask.png", stages["prediction"])
            save_panel(out / "steps.png", stages)
        else:
            run_batch(args.data, out)
        out.rename(target)
    print(f"结果目录 {target.resolve()}")


if __name__ == "__main__":
    main()
