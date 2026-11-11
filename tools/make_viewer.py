"""
Збирає автономну HTML-сторінку з 3D-переглядачем результатів CMakeProject1.

Бере з папки запуску:
  <prefix>_model<i>_mesh.ply      — поверхня (бінарний PLY з кольорами вершин)
  <prefix>_model<i>_cameras.txt   — пози камер
  log.txt (необов'язково)         — звідти береться RMS репроєкції кожної моделі

Приклад:
  CMakeProject1.exe photos > log.txt
  python tools/make_viewer.py . --log log.txt --images photos --out viewer.html

Сторінка повністю автономна (дані вбудовані), відкривається у браузері без сервера.
Потрібен лише numpy.
"""
import argparse
import base64
import glob
import json
import os
import re
import struct
import sys

import numpy as np

# OpenCV: y вниз, z вперед  ->  three.js: y вгору, z назад
CV_TO_GL = np.array([1.0, -1.0, -1.0])


def read_mesh(path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: не знайдено end_header")
            header += line
        lines = header.split(b"\n")
        nv = int(next(l for l in lines if l.startswith(b"element vertex")).split()[-1])
        nf = int(next(l for l in lines if l.startswith(b"element face")).split()[-1])
        verts = np.frombuffer(f.read(nv * 15), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        faces = np.frombuffer(f.read(nf * 13), dtype=[("n", "u1"), ("i", "<i4", 3)])
    return verts["p"].astype(np.float64), np.ascontiguousarray(verts["c"]), np.ascontiguousarray(faces["i"])


def read_cameras(path, half_w, half_h):
    cams = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            R = np.array(v[2:11], float).reshape(3, 3)
            t = np.array(v[11:14], float)
            center = -R.T @ t
            corners = [R.T @ np.array([x, y, 1.0]) for x, y in
                       [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]]
            cams.append({
                "id": int(v[0]),
                "file": v[1],
                "c": (center * CV_TO_GL).tolist(),
                "f": ((R.T @ [0, 0, 1]) * CV_TO_GL).tolist(),
                "u": (-(R.T @ [0, 1, 0]) * CV_TO_GL).tolist(),
                "k": [(k * CV_TO_GL).tolist() for k in corners],
            })
    return cams


def read_rms(log_path):
    """RMS після фінального BA кожної моделі: перший рядок з RMS після 'Фінальний Bundle Adjustment моделі #k'."""
    rms = {}
    if not log_path or not os.path.exists(log_path):
        return rms
    current = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Фінальний Bundle Adjustment моделі #(\d+)", line)
            if m:
                current = int(m.group(1))
                continue
            m = re.search(r"RMS репроєкції: ([0-9.eE+-]+)", line)
            if m and current is not None:
                rms[current] = float(m.group(1))
                current = None
    return rms


def image_half_extent(images_dir, focal_factor):
    """Половини розміру кадру в нормованих координатах (для пірамідок камер). Без фото — 4:3."""
    w, h = 4.0, 3.0
    if images_dir:
        for name in sorted(os.listdir(images_dir)):
            size = jpeg_size(os.path.join(images_dir, name)) if name.lower().endswith((".jpg", ".jpeg")) else None
            if size:
                w, h = size
                break
    f = focal_factor * max(w, h)
    return w / 2 / f, h / 2 / f


def jpeg_size(path):
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            marker = f.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                return None
            if marker[1] in (0xC0, 0xC1, 0xC2):
                f.read(3)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            length = struct.unpack(">H", f.read(2))[0]
            f.seek(length - 2, 1)


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # консоль Windows інакше псує українські повідомлення
    ap = argparse.ArgumentParser(description="HTML-переглядач результатів 3D-реконструкції")
    ap.add_argument("run_dir", help="папка, де запускався CMakeProject1")
    ap.add_argument("--prefix", default="reconstruction")
    ap.add_argument("--log", help="лог запуску (для RMS)")
    ap.add_argument("--images", help="папка з фото (для пропорцій пірамідок камер)")
    ap.add_argument("--focal-factor", type=float, default=0.7)
    ap.add_argument("--title", default="3D-реконструкція")
    ap.add_argument("--subtitle", default="Поверхня, відновлена з фото пайплайном SfM + SGBM. Кольори взято з самих знімків.")
    ap.add_argument("--out", default="viewer.html")
    args = ap.parse_args()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_template.html")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    half_w, half_h = image_half_extent(args.images, args.focal_factor)
    rms = read_rms(args.log)

    mesh_files = glob.glob(os.path.join(args.run_dir, f"{args.prefix}_model*_mesh.ply"))
    mesh_files.sort(key=lambda p: int(re.search(r"_model(\d+)_mesh", p).group(1)))
    if not mesh_files:
        sys.exit(f"Не знайдено {args.prefix}_model*_mesh.ply у {args.run_dir}")

    models = []
    for path in mesh_files:
        index = int(re.search(r"_model(\d+)_mesh", path).group(1))
        cam_path = path.replace("_mesh.ply", "_cameras.txt")
        positions, colors, faces = read_mesh(path)
        positions *= CV_TO_GL
        cams = read_cameras(cam_path, half_w, half_h)

        c0, f0 = np.array(cams[0]["c"]), np.array(cams[0]["f"])
        frame_ids = [c["id"] for c in cams]
        models.append({
            "name": f"Модель {index} · кадри {min(frame_ids)}–{max(frame_ids)}",
            "pos": b64(positions.astype("<f4")),
            "col": b64(colors),
            "idx": b64(faces.astype("<u4")),
            "nv": int(len(positions)),
            "nf": int(len(faces)),
            "cams": cams,
            "depth": float(np.median((positions - c0) @ f0)),
            "size": float(np.percentile(np.linalg.norm(positions - np.median(positions, 0), axis=1), 90)),
            "rms": rms.get(index),
        })
        print(f"Модель {index}: {len(positions)} вершин, {len(faces)} трикутників, {len(cams)} камер")

    data = {"title": args.title, "subtitle": args.subtitle, "models": models}
    html = (template
            .replace("/*TITLE*/", args.title)
            .replace("/*DATA*/", "window.VIEWER=" + json.dumps(data, ensure_ascii=False) + ";"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Збережено {args.out} ({size_mb:.1f} МБ)")
    if size_mb > 50:
        print("Увага: файл великий. Зменште деталізацію: CMakeProject1 ... --mesh-step 3")


if __name__ == "__main__":
    main()
