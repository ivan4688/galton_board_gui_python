# -*- coding: utf-8 -*-
"""
Galton Interactive — окончательный исполняемый скрипт
(Обновление: анимация закрывается автоматически и сразу запускает сохранение)
"""
import os
import sys
import time
import json
import math
import random
import sqlite3
import threading
import subprocess
import shutil
import webbrowser
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  # only for compatibility closing
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as mticker

# ---------------- Reliable Cyrillic font selection ----------------
_candidates = ["DejaVu Sans", "Arial", "Liberation Sans", "Tahoma"]
_chosen_path = None
for _name in _candidates:
    try:
        _path = fm.findfont(_name, fallback_to_default=False)
        if _path and _path.lower().endswith((".ttf", ".otf")):
            _chosen_path = _path
            break
    except Exception:
        continue
if _chosen_path is None:
    _chosen_path = fm.findfont("DejaVu Sans")
try:
    fm.fontManager.addfont(_chosen_path)
except Exception:
    pass
_CHOSEN_NAME = fm.FontProperties(fname=_chosen_path).get_name()
matplotlib.rcParams['font.family'] = _CHOSEN_NAME
matplotlib.rcParams['font.sans-serif'] = [_CHOSEN_NAME]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11
GLOBAL_FONT_PROP = fm.FontProperties(fname=_chosen_path, size=11)
GLOBAL_MONO_FONT_PROP = fm.FontProperties(fname=_chosen_path, size=10)

# ---------------- Theme colors ----------------
BG = "#f7f9fb"
PANEL_BG = "#ffffff"
CARD_BG = "#eef6ff"
TEXT_COLOR = "#0b2540"
ACCENT = "#39a0ed"
BAR_FILL = "#ffd166"
PEG_FILL = "#cfe9ff"
PEG_OUTLINE = "#093042"

ANIM_BG = "#071226"
ANIM_CARD = "#0f2740"
ANIM_TEXT = "#e8f1ff"
ANIM_PEG = "#cfe9ff"
ANIM_PEG_OUTLINE = "#093042"

# Matplotlib defaults
matplotlib.rcParams['figure.facecolor'] = PANEL_BG
matplotlib.rcParams['axes.facecolor'] = PANEL_BG
matplotlib.rcParams['savefig.facecolor'] = "white"
matplotlib.rcParams['text.color'] = TEXT_COLOR
matplotlib.rcParams['axes.labelcolor'] = TEXT_COLOR
matplotlib.rcParams['xtick.color'] = TEXT_COLOR
matplotlib.rcParams['ytick.color'] = TEXT_COLOR
matplotlib.rcParams['axes.edgecolor'] = "#2a3f50"

# ---------------- Database ----------------
DB_PATH = "galton_runs.db"


def sqlite_connect(path=DB_PATH, timeout=30):
    conn = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    return conn


def migrate_db_schema(db_path=DB_PATH):
    expected_cols = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "ts": "TEXT",
        "n": "INTEGER",
        "num_balls": "INTEGER",
        "batch_size": "INTEGER",
        "speed": "REAL",
        "pause": "REAL",
        "duration": "REAL",
        "avg_rel_dev_pct": "REAL",
        "corr": "REAL",
        "chi2": "REAL",
        "counts": "TEXT",
        "images": "TEXT",
        "run_folder": "TEXT"
    }
    conn = sqlite_connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
    if cur.fetchone() is None:
        cols_sql = ",\n  ".join(f"{col} {typ}" for col, typ in expected_cols.items())
        create_sql = f"CREATE TABLE runs (\n  {cols_sql}\n);"
        cur.execute(create_sql)
        conn.commit()
        conn.close()
        return
    cur.execute("PRAGMA table_info(runs)")
    existing = [row[1] for row in cur.fetchall()]
    for col, typ in expected_cols.items():
        if col in existing:
            continue
        try:
            alter_sql = f"ALTER TABLE runs ADD COLUMN {col} {typ}"
            cur.execute(alter_sql)
        except Exception:
            pass
    conn.commit()
    conn.close()


def init_db(db_path=DB_PATH):
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    migrate_db_schema(db_path)


def save_run_to_db(params, metrics, counts, image_paths, duration, run_folder=None, db_path=DB_PATH):
    try:
        counts_list = list(counts)
    except Exception:
        counts_list = counts if isinstance(counts, list) else []
    try:
        conn = sqlite_connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO runs (ts, n, num_balls, batch_size, speed, pause, duration, avg_rel_dev_pct, corr, chi2, counts, images, run_folder)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(timespec="seconds"),
            int(params.get("n", 0)),
            int(params.get("num_balls", 0)),
            int(params.get("batch_size", 0)),
            float(params.get("speed", 0.0)),
            float(params.get("pause", 0.0)),
            float(duration),
            float(metrics.get("avg_rel_dev_pct", float('nan'))),
            float(metrics.get("corr", float('nan'))),
            float(metrics.get("chi2", float('nan'))),
            json.dumps(counts_list, ensure_ascii=False),
            json.dumps(image_paths, ensure_ascii=False),
            os.path.abspath(run_folder) if run_folder else None
        ))
        conn.commit()
        run_id = cur.lastrowid
        conn.close()
        return run_id
    except Exception as err:
        print("[DB SAVE] error:", err)
        return None


def load_all_runs(db_path=DB_PATH):
    conn = sqlite_connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, ts, n, num_balls, batch_size, speed, pause, duration, avg_rel_dev_pct, corr, chi2, counts, images, run_folder
            FROM runs ORDER BY id ASC
        """)
    except sqlite3.OperationalError:
        cur.execute("SELECT id, ts, n, num_balls, counts, images FROM runs ORDER BY id ASC")
        rows = cur.fetchall()
        conn.close()
        recs = []
        for r in rows:
            recs.append({
                "id": r[0], "ts": r[1], "n": r[2], "num_balls": r[3], "batch_size": 0, "speed": 0, "pause": 0, "duration": 0.0,
                "avg_rel_dev_pct": float('nan'), "corr": float('nan'), "chi2": float('nan'),
                "counts": json.loads(r[4]) if r[4] else [], "images": json.loads(r[5]) if r[5] else [], "run_folder": None
            })
        return recs
    rows = cur.fetchall()
    conn.close()
    recs = []
    for r in rows:
        try:
            recs.append({
                "id": r[0], "ts": r[1], "n": r[2], "num_balls": r[3],
                "batch_size": r[4], "speed": r[5], "pause": r[6], "duration": r[7],
                "avg_rel_dev_pct": float(r[8]) if r[8] is not None else float('nan'),
                "corr": float(r[9]) if r[9] is not None else float('nan'),
                "chi2": float(r[10]) if r[10] is not None else float('nan'),
                "counts": json.loads(r[11]) if r[11] else [], "images": json.loads(r[12]) if r[12] else [], "run_folder": r[13]
            })
        except Exception as e:
            print("[DB LOAD] row parse error:", e)
    return recs

# ---------------- Math/helpers ----------------
def binomial_coefs(n):
    return [math.comb(n, k) for k in range(n+1)]

def binomial_pmf(n):
    coefs = binomial_coefs(n)
    denom = 2 ** n
    return [c/denom for c in coefs]

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2*math.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)

def polyline_length(points):
    L = 0.0
    for i in range(len(points)-1):
        x0,y0 = points[i]; x1,y1 = points[i+1]
        L += math.hypot(x1-x0, y1-y0)
    return L

def point_at_distance(points, dist):
    if dist <= 0:
        return points[0]
    total = 0.0
    for i in range(len(points)-1):
        x0,y0 = points[i]; x1,y1 = points[i+1]
        seg = math.hypot(x1-x0, y1-y0)
        if total + seg >= dist:
            t = (dist - total) / seg
            return (x0 + (x1-x0)*t, y0 + (y1-y0)*t)
        total += seg
    return points[-1]

def simulate_headless_counts(n_levels, num_balls, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    ks = rng.binomial(n_levels, 0.5, size=num_balls)
    counts = np.bincount(ks, minlength=n_levels+1)
    return counts.astype(int)

# ---------------- Utilities ----------------
def open_path(path):
    path = os.path.abspath(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            if shutil.which("xdg-open"):
                subprocess.Popen(["xdg-open", path])
            else:
                subprocess.Popen(["xdg-open", path])
    except Exception as e:
        messagebox.showerror("Ошибка открытия", f"Не удалось открыть путь {path}: {e}")

def find_file_in_runs_dir(basename):
    root = os.path.abspath("runs")
    if not os.path.exists(root):
        return None
    for dirpath, dirs, files in os.walk(root):
        if basename in files:
            return os.path.join(dirpath, basename)
    return None

# ---------------- Save progress dialog ----------------
class SaveProgressDialog(tk.Toplevel):
    def __init__(self, master, title="Сохранение", max_steps=100):
        super().__init__(master)
        self.transient(master)
        self.grab_set()
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.max_steps = max_steps
        self.value = tk.DoubleVar(value=0.0)
        ttk.Label(self, text="Сохранение прогона...").pack(padx=12, pady=(10,4))
        self.progress = ttk.Progressbar(self, orient="horizontal", length=420, mode="determinate", maximum=100, variable=self.value)
        self.progress.pack(padx=12, pady=(0,8))
        self.lbl = ttk.Label(self, text="")
        self.lbl.pack(padx=12, pady=(0,12))
        self.update_idletasks()
        w = self.winfo_reqwidth(); h = self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + max(0, (master.winfo_width()-w)//2)
            y = master.winfo_rooty() + max(0, (master.winfo_height()-h)//2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def update_progress(self, value, text=None):
        try:
            v = float(value)
        except Exception:
            v = 0.0
        self.value.set(max(0.0, min(100.0, v)))
        if text:
            self.lbl.config(text=text)
        self.update_idletasks()

    def close(self):
        try:
            self.grab_release()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

# ---------------- Plot generation (Agg) ----------------
def generate_and_save_plots(counts, n_levels, num_balls, outdir, progress_callback=None):
    os.makedirs(outdir, exist_ok=True)
    ks = np.arange(0, n_levels+1)
    obs = np.array(counts, dtype=float)
    probs = binomial_pmf(n_levels)
    exp = np.array([p * num_balls for p in probs], dtype=float)

    mask = exp > 0
    rel_dev = np.abs(obs[mask] - exp[mask]) / exp[mask] if mask.any() else np.array([0.0])
    avg_rel_dev_pct = float(np.mean(rel_dev) * 100) if rel_dev.size > 0 else 0.0
    corr = float(np.corrcoef(obs, exp)[0,1]) if (obs.std()>0 and exp.std()>0) else float('nan')
    chi2 = float(np.sum(((obs[mask] - exp[mask])**2) / exp[mask])) if mask.any() else float('nan')

    saved = []
    total_steps = 6

    def step_callback(idx, text=None):
        if progress_callback:
            try:
                progress_callback(idx, total_steps, text or "")
            except Exception:
                pass

    def save_fig_and_step(fig, path, idx, message=None):
        canvas = FigureCanvasAgg(fig)
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(path, dpi=140)
        try:
            fig.clf()
        except Exception:
            pass
        try:
            del fig
        except Exception:
            pass
        saved.append(path)
        step_callback(idx, message or f"Сохранено: {os.path.basename(path)}")

    # 1) Pascal table
    fig1 = Figure(figsize=(7,2.6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Коэффициенты Паскаля и вероятности", fontproperties=GLOBAL_FONT_PROP)
    col_labels = ["k", "C(n,k)", "P(k)"]
    cell_data = [[str(k), str(c), f"{p:.6f}"] for k, (c,p) in enumerate(zip(binomial_coefs(n_levels), binomial_pmf(n_levels)))]
    ax1.axis('off')
    table = ax1.table(cellText=cell_data, colLabels=col_labels, cellLoc='left', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row,col), cell in table.get_celld().items():
        if row == -1:
            cell.get_text().set_fontproperties(GLOBAL_FONT_PROP)
        else:
            cell.get_text().set_fontproperties(GLOBAL_MONO_FONT_PROP)
    path1 = os.path.join(outdir, "pascal_table.png")
    save_fig_and_step(fig1, path1, 1, "Паскаль сохранён")

    # 2) Metrics
    fig2 = Figure(figsize=(5,3))
    fig2.suptitle("Метрики прогона", fontproperties=GLOBAL_FONT_PROP)
    txt = (f"Эксперимент: {num_balls} шариков, уровней: {n_levels}\n\n"
           f"Среднее относительное отклонение: {avg_rel_dev_pct:.3f}%\n"
           f"Корреляция (obs vs exp): {corr:.4f}\n"
           f"Chi2: {chi2:.3f}\n")
    fig2.text(0.02, 0.5, txt, fontproperties=GLOBAL_FONT_PROP)
    path2 = os.path.join(outdir, "metrics.png")
    save_fig_and_step(fig2, path2, 2, "Метрики сохранены")

    # 3) Histogram counts
    fig3 = Figure(figsize=(7,3.5))
    ax3 = fig3.add_subplot(111)
    ax3.bar(ks, obs, alpha=0.9, color=BAR_FILL)
    ax3.set_xticks(ks)
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax3.set_xlabel("Номер корзины (k)", fontproperties=GLOBAL_FONT_PROP)
    ax3.set_ylabel("Число шариков", fontproperties=GLOBAL_FONT_PROP)
    ax3.set_title("Гистограмма: counts", fontproperties=GLOBAL_FONT_PROP)
    ax3.grid(axis='y', linestyle='--', alpha=0.35)
    path3 = os.path.join(outdir, "hist_counts.png")
    save_fig_and_step(fig3, path3, 3, "Гистограмма сохранена")

    # 4) Normalized comparison
    fig4 = Figure(figsize=(7,3.5))
    ax4 = fig4.add_subplot(111)
    emp_probs = obs / obs.sum() if obs.sum()>0 else obs
    width = 0.6
    ax4.bar(ks - width/3, emp_probs, width=width/3, label="Эксперимент (частоты)", alpha=0.85)
    ax4.bar(ks, probs, width=width/3, label="Биномиальное (теория)", alpha=0.6)
    mu = n_levels/2.0
    sigma = math.sqrt(n_levels/4.0) if n_levels>0 else 1.0
    xs = np.linspace(0, n_levels, 400)
    ys = [normal_pdf(x, mu, sigma) for x in xs]
    ax4.plot(xs, ys, linewidth=2, label="Норм. аппрокс. (плотность)")
    ax4.set_xticks(ks)
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax4.set_xlabel("k", fontproperties=GLOBAL_FONT_PROP)
    ax4.set_ylabel("Частота / Вероятность", fontproperties=GLOBAL_FONT_PROP)
    ax4.set_title("Сравнение: нормаль, биномиал, эксперимент", fontproperties=GLOBAL_FONT_PROP)
    ax4.legend(prop=GLOBAL_FONT_PROP)
    ax4.grid(alpha=0.35)
    path4 = os.path.join(outdir, "normalized_comparison.png")
    save_fig_and_step(fig4, path4, 4, "Сравнение сохранено")

    # 5) Observed vs expected
    fig5 = Figure(figsize=(7,3.5))
    ax5 = fig5.add_subplot(111)
    ax5.bar(ks - 0.18, obs, width=0.35, label="Эксперимент (числа)")
    ax5.bar(ks + 0.18, exp, width=0.35, label="Ожидаемые (биномиал)")
    ax5.set_xticks(ks)
    ax5.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax5.set_xlabel("k", fontproperties=GLOBAL_FONT_PROP)
    ax5.set_ylabel("Число шариков", fontproperties=GLOBAL_FONT_PROP)
    ax5.set_title("Наблюдаемое vs Ожидаемое (биномиал)", fontproperties=GLOBAL_FONT_PROP)
    ax5.legend(prop=GLOBAL_FONT_PROP)
    ax5.grid(axis='y', alpha=0.35)
    path5 = os.path.join(outdir, "observed_vs_expected.png")
    save_fig_and_step(fig5, path5, 5, "Obs vs Exp сохранено")

    metadata = {
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "n": int(n_levels),
        "num_balls": int(num_balls),
        "counts": list(map(int, list(counts))),
        "metrics": {"avg_rel_dev_pct": avg_rel_dev_pct, "corr": corr, "chi2": chi2},
        "images": saved
    }
    meta_path = os.path.join(outdir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    step_callback(total_steps, "Готово: metadata сохранён")

    return saved, metadata["metrics"]

# ---------------- Galton animator (dark themed) ----------------
class GaltonBoardAnimator(tk.Toplevel):
    def __init__(self, master, n_levels=10, num_balls=500, pixels_per_second=240.0,
                 batch_size=50, batch_pause=0.5, canvas_w=1200, canvas_h=680, on_finish=None, release_interval_sec=None):
        super().__init__(master)
        self.title("Galton Board — анимация")
        self.resizable(False, False)
        self.on_finish = on_finish

        # params
        self.canvas_w = int(canvas_w); self.canvas_h = int(canvas_h)
        self.n = max(1, int(n_levels)); self.num_balls = max(1, int(num_balls))
        self.pixels_per_second = float(pixels_per_second)
        self.batch_size = max(1, int(batch_size)); self.batch_pause = float(batch_pause)
        self.frame_ms = 16
        # release_interval_sec: if provided (>0) -> release balls one-by-one with this interval
        self.release_interval_ms = int(release_interval_sec * 1000) if (release_interval_sec is not None and release_interval_sec > 0) else None

        # style
        self.bg_color = ANIM_BG; self.card_color = ANIM_CARD; self.text_color = ANIM_TEXT
        self.pegs_color = ANIM_PEG; self.peg_outline = ANIM_PEG_OUTLINE; self.bar_fill = BAR_FILL

        # layout defaults (will be scaled)
        self.peg_spacing_x = 44; self.peg_spacing_y = 44
        self.peg_radius = 5; self.ball_radius = 6; self.font_pegs = ("Helvetica", 10, "bold")
        self.desired_hist_height = 120

        self._compute_scaling_to_fit()
        self.peg_positions = []
        self.bin_centers = []
        self.bin_counts = [0] * (self.n + 1)
        self.running = False
        self._stop_requested = False
        self.run_start_time = None

        self._build_ui()
        self._compute_pegs()
        self._draw_static()

        self.protocol("WM_DELETE_WINDOW", self._on_user_close)

    def _compute_scaling_to_fit(self):
        max_step_x = int(self.canvas_w * 0.06)
        step_x = max(20, min(max_step_x, 56))
        step_y = int(step_x * 0.95)
        top_margin = 60; bottom_margin = 56; min_hist_height = 60; hist_padding = 40
        while True:
            total_pegs_height = (self.n - 1) * step_y if self.n > 0 else 0
            required_v = top_margin + total_pegs_height + step_y + self.desired_hist_height + hist_padding + bottom_margin
            largest_row_width = (self.n - 1) * step_x if self.n > 0 else step_x
            required_w = largest_row_width + 160
            if required_v <= self.canvas_h and required_w <= self.canvas_w:
                break
            step_x -= 2
            if step_x < 20:
                break
            step_y = int(step_x * 0.95)
        self.peg_spacing_x = step_x; self.peg_spacing_y = step_y
        self.peg_radius = max(3, int(step_x * 0.10)); self.ball_radius = max(5, int(step_x * 0.14))
        font_size = max(8, min(18, int(step_x * 0.14))); self.font_pegs = ("Helvetica", font_size, "bold")
        self.top_margin = top_margin; self.bottom_margin = bottom_margin
        self.min_hist_height = min_hist_height; self.hist_padding = hist_padding

    def _build_ui(self):
        screen_w = self.winfo_screenwidth(); screen_h = self.winfo_screenheight()
        x = max(0, (screen_w - self.canvas_w)//2); y = max(0, (screen_h - self.canvas_h)//2 - 20)
        self.geometry(f"{self.canvas_w}x{self.canvas_h}+{x}+{y}")
        self.configure(bg=self.bg_color)

        ctrl = tk.Frame(self, bg=self.bg_color); ctrl.pack(fill="x", padx=6, pady=6)
        self.info_label = tk.Label(ctrl, text=f"Уровни: {self.n}  Шариков: {self.num_balls}  Партия: {self.batch_size}",
                                   bg=self.bg_color, fg=self.text_color, font=("Helvetica", 12, "bold"))
        self.info_label.pack(side="left", padx=6)
        btns = tk.Frame(ctrl, bg=self.bg_color); btns.pack(side="right")
        self.start_btn = tk.Button(btns, text="Start", bg=ACCENT, fg="white", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(btns, text="Stop", bg="#d9534f", fg="white", command=self.stop)
        self.stop_btn.pack(side="left", padx=4)

        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h, bg=self.card_color, highlightthickness=0)
        self.canvas.pack(padx=6, pady=(0,6))
        foot = tk.Label(self, text="© Галтовская анимация", bg=self.bg_color, fg="#b8d6ff", font=("Helvetica", 10, "italic"))
        foot.pack(side="bottom", pady=(0,4))

    def _compute_pegs(self):
        cx = self.canvas_w // 2
        total_pegs_h = (self.n - 1) * self.peg_spacing_y if self.n > 0 else 0
        est_required_for_hist = self.desired_hist_height + self.hist_padding + self.bottom_margin
        avail_for_pegs = self.canvas_h - self.top_margin - est_required_for_hist
        if avail_for_pegs < total_pegs_h:
            max_hist_space = max(self.min_hist_height, self.canvas_h - self.top_margin - total_pegs_h - self.hist_padding - self.bottom_margin)
            self.hist_height = max_hist_space
        else:
            self.hist_height = self.desired_hist_height

        top_y = int((self.canvas_h - (total_pegs_h + self.hist_height + 100)) * 0.5) + 50
        top_y = max(self.top_margin, top_y)
        self.top_y = top_y

        self.peg_positions = []
        for row in range(self.n):
            row_y = top_y + row * self.peg_spacing_y
            count = row + 1
            total_width = (count - 1) * self.peg_spacing_x
            start_x = cx - total_width / 2
            row_positions = []
            for i in range(count):
                x = start_x + i * self.peg_spacing_x; y = row_y
                row_positions.append((x, y))
            self.peg_positions.append(row_positions)

        last_row_y = top_y + (self.n - 1) * self.peg_spacing_y if self.n > 0 else top_y
        bin_y = last_row_y + self.peg_spacing_y + 36

        max_bottom_space = self.canvas_h - 36
        self.hist_top = bin_y + 12
        self.hist_bottom = min(self.hist_top + self.hist_height, max_bottom_space)
        self.hist_height = max(40, self.hist_bottom - self.hist_top)
        self.hist_max_vis_height = max(30, self.hist_bottom - self.hist_top - 12)

        self.bin_centers = []
        total_bins_width = (self.n) * self.peg_spacing_x if self.n > 0 else self.peg_spacing_x
        start_x = cx - total_bins_width / 2
        for k in range(self.n + 1):
            x = start_x + k * self.peg_spacing_x
            self.bin_centers.append((x, bin_y))

    def _draw_static(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_w, 62, fill=ANIM_BG, width=0)
        self.canvas.create_text(18, 30, anchor="w", text="Доска Гальтона — анимация", fill=self.text_color,
                                font=("Helvetica", 14, "bold"))
        for r, row in enumerate(self.peg_positions):
            for k, (x,y) in enumerate(row):
                self.canvas.create_oval(x-self.peg_radius, y-self.peg_radius, x+self.peg_radius, y+self.peg_radius,
                                        fill=self.pegs_color, outline=self.peg_outline)
                try:
                    val = math.comb(r, k)
                except Exception:
                    val = 0
                text_y = y - self.peg_radius - 8
                font_size = max(7, int(self.peg_spacing_x * 0.12))
                font = ("Helvetica", font_size, "bold")
                self.canvas.create_text(x+1, text_y+1, text=str(val), font=font, fill="#02101a")
                self.canvas.create_text(x, text_y, text=str(val), font=font, fill="#ffffff")
        self.bin_rects = []; self.bin_bar_ids = []; self.bin_texts = []
        bw = self.peg_spacing_x * 0.9
        for k, (x, y) in enumerate(self.bin_centers):
            left = x - bw/2; right = x + bw/2; top = self.hist_top; bottom = self.hist_bottom
            self.bin_rects.append((left, top, right, bottom))
            self.canvas.create_rectangle(left, top, right, bottom, outline="#2d4b63", width=1, fill="#08121a")
            bar = self.canvas.create_rectangle(left+6, bottom-6, right-6, bottom-6, fill=self.bar_fill, outline="")
            txt = self.canvas.create_text(x, bottom+18, text="0", fill=self.text_color, font=("Helvetica", 10))
            self.bin_bar_ids.append(bar); self.bin_texts.append(txt)
            self.canvas.create_text(x, bottom+34, text=str(k), fill=self.text_color, font=("Helvetica", 9))

    def _smooth_path(self, pts):
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for i in range(len(pts)-1):
            p0 = pts[i]; p1 = pts[i+1]
            mx = (p0[0] + p1[0]) / 2.0
            my = (p0[1] + p1[1]) / 2.0
            offset = max(6.0, abs(p1[0]-p0[0]) * 0.15)
            out.append((mx, my + offset))
            out.append(p1)
        return out

    def start(self):
        if self.running:
            return
        if self.n > 28:
            if not messagebox.askyesno("Внимание", "Много уровней. Продолжить?"):
                return
        self.running = True; self._stop_requested = False
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.bin_counts = [0]*(self.n+1)
        for k, bar in enumerate(self.bin_bar_ids):
            left, top, right, bottom = self.bin_rects[k]
            self.canvas.coords(bar, left+6, bottom-6, right-6, bottom-6)
            self.canvas.itemconfigure(self.bin_texts[k], text="0")
        self._current_launched = 0
        self.run_start_time = time.time()
        # if release_interval_ms is set -> single release mode (one-by-one at given interval)
        if self.release_interval_ms is not None:
            self._active_balls = []
            self._schedule_release()
            self._continuous_tick()
        else:
            # fallback: batch mode
            self._animate_batches()

    def stop(self):
        self.running = False
        self._stop_requested = True
        self.start_btn.config(state="normal"); self.stop_btn.config(state="normal")

    def _schedule_release(self):
        if (not self.running) or (self._current_launched >= self.num_balls):
            return
        flips = [random.choice([0,1]) for _ in range(self.n)]
        start_x = self.canvas_w // 2; start_y = max(50, self.top_y - 32)
        pts = [(start_x, start_y)]
        rights = 0
        for row_idx in range(self.n):
            peg_x, peg_y = self.peg_positions[row_idx][rights]
            pts.append((peg_x, peg_y))
            if flips[row_idx] == 1:
                rights += 1
        bin_x, bin_y = self.bin_centers[rights]
        pts.append((bin_x, bin_y - 22))
        smooth_pts = self._smooth_path(pts)
        L = polyline_length(smooth_pts)
        tag = f"ball_{self._current_launched}_{random.randint(1,999999)}"
        r = self.ball_radius
        oval = self.canvas.create_oval(start_x-r, start_y-r, start_x+r, start_y+r, fill=self.bar_fill, outline="#7a4f0f", tags=(tag,))
        ball = {"tag": tag, "pts": smooth_pts, "L": L, "s": 0.0, "final_k": rights, "pos": (start_x, start_y), "finished": False}
        self._active_balls.append(ball)
        self._current_launched += 1
        # schedule next release
        if self.running and self._current_launched < self.num_balls:
            self.after(self.release_interval_ms, self._schedule_release)

    def _continuous_tick(self):
        # if stopped and no active balls -> finish
        if (not self.running) and (not getattr(self, '_active_balls', [])):
            duration = time.time() - (self.run_start_time or time.time())
            # call on_finish synchronously so saving can start immediately
            try:
                if callable(self.on_finish):
                    self.on_finish(self.get_counts(), duration)
            except Exception as e:
                print("[ON_FINISH] error:", e)
            self._safe_destroy()
            return

        delta = self.pixels_per_second * (self.frame_ms / 1000.0)
        for b in list(getattr(self, '_active_balls', [])):
            if b["finished"]:
                continue
            b["s"] += delta
            if b["s"] >= b["L"]:
                x,y = b["pts"][-1]; cx,cy = b["pos"]
                dx = x - cx; dy = y - cy
                if abs(dx)>1e-6 or abs(dy)>1e-6:
                    try: self.canvas.move(b["tag"], dx, dy)
                    except: pass
                    b["pos"] = (x,y)
                b["finished"] = True
                k = b["final_k"]; self.bin_counts[k] += 1
                count = self.bin_counts[k]
                probs = binomial_pmf(self.n)
                max_prob = max(probs) if probs else 1.0
                est_peak = max_prob * self.num_balls
                if est_peak > 0:
                    h_per_ball = max(0.2, self.hist_max_vis_height / (est_peak * 1.4))
                else:
                    h_per_ball = 1.0
                left, top, right, bottom = self.bin_rects[k]
                filled_height = min(count * h_per_ball, self.hist_max_vis_height)
                new_top = bottom - 6 - filled_height
                try:
                    self.canvas.coords(self.bin_bar_ids[k], left+6, new_top, right-6, bottom-6)
                    self.canvas.itemconfigure(self.bin_texts[k], text=str(count))
                except Exception:
                    pass
                try: self.canvas.delete(b["tag"])
                except: pass
                try:
                    self._active_balls.remove(b)
                except Exception:
                    pass
            else:
                new_pos = point_at_distance(b["pts"], b["s"])
                cx,cy = b["pos"]
                dx = new_pos[0] - cx; dy = new_pos[1] - cy
                if abs(dx)>1e-6 or abs(dy)>1e-6:
                    try: self.canvas.move(b["tag"], dx, dy)
                    except: pass
                    b["pos"] = new_pos
        # continue ticking or finish if no more work
        if self.running or getattr(self, '_active_balls', []):
            self.after(self.frame_ms, self._continuous_tick)
        else:
            duration = time.time() - (self.run_start_time or time.time())
            try:
                if callable(self.on_finish):
                    self.on_finish(self.get_counts(), duration)
            except Exception as e:
                print("[ON_FINISH] error:", e)
            self._safe_destroy()

    def _animate_batches(self):
        if (not self.running) or (self._current_launched >= self.num_balls):
            self.running = False
            self.start_btn.config(state="normal"); self.stop_btn.config(state="normal")
            duration = time.time() - (self.run_start_time or time.time())
            try:
                if callable(self.on_finish):
                    self.on_finish(self.get_counts(), duration)
            except Exception as e:
                print("[ON_FINISH] error:", e)
            self._safe_destroy()
            return

        remaining = self.num_balls - self._current_launched
        to_launch = min(self.batch_size, remaining)
        batch_id = (self._current_launched // max(1, self.batch_size)) + 1
        self._show_batch_label(f"Партия {batch_id}: {to_launch} шариков", 700)

        balls = []
        for i in range(to_launch):
            flips = [random.choice([0,1]) for _ in range(self.n)]
            start_x = self.canvas_w // 2; start_y = max(50, self.top_y - 32)
            pts = [(start_x, start_y)]
            rights = 0
            for row_idx in range(self.n):
                peg_x, peg_y = self.peg_positions[row_idx][rights]
                pts.append((peg_x, peg_y))
                if flips[row_idx] == 1:
                    rights += 1
            bin_x, bin_y = self.bin_centers[rights]
            pts.append((bin_x, bin_y - 22))
            smooth_pts = self._smooth_path(pts)
            L = polyline_length(smooth_pts)
            tag = f"ball_{batch_id}_{i}_{random.randint(1,999999)}"
            r = self.ball_radius
            oval = self.canvas.create_oval(start_x-r, start_y-r, start_x+r, start_y+r, fill=self.bar_fill, outline="#7a4f0f", tags=(tag,))
            balls.append({"tag": tag, "pts": smooth_pts, "L": L, "s": 0.0, "final_k": rights, "pos": (start_x, start_y), "finished": False})

        def batch_tick():
            if not self.running and self._stop_requested:
                for b in balls:
                    try: self.canvas.delete(b["tag"])
                    except: pass
                duration = time.time() - (self.run_start_time or time.time())
                try:
                    if callable(self.on_finish):
                        self.on_finish(self.get_counts(), duration)
                except Exception as e:
                    print("[ON_FINISH] error:", e)
                self._safe_destroy()
                return

            delta = self.pixels_per_second * (self.frame_ms / 1000.0)
            all_done = True
            for b in balls:
                if b["finished"]:
                    continue
                all_done = False
                b["s"] += delta
                if b["s"] >= b["L"]:
                    x,y = b["pts"][-1]; cx,cy = b["pos"]
                    dx = x - cx; dy = y - cy
                    if abs(dx)>1e-6 or abs(dy)>1e-6:
                        try: self.canvas.move(b["tag"], dx, dy)
                        except: pass
                        b["pos"] = (x,y)
                    b["finished"] = True
                    k = b["final_k"]; self.bin_counts[k] += 1
                    count = self.bin_counts[k]
                    probs = binomial_pmf(self.n)
                    max_prob = max(probs) if probs else 1.0
                    est_peak = max_prob * self.num_balls
                    if est_peak > 0:
                        h_per_ball = max(0.2, self.hist_max_vis_height / (est_peak * 1.4))
                    else:
                        h_per_ball = 1.0
                    left, top, right, bottom = self.bin_rects[k]
                    filled_height = min(count * h_per_ball, self.hist_max_vis_height)
                    new_top = bottom - 6 - filled_height
                    try:
                        self.canvas.coords(self.bin_bar_ids[k], left+6, new_top, right-6, bottom-6)
                        self.canvas.itemconfigure(self.bin_texts[k], text=str(count))
                    except Exception:
                        pass
                    try: self.canvas.delete(b["tag"])
                    except: pass
                else:
                    new_pos = point_at_distance(b["pts"], b["s"])
                    cx,cy = b["pos"]
                    dx = new_pos[0] - cx; dy = new_pos[1] - cy
                    if abs(dx)>1e-6 or abs(dy)>1e-6:
                        try: self.canvas.move(b["tag"], dx, dy)
                        except: pass
                        b["pos"] = new_pos
            if not all_done:
                self.after(self.frame_ms, batch_tick)
            else:
                self._current_launched += to_launch
                if self.running:
                    self.after(int(self.batch_pause * 1000), self._animate_batches)
                else:
                    self.after(50, self._animate_batches)

        self.after(self.frame_ms, batch_tick)

    def _show_batch_label(self, text, duration_ms=700):
        tag = "batch_label"
        self.canvas.delete(tag)
        x = self.canvas_w - 300; y = 28
        self.canvas.create_rectangle(x-6, y-16, x+240, y+16, fill="#083142", outline="", tags=tag)
        self.canvas.create_text(x+117, y, text=text, fill=self.text_color, font=("Helvetica", 10, "bold"), tags=tag)
        def rem():
            try: self.canvas.delete(tag)
            except: pass
        self.after(duration_ms, rem)

    def get_counts(self):
        return self.bin_counts.copy()

    def _on_user_close(self):
        # user requested close -> stop and trigger on_finish immediately then destroy
        self.running = False; self._stop_requested = True
        duration = time.time() - (self.run_start_time or time.time())
        try:
            if callable(self.on_finish):
                self.on_finish(self.get_counts(), duration)
        except Exception as e:
            print("[ON_FINISH] error:", e)
        self._safe_destroy()

    def _safe_destroy(self):
        try:
            if self.winfo_exists():
                self.destroy()
        except Exception:
            pass

# ---------------- Main GUI ----------------
HELP_TEXT = """
От азартных игр к законам Вселенной: увлекательный путь теории вероятностей

XVII век: Рождение из азарта

Представьте: XVII век, Франция. Два математических титана — Блез Паскаль и Пьер Ферма — ведут оживлённую переписку. Повод? Отнюдь не абстрактные теоремы, а скандал среди азартных игроков! Как честно разделить ставку, если игра прервана до финала? Решая эту, казалось бы, приземлённую задачу, они заложили основы теории вероятностей. Главным инструментом расчёта шансов стал треугольник Паскаля — изящная числовая пирамида, позволяющая предсказать исходы множества испытаний.

От идеала к реальности: прорыв Бернулли

Но теория — одно дело, а жизнь — другое. Следующий прорыв совершил Якоб Бернулли. Он задался смелым вопросом: «А работают ли эти расчёты для чего-то большего, чем карты и кости? Для статистики рождений, данных переписей, реального хаоса жизни?» Ответом стал Закон больших чисел: да, хаос отдельных событий при большом их количестве подчиняется строгой закономерности!

Что же происходит? Когда мы многократно повторяем одно и то же случайное испытание (например, подбрасываем монету), совокупность всех исходов начинает описываться биномиальным распределением. Оно точно показывает, как вероятность распределяется между различными количествами «успехов». И вот здесь начинается магия: Закон больших чисел утверждает, что частота успеха будет неуклонно стремиться к предсказанной биномиальным распределением вероятности. Случайность усредняется — проявляется порядок!

Русский вклад: укрощение случайности

Однако блестящую догадку Бернулли нужно было доказать с математической строгостью. Этим занялись гении русской школы. Пафнутий Чебышёв создал своё знаменитое неравенство Чебышёва — мощный инструмент, позволивший доказать, что закон Бернулли — не просто наблюдение, а железный математический факт. Его ученик, Андрей Марков, предложил ещё более универсальную оценку — неравенство Маркова, работающее даже при минимуме информации о ситуации.

Великий синтез: рождение Гауссова колокола

Куда же ведёт эта дорога? К величайшему обобщению! Математики заметили удивительную вещь: стоит нам начать складывать результаты множества независимых испытаний (то есть работать с биномиальным распределением при большом числе экспериментов), как его форма начинает меняться. Грубые «ступеньки» графика сглаживаются, превращаясь в удивительно плавную и симметричную кривую-колокол.

Это и есть нормальное распределение, или «колокол Гаусса». Открытие, известное как Центральная предельная теорема, стало триумфом математической мысли. Вклад в него внесли Карл Фридрих Гаусс, Пьер-Симон Лаплас и Александр Ляпунов. Суть в том, что биномиальное распределение при стремлении числа испытаний к бесконечности не просто стабилизируется — оно стремится к нормальному! Эта универсальная закономерность проявляется повсюду: в ошибках измерений, росте людей, шумах и помехах.

Наша лабораторная работа — машина времени!

За полтора часа вы совершите увлекательное путешествие длиной в три столетия:

1. Рассчитаете вероятность по треугольнику Паскаля, как это делали Паскаль и Ферма.
2. Проверите на опыте Закон больших чисел Бернулли: увидите, как частота стабилизируется, стремясь к предсказаниям биномиального распределения.
3. Оцените риски с помощью универсальных неравенств Чебышёва и Маркова.
4. Станьте свидетелями чуда: увидите своими глазами, как при увеличении числа испытаний «ступеньки» биномиального распределения плавно превращаются в изящный Гауссов колокол — прямое подтверждение Центральной предельной теоремы!

От бытового спора игроков — к универсальным законам, управляющим мирозданием. Вот мощь и красота теории вероятностей
"""

WIKI_ENTRIES = {
    "Блез Паскаль": "https://ru.wikipedia.org/wiki/%D0%9F%D0%B0%D1%81%D0%BA%D0%B0%D0%BB%D1%8C,_%D0%91%D0%BB%D0%B5%D0%B7",
    "Пьер Ферма": "https://ru.wikipedia.org/wiki/%D0%A4%D0%B5%D1%80%D0%BC%D0%B0,_%D0%9F%D1%8C%D0%B5%D1%80",
    "Якоб Бернулли": "https://ru.wikipedia.org/wiki/%D0%91%D0%B5%D1%80%D0%BD%D1%83%D0%BB%D0%BB%D0%B8,_%D0%AF%D0%BA%D0%BE%D0%B1",
    "Пафнутий Чебышёв": "https://ru.wikipedia.org/wiki/%D0%A7%D0%B5%D0%B1%D1%8B%D1%88%D1%91%D0%B2,_%D0%9F%D0%B0%D1%84%D0%BD%D1%83%D1%82%D0%B8%D0%B9",
    "Андрей Марков": "https://ru.wikipedia.org/wiki/%D0%9C%D0%B0%D1%80%D0%BA%D0%BE%D0%B2,_%D0%90%D0%BD%D0%B4%D1%80%D0%B5%D0%B9",
}

class InteractiveGaltonApp:
    def __init__(self, root):
        init_db()
        self.root = root
        self.root.title("Galton Interactive — © Голиков Иван 9М")
        self._build_ui()
        self._preload_runs_folder_into_db()
        self._refresh_runs_list()
        self._update_aggregated_plots()

    def _build_ui(self):
        self.root.geometry("1200x720")
        self.root.configure(bg=BG)
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main); left.pack(side='left', fill='y', padx=(0,6))

        settings = ttk.LabelFrame(left, text="Параметры прогона")
        settings.pack(fill='x', pady=6)
        row = 0
        ttk.Label(settings, text="Уровней (n):").grid(row=row, column=0, sticky='e'); self.e_n = ttk.Entry(settings, width=8); self.e_n.insert(0, "10"); self.e_n.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(settings, text="Шариков:").grid(row=row, column=0, sticky='e'); self.e_balls = ttk.Entry(settings, width=8); self.e_balls.insert(0, "500"); self.e_balls.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(settings, text="Партия (batch_size):").grid(row=row, column=0, sticky='e'); self.e_batch = ttk.Entry(settings, width=8); self.e_batch.insert(0, "50"); self.e_batch.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(settings, text="Скорость (пикс/сек):").grid(row=row, column=0, sticky='e'); self.e_speed = ttk.Entry(settings, width=8); self.e_speed.insert(0, "240"); self.e_speed.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(settings, text="Пауза между партиями (с):").grid(row=row, column=0, sticky='e'); self.e_pause = ttk.Entry(settings, width=8); self.e_pause.insert(0, "0.05"); self.e_pause.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(settings, text="Интервал между шариками (с):").grid(row=row, column=0, sticky='e'); self.e_interval = ttk.Entry(settings, width=8); self.e_interval.insert(0, "0.05"); self.e_interval.grid(row=row, column=1, padx=6, pady=4)

        ttk.Button(settings, text="Справка", command=self._open_help).grid(row=0, column=2, rowspan=2, padx=6)

        btns = ttk.Frame(left); btns.pack(fill='x', pady=6)
        self.btn_run_anim = ttk.Button(btns, text="Запустить анимированный прогон (авто-сохранение)", command=self._on_run_animated)
        self.btn_run_anim.pack(fill='x', pady=2)
        self.btn_run_headless = ttk.Button(btns, text="Быстрый прогон (без анимации)", command=self._on_run_headless)
        self.btn_run_headless.pack(fill='x', pady=2)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(left, textvariable=self.status_var, foreground="#0b6b3b").pack(fill='x', pady=(6,0))

        runs_frame = ttk.LabelFrame(left, text="Сохранённые прогоны")
        runs_frame.pack(fill='both', expand=True, pady=6)
        cols = ("id","ts","n","num_balls","avg_rel_dev_pct","corr","chi2")
        self.tree = ttk.Treeview(runs_frame, columns=cols, show='headings', height=12)
        for c in cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=100, anchor='center')
        self.tree.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(runs_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscroll=sb.set); sb.pack(side='left', fill='y')
        tree_btns = ttk.Frame(runs_frame); tree_btns.pack(side='left', fill='y', padx=6)
        ttk.Button(tree_btns, text="Обновить", command=self._refresh_runs_list).pack(fill='x', pady=2)
        ttk.Button(tree_btns, text="Показать изображения", command=self._view_selected_images).pack(fill='x', pady=2)
        ttk.Button(tree_btns, text="Открыть папку run", command=self._open_selected_run_folder).pack(fill='x', pady=2)
        ttk.Button(tree_btns, text="Экспорт CSV", command=self._export_csv_all).pack(fill='x', pady=2)
        ttk.Button(tree_btns, text="Очистить базу", command=self._clear_db_prompt).pack(fill='x', pady=2)

        ttk.Label(left, text="© Голиков Иван 9М", foreground="#0b4f8a").pack(side='bottom', pady=(8,0))

        right = ttk.Frame(main); right.pack(side='left', fill='both', expand=True)
        nb = ttk.Notebook(right); nb.pack(fill='both', expand=True)

        tab1 = ttk.Frame(nb); nb.add(tab1, text="Среднее ± σ")
        self.fig_mean = Figure(figsize=(6,4), dpi=100); self.ax_mean = self.fig_mean.add_subplot(111)
        self.canvas_mean = FigureCanvasTkAgg(self.fig_mean, master=tab1); self.canvas_mean.get_tk_widget().pack(fill='both', expand=True)

        tab2 = ttk.Frame(nb); nb.add(tab2, text="Частоты vs Биномиал")
        self.fig_norm = Figure(figsize=(6,4), dpi=100); self.ax_norm = self.fig_norm.add_subplot(111)
        self.canvas_norm = FigureCanvasTkAgg(self.fig_norm, master=tab2); self.canvas_norm.get_tk_widget().pack(fill='both', expand=True)

        tab3 = ttk.Frame(nb); nb.add(tab3, text="Распределение отклонений")
        self.fig_dev = Figure(figsize=(6,4), dpi=100); self.ax_dev = self.fig_dev.add_subplot(111)
        self.canvas_dev = FigureCanvasTkAgg(self.fig_dev, master=tab3); self.canvas_dev.get_tk_widget().pack(fill='both', expand=True)

        tab4 = ttk.Frame(nb); nb.add(tab4, text="Таблица Паскаля")
        self.fig_pascal = Figure(figsize=(6,3), dpi=100); self.ax_pascal = self.fig_pascal.add_subplot(111); self.ax_pascal.axis('off')
        self.canvas_pascal = FigureCanvasTkAgg(self.fig_pascal, master=tab4); self.canvas_pascal.get_tk_widget().pack(fill='both', expand=True)

    # --------------- Preload / Runs actions (kept brief) ---------------
    def _preload_runs_folder_into_db(self):
        runs_root = os.path.abspath("runs")
        if not os.path.exists(runs_root):
            return
        db_recs = load_all_runs()
        existing_folders = set()
        for r in db_recs:
            rf = r.get("run_folder")
            if rf:
                existing_folders.add(os.path.abspath(rf))
        imported = 0
        for name in sorted(os.listdir(runs_root)):
            folder = os.path.join(runs_root, name)
            if not os.path.isdir(folder):
                continue
            meta = os.path.join(folder, "metadata.json")
            if not os.path.exists(meta):
                continue
            folder_abs = os.path.abspath(folder)
            if folder_abs in existing_folders:
                continue
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    md = json.load(f)
                counts = md.get("counts", [])
                images = md.get("images", [])
                metrics = md.get("metrics", {})
                n_val = md.get("n", len(counts)-1)
                n_val = max(0, int(n_val))
                params = {"n": n_val, "num_balls": int(md.get("num_balls", sum(counts))), "batch_size": int(md.get("batch_size", 0)), "speed": float(md.get("speed", 0)), "pause": float(md.get("pause", 0))}
                duration = float(md.get("duration", 0.0))
                run_id = save_run_to_db(params, metrics, counts, images, duration, run_folder=folder)
                if run_id:
                    imported += 1
            except Exception as e:
                print("[PRELOAD] import error for", folder, e)
        if imported > 0:
            messagebox.showinfo("Импорт", f"Импортировано {imported} прогона(ов) из папки runs/.")

    def _on_run_animated(self):
        try:
            n = int(self.e_n.get()); num = int(self.e_balls.get())
            batch = int(self.e_batch.get()); speed = float(self.e_speed.get()); pause = float(self.e_pause.get())
            interval = float(self.e_interval.get())
            if n < 1 or num < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка", "Проверьте параметры (целые/числа)."); return
        self._set_running(True)
        self.status_var.set("Запуск анимированного прогона...")
        release_interval = interval if interval > 0 else None
        animator = GaltonBoardAnimator(self.root, n_levels=n, num_balls=num, pixels_per_second=speed,
                                       batch_size=batch, batch_pause=pause, canvas_w=1200, canvas_h=680,
                                       on_finish=lambda counts, duration: self._animator_done_callback(n, num, batch, speed, pause, counts, duration),
                                       release_interval_sec=release_interval)
        animator.bind("<Destroy>", lambda ev, a=animator: self._on_animator_destroy(a, ev))
        animator.start()

    def _on_animator_destroy(self, animator, event):
        try:
            if event.widget is animator:
                self._set_running(False)
                self.status_var.set("Готово (сохранение может выполняться в фоне)")
        except Exception:
            pass

    def _animator_done_callback(self, n, num, batch, speed, pause, counts, duration):
        def progress_cb(step_idx, total_steps, text):
            pct = int((step_idx / float(total_steps)) * 100) if total_steps > 0 else 0
            self.root.after(0, lambda: self._save_progress_dialog.update_progress(pct, text))

        def bg_save():
            try:
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = os.path.join("runs", f"run_{ts_str}")
                counts_list = list(counts) if not isinstance(counts, list) else counts
                image_paths, metrics = generate_and_save_plots(counts_list, n, num, outdir, progress_callback=progress_cb)
                params = {"n": n, "num_balls": num, "batch_size": batch, "speed": speed, "pause": pause}
                self.root.after(0, lambda: self._save_progress_dialog.update_progress(95, "Запись в базу..."))
                run_id = save_run_to_db(params, metrics, counts_list, image_paths, duration, run_folder=outdir)
                self.root.after(0, lambda: self._save_progress_dialog.update_progress(100, "Готово"))
                time.sleep(0.12)
                self.root.after(0, lambda: self._save_progress_dialog.close())
                if run_id:
                    self.root.after(0, lambda: messagebox.showinfo("Сохранено", f"Прогон сохранён в БД (id={run_id}). Файлы: {outdir}"))
                else:
                    self.root.after(0, lambda: messagebox.showwarning("Сохранено частично", f"Файлы сохранены в {outdir}, но запись в БД не создана."))
                self.root.after(0, self._refresh_runs_list)
                self.root.after(0, self._update_aggregated_plots)
            except Exception as exc:
                msg = f"Ошибка при сохранении прогона: {exc}"
                print("[BG SAVE] error:", exc)
                try:
                    self.root.after(0, lambda: self._save_progress_dialog.close())
                except Exception:
                    pass
                self.root.after(0, lambda: messagebox.showerror("Ошибка сохранения", msg))
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self._save_progress_dialog = SaveProgressDialog(self.root, title="Сохранение прогона", max_steps=100)
        self._save_progress_dialog.update_progress(2, "Подготовка...")
        threading.Thread(target=bg_save, daemon=True).start()

    def _on_run_headless(self):
        try:
            n = int(self.e_n.get()); num = int(self.e_balls.get()); batch = int(self.e_batch.get())
            speed = float(self.e_speed.get()); pause = float(self.e_pause.get())
            if n < 1 or num < 1:
                raise ValueError
        except Exception:
            messagebox.showerror("Ошибка", "Проверьте параметры."); return
        self._set_running(True); self.status_var.set("Выполняется headless прогон...")

        def bg():
            try:
                t0 = time.time()
                counts = simulate_headless_counts(n, num)
                duration = time.time() - t0
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                outdir = os.path.join("runs", f"run_{ts_str}")
                dlg = SaveProgressDialog(self.root, title="Сохранение headless прогона")
                dlg.update_progress(5, "Генерация графиков...")
                image_paths, metrics = generate_and_save_plots(counts, n, num, outdir, progress_callback=lambda a,b,c: dlg.update_progress(int((a/b)*100), c))
                dlg.update_progress(95, "Запись в базу...")
                params = {"n": n, "num_balls": num, "batch_size": batch, "speed": speed, "pause": pause}
                run_id = save_run_to_db(params, metrics, list(counts), image_paths, duration, run_folder=outdir)
                dlg.update_progress(100, "Готово")
                time.sleep(0.12)
                dlg.close()
                self.root.after(0, lambda: messagebox.showinfo("Сохранено", f"Headless прогон сохранён (id={run_id})."))
                self.root.after(0, self._refresh_runs_list)
                self.root.after(0, self._update_aggregated_plots)
            except Exception as e:
                print("[HEADLESS] error:", e)
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при headless прогоне: {e}"))
            finally:
                self.root.after(0, lambda: (self._set_running(False), self.status_var.set("Готово")))

        threading.Thread(target=bg, daemon=True).start()

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        for w in (getattr(self, 'e_n', None), getattr(self, 'e_balls', None), getattr(self, 'e_batch', None), getattr(self, 'e_speed', None), getattr(self, 'e_pause', None), getattr(self, 'e_interval', None), getattr(self, 'btn_run_anim', None), getattr(self, 'btn_run_headless', None)):
            try:
                if w: w.config(state=state)
            except Exception:
                pass

    def _refresh_runs_list(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        recs = load_all_runs()
        for r in recs:
            try:
                avg = f"{r['avg_rel_dev_pct']:.4f}" if (r.get('avg_rel_dev_pct') is not None and not math.isnan(r.get('avg_rel_dev_pct')) ) else "nan"
            except Exception:
                avg = "nan"
            try:
                corr = f"{r['corr']:.4f}" if (r.get('corr') is not None and not math.isnan(r.get('corr'))) else "nan"
            except Exception:
                corr = "nan"
            try:
                chi = f"{r['chi2']:.3f}" if (r.get('chi2') is not None and not math.isnan(r.get('chi2'))) else "nan"
            except Exception:
                chi = "nan"
            vals = (r["id"], r["ts"], r["n"], r["num_balls"], avg, corr, chi)
            self.tree.insert("", "end", iid=str(r["id"]), values=vals)

    def _view_selected_images(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Выбор", "Выберите прогон."); return
        run_id = int(sel[0])
        recs = load_all_runs()
        rec = next((r for r in recs if r["id"]==run_id), None)
        if not rec:
            messagebox.showerror("Ошибка", "Запись не найдена."); return
        images = rec.get("images", [])
        opened_any = False
        for p in images:
            p_exp = os.path.expanduser(p)
            if not os.path.isabs(p_exp):
                p_exp = os.path.abspath(p_exp)
            if os.path.exists(p_exp):
                try:
                    open_path(p_exp); opened_any = True
                except Exception as e:
                    print("[OPEN IMAGE] error:", e)
            else:
                candidate = find_file_in_runs_dir(os.path.basename(p))
                if candidate:
                    try:
                        open_path(candidate); opened_any = True
                    except Exception as e:
                        print("[OPEN CAND] error:", e)
        if not opened_any:
            rf = rec.get("run_folder")
            if rf and os.path.exists(rf):
                open_path(rf); return
            runs_root = os.path.abspath("runs")
            if os.path.exists(runs_root):
                open_path(runs_root); return
            messagebox.showerror("Не найдено", "Файлы изображений не обнаружены на диске.")

    def _open_selected_run_folder(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Выбор", "Выберите прогон."); return
        run_id = int(sel[0])
        recs = load_all_runs()
        rec = next((r for r in recs if r["id"]==run_id), None)
        if not rec:
            messagebox.showerror("Ошибка", "Запись не найдена."); return
        images = rec.get("images", [])
        if images:
            candidate = images[0]
            candidate = os.path.expanduser(candidate)
            if not os.path.isabs(candidate):
                candidate = os.path.abspath(candidate)
            folder = os.path.dirname(candidate) if os.path.exists(candidate) else (rec.get("run_folder") or os.path.abspath("runs"))
            if os.path.exists(folder):
                open_path(folder); return
        runs_root = os.path.abspath("runs")
        if os.path.exists(runs_root):
            open_path(runs_root)
        else:
            messagebox.showinfo("Нет папки", "Папка runs не найдена.")

    def _export_csv_all(self):
        recs = load_all_runs()
        if not recs:
            messagebox.showinfo("Нет данных", "Нет сохранённых прогона."); return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not fpath: return
        header = ["run_id","ts","n","num_balls","batch_size","speed","pause","duration","avg_rel_dev_pct","corr","chi2","counts","images","run_folder"]
        try:
            import csv
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(header)
                for r in recs:
                    w.writerow([r["id"], r["ts"], r["n"], r["num_balls"], r.get("batch_size",""), r.get("speed",""), r.get("pause",""), r.get("duration",""), r["avg_rel_dev_pct"], r["corr"], r["chi2"], ";".join(map(str,r["counts"])), ";".join(r["images"]), r.get("run_folder","")])
            messagebox.showinfo("Экспорт", f"Экспорт завершён: {os.path.abspath(fpath)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить CSV: {e}")

    def _clear_db_prompt(self):
        if not messagebox.askyesno("Подтвердите", "Удалить все записи из базы? Это действие необратимо."):
            return
        conn = sqlite_connect(DB_PATH); cur = conn.cursor(); cur.execute("DELETE FROM runs;"); conn.commit(); conn.close()
        self._refresh_runs_list(); self._update_aggregated_plots(); messagebox.showinfo("Готово", "База очищена.")

    # ---------------- Aggregated plots ----------------
    def _update_aggregated_plots(self):
        recs = load_all_runs()
        if not recs:
            for ax in (self.ax_mean, self.ax_norm, self.ax_dev, self.ax_pascal):
                ax.clear()
            self.canvas_mean.draw(); self.canvas_norm.draw(); self.canvas_dev.draw(); self.canvas_pascal.draw()
            return

        max_len = max(len(r["counts"]) for r in recs)
        all_counts = np.zeros((len(recs), max_len), dtype=float)
        for i, r in enumerate(recs):
            cnt = np.array(r["counts"], dtype=float)
            all_counts[i, :len(cnt)] = cnt
        mean_counts = all_counts.mean(axis=0); std_counts = all_counts.std(axis=0, ddof=0)
        n_levels = max_len - 1
        num_runs = all_counts.shape[0]
        num_balls = recs[0].get("num_balls", int(mean_counts.sum())) if recs else int(mean_counts.sum())
        ks = np.arange(0, max_len); probs = binomial_pmf(n_levels) if n_levels>=0 else []

        self.ax_mean.clear()
        self.ax_mean.bar(ks, mean_counts, yerr=std_counts, alpha=0.85, capsize=3, label="Mean ± std", color=BAR_FILL)
        if len(probs) == len(ks):
            self.ax_mean.plot(ks, np.array(probs)*num_balls, marker='o', linestyle='-', color=ACCENT, label="Expected (binomial)")
        self.ax_mean.set_xlabel("k", fontproperties=GLOBAL_FONT_PROP); self.ax_mean.set_ylabel("Среднее число шариков", fontproperties=GLOBAL_FONT_PROP)
        self.ax_mean.set_title(f"Агрегированная гистограмма ({num_runs} прогонов)", fontproperties=GLOBAL_FONT_PROP)
        self.ax_mean.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        self.ax_mean.legend(prop=GLOBAL_FONT_PROP); self.ax_mean.grid(alpha=0.3)
        self.canvas_mean.draw()

        self.ax_norm.clear()
        mean_probs = mean_counts / mean_counts.sum() if mean_counts.sum()>0 else mean_counts
        width = 0.28
        self.ax_norm.bar(ks - width/2, mean_probs, width=width, label="Mean emp. freq", alpha=0.85)
        if len(probs) == len(ks):
            self.ax_norm.bar(ks + width/2, probs, width=width, label="Binomial P(k)", alpha=0.6)
        self.ax_norm.set_xlabel("k", fontproperties=GLOBAL_FONT_PROP); self.ax_norm.set_ylabel("Частота / Вероятность", fontproperties=GLOBAL_FONT_PROP)
        self.ax_norm.set_title("Средние частоты vs теоретическая биномиальная", fontproperties=GLOBAL_FONT_PROP)
        self.ax_norm.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        self.ax_norm.legend(prop=GLOBAL_FONT_PROP); self.ax_norm.grid(alpha=0.3)
        self.canvas_norm.draw()

        self.ax_dev.clear()
        devs = np.array([r.get("avg_rel_dev_pct", float('nan')) for r in recs if r.get("avg_rel_dev_pct") is not None])
        if devs.size > 0:
            self.ax_dev.hist(devs, bins=min(30, max(6, len(devs)//2)), alpha=0.85, color=BAR_FILL)
        self.ax_dev.set_xlabel("Avg relative deviation (%)", fontproperties=GLOBAL_FONT_PROP)
        self.ax_dev.set_title("Распределение AvgRelDev %", fontproperties=GLOBAL_FONT_PROP)
        self.canvas_dev.draw()

        self.ax_pascal.clear()
        coefs = binomial_coefs(n_levels); probs_n = binomial_pmf(n_levels) if n_levels>=0 else []
        col_labels = ["k", "C(n,k)", "P(k)"]
        cell_data = [[str(k), str(c), f"{p:.6f}"] for k,(c,p) in enumerate(zip(coefs, probs_n))] if n_levels>=0 else []
        self.ax_pascal.axis('off')
        if cell_data:
            tbl = self.ax_pascal.table(cellText=cell_data, colLabels=col_labels, cellLoc='left', loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(9)
            for (row,col), cell in tbl.get_celld().items():
                if row == -1:
                    cell.get_text().set_fontproperties(GLOBAL_FONT_PROP)
                else:
                    cell.get_text().set_fontproperties(GLOBAL_MONO_FONT_PROP)
        self.canvas_pascal.draw()

    # ---------------- Help dialog ----------------
    def _open_help(self):
        d = tk.Toplevel(self.root)
        d.title("© Иван Голиков")
        d.geometry("780x640")
        txt = scrolledtext.ScrolledText(d, wrap='word', font=(_CHOSEN_NAME, 11))
        txt.pack(fill='both', expand=True, padx=8, pady=8)
        txt.insert('end', HELP_TEXT + "\n\nОсновные ссылки:\n")
        for name, url in WIKI_ENTRIES.items():
            start = txt.index('end-1c')
            txt.insert('end', f"{name}\n")
            end = txt.index('end-1c')
            txt.tag_add(name, start, end)
            txt.tag_config(name, foreground='blue', underline=1)
            txt.tag_bind(name, '<Button-1>', lambda e, u=url: webbrowser.open(u))
        txt.configure(state='disabled')

# ---------------- Main ----------------
def main():
    init_db()
    root = tk.Tk()
    app = InteractiveGaltonApp(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    main()
