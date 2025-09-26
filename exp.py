# -*- coding: utf-8 -*-
"""
Galton Board — 1366x768 fit (font fix v2)
Таблица Паскаля рендерится как таблица matplotlib с явным FontProperties -> корректный кириллический вывод
© Голиков Иван 9М
Save: galton_board_1366_fit_fontfixed_v2.py
Requires: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import tkinter as tk
from tkinter import messagebox
import math, random, threading
import numpy as np

# ---- Надёжная установка шрифта для корректного вывода кириллицы в matplotlib ----
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_candidates = ["DejaVu Sans", "DejaVu Sans Mono", "Arial", "Liberation Sans", "Tahoma"]

_chosen = None
_chosen_path = None
for _name in _candidates:
    try:
        # try to find font path without falling back to default
        _path = fm.findfont(_name, fallback_to_default=False)
        if _path and _path.lower().endswith((".ttf", ".otf")):
            _chosen = _name
            _chosen_path = _path
            break
    except Exception:
        continue

if _chosen_path is None:
    # final fallback: let matplotlib pick a font (usually DejaVu is bundled)
    _chosen_path = fm.findfont("DejaVu Sans")
    _chosen = fm.FontProperties(fname=_chosen_path).get_name()

# register font file to font manager to be safe
try:
    fm.fontManager.addfont(_chosen_path)
except Exception:
    pass

# set global rcParams to use chosen font name
_chosen_name = fm.FontProperties(fname=_chosen_path).get_name()
matplotlib.rcParams['font.family'] = _chosen_name
matplotlib.rcParams['font.sans-serif'] = [_chosen_name]
matplotlib.rcParams['font.monospace'] = [_chosen_name]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

GLOBAL_FONT_PROP = fm.FontProperties(fname=_chosen_path, size=11)
GLOBAL_MONO_FONT_PROP = fm.FontProperties(fname=_chosen_path, size=10)
# ------------------------------------------------------------------------------------

# ---------- math helpers ----------
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

# ---------- Main App (fitted) ----------
class GaltonBoardFit:
    def __init__(self, n_levels=10, num_balls=1000, pixels_per_second=240.0,
                 batch_size=50, batch_pause=0.5, frame_ms=16,
                 canvas_w=1200, canvas_h=680):
        # fixed canvas tuned for 1366x768 (but safe margin)
        self.canvas_w = int(canvas_w)
        self.canvas_h = int(canvas_h)

        # params
        self.n = max(1, int(n_levels))
        self.num_balls = max(1, int(num_balls))
        self.pixels_per_second = float(pixels_per_second)
        self.batch_size = max(1, int(batch_size))
        self.batch_pause = float(batch_pause)
        self.frame_ms = int(frame_ms)

        # colors
        self.bg_color = "#071226"
        self.card_color = "#0f2740"
        self.text_color = "#e8f1ff"
        self.pegs_color = "#cfe9ff"
        self.bar_fill = "#ffd166"
        self.peg_outline = "#093042"

        # layout placeholders (computed)
        self.peg_spacing_x = 44
        self.peg_spacing_y = 44
        self.peg_radius = 5
        self.ball_radius = 6
        self.font_pegs = ("Helvetica", 10, "bold")
        # histogram height reserved
        self.hist_height = 120

        # compute layout that guarantees fit
        self._compute_scaling_to_fit()

        # internal state
        self.peg_positions = []
        self.bin_centers = []
        self.bin_counts = [0]*(self.n+1)
        self.running = False

        # build UI and draw
        self._build_ui()
        self._compute_pegs()
        self._draw_static()

    def _compute_scaling_to_fit(self):
        # initial guesses
        max_step_x = int(self.canvas_w * 0.06)
        step_x = max(24, min(max_step_x, 56))
        step_y = int(step_x * 0.95)

        top_margin = 60
        bottom_margin = 60
        histogram_required = self.hist_height + 40
        while True:
            total_pegs_height = (self.n - 1) * step_y if self.n>0 else 0
            required_v = top_margin + total_pegs_height + step_y + histogram_required + bottom_margin
            largest_row_width = (self.n - 1) * step_x if self.n>0 else step_x
            required_w = largest_row_width + 160
            if required_v <= self.canvas_h and required_w <= self.canvas_w:
                break
            step_x -= 2
            if step_x < 20:
                break
            step_y = int(step_x * 0.95)
        self.peg_spacing_x = step_x
        self.peg_spacing_y = step_y
        self.peg_radius = max(3, int(step_x * 0.10))
        self.ball_radius = max(4, int(step_x * 0.12))
        font_size = max(8, min(16, int(step_x * 0.14)))
        self.font_pegs = ("Helvetica", font_size, "bold")
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("Galton Board — fit 1366×768 — © Голиков Иван 9М")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(0, (screen_w - self.canvas_w)//2)
        y = max(0, (screen_h - self.canvas_h)//2 - 20)
        self.root.geometry(f"{self.canvas_w}x{self.canvas_h}+{x}+{y}")
        self.root.configure(bg=self.bg_color)

        ctrl = tk.Frame(self.root, bg=self.bg_color)
        ctrl.pack(fill="x", padx=6, pady=6)
        self.info_label = tk.Label(ctrl, text=f"Уровни: {self.n}  Шариков: {self.num_balls}  Партия: {self.batch_size}",
                                   bg=self.bg_color, fg=self.text_color, font=("Helvetica", 12, "bold"))
        self.info_label.pack(side="left", padx=6)
        btns = tk.Frame(ctrl, bg=self.bg_color)
        btns.pack(side="right")
        self.start_btn = tk.Button(btns, text="Start", bg="#39a0ed", fg="white", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = tk.Button(btns, text="Stop", bg="#d9534f", fg="white", command=self.stop)
        self.stop_btn.pack(side="left", padx=4)

        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h,
                                bg=self.card_color, highlightthickness=0)
        self.canvas.pack(padx=6, pady=(0,6))
        foot = tk.Label(self.root, text="© Голиков Иван 9М", bg=self.bg_color, fg="#b8d6ff", font=("Helvetica", 10, "italic"))
        foot.pack(side="bottom", pady=(0,4))

    def _compute_pegs(self):
        cx = self.canvas_w // 2
        total_pegs_h = (self.n - 1) * self.peg_spacing_y if self.n>0 else 0
        top_y = int((self.canvas_h - (total_pegs_h + self.hist_height + 100)) * 0.5) + 50
        top_y = max(self.top_margin, top_y)
        self.top_y = top_y
        self.peg_positions = []
        for row in range(self.n):
            row_y = top_y + row * self.peg_spacing_y
            count = row + 1
            total_width = (count - 1) * self.peg_spacing_x
            start_x = cx - total_width/2
            row_positions = []
            for i in range(count):
                x = start_x + i * self.peg_spacing_x
                y = row_y
                row_positions.append((x, y))
            self.peg_positions.append(row_positions)
        last_row_y = top_y + (self.n - 1) * self.peg_spacing_y if self.n>0 else top_y
        bin_y = last_row_y + self.peg_spacing_y + 36
        self.bin_centers = []
        total_bins_width = (self.n) * self.peg_spacing_x if self.n>0 else self.peg_spacing_x
        start_x = cx - total_bins_width/2
        for k in range(self.n+1):
            x = start_x + k * self.peg_spacing_x
            self.bin_centers.append((x, bin_y))
        self.hist_top = bin_y + 12
        self.hist_bottom = self.hist_top + self.hist_height
        self.hist_max_vis_height = self.hist_bottom - self.hist_top - 12

    def _draw_static(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_w, 62, fill="#071226", width=0)
        self.canvas.create_text(18, 30, anchor="w", text="Доска Гальтона — поместить треугольник + гистограмму", fill=self.text_color,
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
                self.canvas.create_text(x, text_y, text=str(val), font=font, fill="#052238")
        self.bin_rects = []
        self.bin_bar_ids = []
        self.bin_texts = []
        bw = self.peg_spacing_x * 0.9
        for k, (x, y) in enumerate(self.bin_centers):
            left = x - bw/2; right = x + bw/2
            top = self.hist_top; bottom = self.hist_bottom
            self.bin_rects.append((left, top, right, bottom))
            self.canvas.create_rectangle(left, top, right, bottom, outline="#2d4b63", width=1, fill="#08121a")
            bar = self.canvas.create_rectangle(left+6, bottom-6, right-6, bottom-6, fill=self.bar_fill, outline="")
            txt = self.canvas.create_text(x, bottom+18, text="0", fill=self.text_color, font=("Helvetica", 10))
            self.bin_bar_ids.append(bar)
            self.bin_texts.append(txt)
            self.canvas.create_text(x, bottom+34, text=str(k), fill=self.text_color, font=("Helvetica", 9))

    # start / stop
    def start(self):
        if self.running:
            return
        if self.n > 28:
            if not messagebox.askyesno("Внимание", "Много уровней. Продолжить?"):
                return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.bin_counts = [0]*(self.n+1)
        for k, bar in enumerate(self.bin_bar_ids):
            left, top, right, bottom = self.bin_rects[k]
            self.canvas.coords(bar, left+6, bottom-6, right-6, bottom-6)
            self.canvas.itemconfigure(self.bin_texts[k], text="0")
        self._current_launched = 0
        self._animate_batches()

    def stop(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="normal")

    # animate batches: parallel simple ovals
    def _animate_batches(self):
        if (not self.running) or (self._current_launched >= self.num_balls):
            self.running = False
            self.start_btn.config(state="normal")
            threading.Thread(target=self._produce_plots, daemon=True).start()
            return

        remaining = self.num_balls - self._current_launched
        to_launch = min(self.batch_size, remaining)
        batch_id = (self._current_launched // max(1, self.batch_size)) + 1
        self._show_batch_label(f"Партия {batch_id}: {to_launch} шариков", 800)

        balls = []
        for i in range(to_launch):
            flips = [random.choice([0,1]) for _ in range(self.n)]
            start_x = self.canvas_w // 2
            start_y = max(50, self.top_y - 32)
            pts = [(start_x, start_y)]
            rights = 0
            for row_idx in range(self.n):
                peg_idx = rights
                peg_x, peg_y = self.peg_positions[row_idx][peg_idx]
                pts.append((peg_x, peg_y))
                if flips[row_idx] == 1:
                    rights += 1
            bin_x, bin_y = self.bin_centers[rights]
            pts.append((bin_x, bin_y - 22))
            L = polyline_length(pts)
            tag = f"ball_{batch_id}_{i}_{random.randint(1,1_000_000)}"
            r = self.ball_radius
            self.canvas.create_oval(start_x-r, start_y-r, start_x+r, start_y+r, fill="#ffd166", outline="#7a4f0f", tags=(tag,))
            balls.append({"tag": tag, "pts": pts, "L": L, "s": 0.0, "final_k": rights, "pos": (start_x, start_y), "finished": False})

        def batch_tick():
            if not self.running:
                for b in balls:
                    try: self.canvas.delete(b["tag"])
                    except: pass
                return
            delta = self.pixels_per_second * (self.frame_ms / 1000.0)
            all_done = True
            for b in balls:
                if b["finished"]:
                    continue
                all_done = False
                b["s"] += delta
                if b["s"] >= b["L"]:
                    x,y = b["pts"][-1]
                    cx,cy = b["pos"]
                    dx = x - cx; dy = y - cy
                    if abs(dx)>1e-6 or abs(dy)>1e-6:
                        try: self.canvas.move(b["tag"], dx, dy)
                        except: pass
                        b["pos"] = (x,y)
                    b["finished"] = True
                    k = b["final_k"]
                    self.bin_counts[k] += 1
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
                self.root.after(self.frame_ms, batch_tick)
            else:
                self._current_launched += to_launch
                if self.running:
                    self.root.after(int(self.batch_pause * 1000), self._animate_batches)
                else:
                    self.running = False
                    self.start_btn.config(state="normal")
                    threading.Thread(target=self._produce_plots, daemon=True).start()

        self.root.after(self.frame_ms, batch_tick)

    def _show_batch_label(self, text, duration_ms=700):
        tag = "batch_label"
        self.canvas.delete(tag)
        x = self.canvas_w - 240; y = 28
        self.canvas.create_rectangle(x-6, y-16, x+200, y+16, fill="#083142", outline="", tags=tag)
        self.canvas.create_text(x+97, y, text=text, fill=self.text_color, font=("Helvetica", 10, "bold"), tags=tag)
        def rem():
            self.canvas.delete(tag)
        self.root.after(duration_ms, rem)

    # ---------- Final matplotlib plots (table uses ax.table + explicit fontprops) ----------
    def _produce_plots(self):
        coefs = binomial_coefs(self.n)
        probs = binomial_pmf(self.n)
        expected_counts = np.array([p * self.num_balls for p in probs])
        observed = np.array(self.bin_counts, dtype=float)
        expected = expected_counts

        mask = expected > 0
        rel_dev = np.abs(observed[mask] - expected[mask]) / expected[mask] if mask.any() else np.array([])
        avg_dev_pct = np.mean(rel_dev) * 100 if rel_dev.size>0 else float('nan')
        corr = np.corrcoef(observed, expected)[0,1] if (observed.std()>0 and expected.std()>0) else float('nan')

        ks = np.arange(0, self.n+1)

        # 1) Pascal coefficients table — render as actual matplotlib table with explicit FontProperties
        fig1, ax1 = plt.subplots(figsize=(9,3))
        ax1.set_title("Коэффициенты Паскаля и вероятности", fontproperties=GLOBAL_FONT_PROP)
        # prepare cell data
        col_labels = ["k", "C(n,k)", "P(k)"]
        cell_data = []
        for k, (c, p) in enumerate(zip(coefs, probs)):
            cell_data.append([str(k), str(c), f"{p:.6f}"])
        # hide axes
        ax1.axis('off')
        # make table centered
        table = ax1.table(cellText=cell_data, colLabels=col_labels, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        # force fontproperties for header and cells
        # header row has row=-1 in get_celld mapping for matplotlib table
        for (row, col), cell in table.get_celld().items():
            if row ==  -1:
                # header
                cell.get_text().set_fontproperties(GLOBAL_FONT_PROP)
                cell.set_text_props(fontproperties=GLOBAL_FONT_PROP)
            else:
                cell.get_text().set_fontproperties(GLOBAL_MONO_FONT_PROP)
                cell.set_text_props(fontproperties=GLOBAL_MONO_FONT_PROP)
        # adjust layout
        plt.tight_layout()
        plt.show()

        # 2) Metrics (use GLOBAL_FONT_PROP)
        fig2 = plt.figure(figsize=(6,3))
        fig2.suptitle("Метрики эксперимента", fontsize=12, fontproperties=GLOBAL_FONT_PROP)
        metrics = (f"Эксперимент: {self.num_balls} шариков, уровней: {self.n}\n\n"
                   f"Среднее относительное отклонение: {avg_dev_pct:.2f}%\n"
                   f"Корреляция с ожидаемыми (Pascal): {corr:.3f}\n\n"
                   f"Сумма наблюдений: {int(observed.sum())} = {self.num_balls}")
        fig2.text(0.02, 0.45, metrics, fontproperties=GLOBAL_FONT_PROP)
        plt.show()

        # 3) Histogram counts
        fig3 = plt.figure(figsize=(8,4))
        plt.bar(ks, observed, alpha=0.85)
        plt.xticks(ks)
        plt.xlabel("Номер корзины (k)", fontproperties=GLOBAL_FONT_PROP)
        plt.ylabel("Число шариков", fontproperties=GLOBAL_FONT_PROP)
        plt.title("Гистограмма: counts", fontproperties=GLOBAL_FONT_PROP)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

        # 4) Comparison normalized
        fig4 = plt.figure(figsize=(9,4))
        emp_probs = observed / observed.sum() if observed.sum() > 0 else observed
        width = 0.6
        plt.bar(ks - width/3, emp_probs, width=width/3, label="Эксперимент (частоты)", alpha=0.85)
        plt.bar(ks, probs, width=width/3, label="Биномиальное (теория)", alpha=0.6)
        mu = self.n/2.0
        sigma = math.sqrt(self.n/4.0)
        xs = np.linspace(0, self.n, 400)
        ys = [normal_pdf(x, mu, sigma) for x in xs]
        plt.plot(xs, ys, color='green', linewidth=2, label="Нормальная аппроксимация (плотность)")
        plt.xticks(ks)
        plt.xlabel("Номер корзины (k)", fontproperties=GLOBAL_FONT_PROP)
        plt.ylabel("Вероятность / Частота", fontproperties=GLOBAL_FONT_PROP)
        plt.title("Сравнение: нормаль, биномиал, эксперимент", fontproperties=GLOBAL_FONT_PROP)
        plt.legend(prop=GLOBAL_FONT_PROP)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()

        # 5) Observed vs expected counts
        fig5 = plt.figure(figsize=(8,4))
        plt.bar(ks - 0.18, observed, width=0.35, label="Эксперимент (числа)")
        plt.bar(ks + 0.18, expected, width=0.35, label="Ожидаемые (биномиал)")
        plt.xticks(ks)
        plt.xlabel("Номер корзины (k)", fontproperties=GLOBAL_FONT_PROP)
        plt.ylabel("Число шариков", fontproperties=GLOBAL_FONT_PROP)
        plt.title("Наблюдаемое vs Ожидаемое (биномиал)", fontproperties=GLOBAL_FONT_PROP)
        plt.legend(prop=GLOBAL_FONT_PROP)
        plt.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.show()

# ------- Settings dialog -------
def ask_and_run():
    top = tk.Tk()
    top.title("Galton Board — настройки (1366×768 fit)")
    top.geometry("520x320")
    top.configure(bg="#071226")

    tk.Label(top, text="Настройки симуляции", bg="#071226", fg="#e8f1ff", font=("Helvetica", 14, "bold")).pack(pady=8)
    frm = tk.Frame(top, bg="#071226"); frm.pack(padx=12, pady=6)

    tk.Label(frm, text="Уровней (n_levels):", bg="#071226", fg="#dbe9ff").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    n_e = tk.Entry(frm); n_e.insert(0, "10"); n_e.grid(row=0, column=1, padx=4, pady=4)

    tk.Label(frm, text="Кол-во шариков (num_balls):", bg="#071226", fg="#dbe9ff").grid(row=1, column=0, sticky="e", padx=4, pady=4)
    b_e = tk.Entry(frm); b_e.insert(0, "1000"); b_e.grid(row=1, column=1, padx=4, pady=4)

    tk.Label(frm, text="Скорость (пикс/сек, больше = быстрее):", bg="#071226", fg="#dbe9ff").grid(row=2, column=0, sticky="e", padx=4, pady=4)
    s_e = tk.Entry(frm); s_e.insert(0, "240"); s_e.grid(row=2, column=1, padx=4, pady=4)

    tk.Label(frm, text="Размер партии (batch_size):", bg="#071226", fg="#dbe9ff").grid(row=3, column=0, sticky="e", padx=4, pady=4)
    batch_e = tk.Entry(frm); batch_e.insert(0, "50"); batch_e.grid(row=3, column=1, padx=4, pady=4)

    tk.Label(frm, text="Пауза между партиями (сек):", bg="#071226", fg="#dbe9ff").grid(row=4, column=0, sticky="e", padx=4, pady=4)
    pause_e = tk.Entry(frm); pause_e.insert(0, "0.5"); pause_e.grid(row=4, column=1, padx=4, pady=4)

    def start_cb():
        try:
            n = int(n_e.get()); num = int(b_e.get()); speed = float(s_e.get())
            batch = int(batch_e.get()); pause = float(pause_e.get())
            if n < 1 or num < 1 or batch < 1:
                raise ValueError
            top.destroy()
            app = GaltonBoardFit(n_levels=n, num_balls=num, pixels_per_second=speed,
                                 batch_size=batch, batch_pause=pause, frame_ms=16,
                                 canvas_w=1200, canvas_h=680)
            app.root.mainloop()
        except Exception:
            messagebox.showerror("Ошибка", "Проверьте параметры (n, num, batch).")

    tk.Button(top, text="Запустить симуляцию", bg="#39a0ed", fg="white", command=start_cb).pack(pady=10)
    top.mainloop()

if __name__ == "__main__":
    ask_and_run()
