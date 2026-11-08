import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image

from ui.util.image_utils import place_over_sidebar
from processing import colors_fast


def _as_rgb(img):
    return img if img.ndim == 3 else np.stack([img] * 3, axis=-1)


class ColorAdjustWindow(tk.Toplevel):
    PREVIEW_SIDE = 700 

    def __init__(self, master, image, on_apply):
        super().__init__(master)
        self.title("Ajuste de cores (HSI)")
        self.transient(master)
        self.resizable(False, False)

        self.on_apply = on_apply
        self.full = _as_rgb(image)

        thumb = Image.fromarray(self.full)
        thumb.thumbnail((self.PREVIEW_SIDE, self.PREVIEW_SIDE))
        self.small = np.asarray(thumb, dtype=np.uint8)
        self.small_hsi = colors_fast.rgb2hsi_array(self.small)  # calculado uma única vez

        self.balance = {faixa: [0, 0, 0] for faixa in colors_fast.RANGES}
        self._job = None
        self._loading = False

        self._build()
        place_over_sidebar(self, master)

        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

    def _build(self):
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack()

        self.hue = tk.IntVar(value=0)
        self.sat = tk.IntVar(value=0)
        self.bri = tk.IntVar(value=0)
        self.cr = tk.IntVar(value=0)
        self.mg = tk.IntVar(value=0)
        self.yb = tk.IntVar(value=0)
        self.sepia = tk.IntVar(value=0)

        hsi = ttk.LabelFrame(ctrl, text="HSI", padding=6)
        hsi.pack(fill="x")
        self._slider(hsi, "Matiz (°)", self.hue, -180, 180)
        self._slider(hsi, "Saturação", self.sat, -100, 100)
        self._slider(hsi, "Brilho (intensidade)", self.bri, -100, 100)

        bal = ttk.LabelFrame(ctrl, text="Balanço de cores", padding=6)
        bal.pack(fill="x", pady=8)
        row = ttk.Frame(bal)
        row.pack(fill="x")
        ttk.Label(row, text="Faixa tonal:").pack(side="left")
        self.range_var = tk.StringVar(value=colors_fast.MIDTONES)
        combo = ttk.Combobox(row, textvariable=self.range_var, values=colors_fast.RANGES,
                             state="readonly", width=14)
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda e: self._load_range())
        self._slider(bal, "Ciano  ↔  Vermelho", self.cr, -100, 100)
        self._slider(bal, "Magenta  ↔  Verde", self.mg, -100, 100)
        self._slider(bal, "Amarelo  ↔  Azul", self.yb, -100, 100)
        self._load_range()

        buttons = ttk.Frame(ctrl)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Redefinir", command=self._reset).pack(side="left")
        ttk.Button(buttons, text="Aplicar", command=self._apply).pack(side="right")
        ttk.Button(buttons, text="Cancelar", command=self.destroy).pack(side="right", padx=6)

    def _slider(self, parent, text, var, lo, hi):
        scale = tk.Scale(parent, from_=lo, to=hi, orient="horizontal", label=text,
                         variable=var, length=300, resolution=1,
                         command=lambda _value: self._on_change())
        scale.pack(fill="x")
        return scale

    def _on_change(self):
        if self._loading:
            return
        self.balance[self.range_var.get()] = [self.cr.get(), self.mg.get(), self.yb.get()]
        self._schedule()

    def _load_range(self):
        self._loading = True
        cr, mg, yb = self.balance[self.range_var.get()]
        self.cr.set(cr)
        self.mg.set(mg)
        self.yb.set(yb)
        self.after_idle(self._end_loading)

    def _end_loading(self):
        self._loading = False

    def _reset(self):
        self._loading = True
        for var in (self.hue, self.sat, self.bri, self.cr, self.mg, self.yb, self.sepia):
            var.set(0)
        self.balance = {faixa: [0, 0, 0] for faixa in colors_fast.RANGES}
        self.after_idle(self._end_loading)
        self._schedule()

    def _schedule(self):
        if self._job is not None:
            self.after_cancel(self._job)
        self._job = self.after(20, self._update_preview)

    def _adjust(self, rgb, hsi):
        return colors_fast.adjust(
            rgb, hsi,
            hue=self.hue.get(), saturation=self.sat.get(), brightness=self.bri.get(),
            balance=self.balance, sepia=self.sepia.get())

    def _describe(self):
        parts = []
        if self.hue.get():
            parts.append(f"matiz {self.hue.get():+d}°")
        if self.sat.get():
            parts.append(f"saturação {self.sat.get():+d}")
        if self.bri.get():
            parts.append(f"brilho {self.bri.get():+d}")
        if any(any(values) for values in self.balance.values()):
            parts.append("balanço de cores")
        return "Ajuste de cores (" + (", ".join(parts) if parts else "sem alterações") + ")"

    def _update_preview(self):
        self._job = None
        self.master.preview_image(self._adjust(self.small, self.small_hsi))

    def _apply(self):
        self.config(cursor="watch")
        self.update_idletasks()
        result = self._adjust(self.full, None)
        self.on_apply(result, self._describe())
        self.destroy()

    def destroy(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None
        self.master.end_preview() 
        super().destroy()