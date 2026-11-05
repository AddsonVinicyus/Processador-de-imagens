import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ui.util.image_utils import CHANNEL_TYPES, HIST_ALL, HIST_TYPES

#Layout vertical da lateral esquerda
class OpTab(ttk.Frame):
    def __init__(self, master, ops, on_apply, channel_option):
        super().__init__(master, padding=8)
        self.ops = ops
        self.on_apply = on_apply
        self.param_vars = {}

        ttk.Label(self, text="Operação").pack(anchor="w")
        self.name = tk.StringVar(value=next(iter(ops)))
        combo = ttk.Combobox(self, textvariable=self.name, state="readonly", values=list(ops))
        combo.pack(fill="x", pady=(2, 8))
        combo.bind("<<ComboboxSelected>>", lambda e: self._build_params())

        self.params_frame = ttk.Frame(self)
        self.params_frame.pack(fill="x")

        self.per_channel = tk.BooleanVar(value=False)
        if channel_option:
            ttk.Checkbutton(
                self,
                text="Aplicar em cada canal RGB (mais lento)",
                variable=self.per_channel,
            ).pack(anchor="w", pady=(8, 0))

        self.button = ttk.Button(self, text="Aplicar", command=self._apply)
        self.button.pack(fill="x", pady=(14, 0))

        self._build_params()

    def _build_params(self):
        op = self.ops[self.name.get()]
        self.button.configure(text="Abrir" if op.kind == "action" else "Aplicar")
        for w in self.params_frame.winfo_children():
            w.destroy()
        self.param_vars = {}
        for p in op.params:
            row = ttk.Frame(self.params_frame)
            row.pack(fill="x", pady=2)
            if p.kind == "bool":
                var = tk.BooleanVar(value=p.default)
                ttk.Checkbutton(row, text=p.label, variable=var).pack(side="left")
            else:
                ttk.Label(row, text=p.label).pack(side="left")
                if p.kind == "choice":
                    var = tk.StringVar(value=p.default)
                    width = max(14, max(len(c) for c in p.choices) + 1)
                    ttk.Combobox(
                        row,
                        textvariable=var,
                        values=p.choices,
                        state="readonly",
                        width=width,
                    ).pack(side="right")
                else:
                    var = (
                        tk.IntVar(value=p.default)
                        if p.kind == "int"
                        else tk.DoubleVar(value=p.default)
                    )
                    ttk.Spinbox(
                        row,
                        from_=p.lo,
                        to=p.hi,
                        increment=p.step,
                        textvariable=var,
                        width=8,
                    ).pack(side="right")
            self.param_vars[p.key] = var

    def _collect(self, op):
        values = {}
        for p in op.params:
            try:
                v = self.param_vars[p.key].get()
            except tk.TclError:
                raise ValueError(f"Valor inválido em “{p.label}”.")
            if p.kind in ("int", "float"):
                v = max(p.lo, min(p.hi, v))
                if p.odd:
                    v = int(v)
                    v = v if v % 2 else v + 1
            values[p.key] = v
        return values

    def _apply(self):
        name = self.name.get()
        op = self.ops[name]
        try:
            params = self._collect(op)
        except ValueError as exc:
            messagebox.showwarning("Parâmetro inválido", str(exc))
            return
        self.on_apply(name, op, params, self.per_channel.get())

    def set_enabled(self, enabled):
        self.button.state(["!disabled"] if enabled else ["disabled"])

#Janela de histogramas
class HistogramWindow(tk.Toplevel):
    COLORS = {
        "Tons de cinza": "gray",
        "R": "tab:red",
        "G": "tab:green",
        "B": "tab:blue",
        "I (HSI)": "dimgray",
    }

    def __init__(self, master, tipo, on_type):
        super().__init__(master)
        self.title("Histogramas")
        self.geometry("980x540")

        top = ttk.Frame(self, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Canal:").pack(side="left")
        self.type_var = tk.StringVar(value=tipo)
        combo = ttk.Combobox(
            top, textvariable=self.type_var, values=HIST_TYPES, state="readonly", width=22
        )
        combo.pack(side="left", padx=6)
        combo.bind("<<ComboboxSelected>>", lambda e: on_type(self.type_var.get()))

        self.fig = Figure(figsize=(9, 4.6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.info = tk.StringVar(value="Calculando…")
        ttk.Label(self, textvariable=self.info, anchor="w", padding=(6, 2)).pack(fill="x")

    def set_type(self, tipo):
        self.type_var.set(tipo)

    def set_loading(self):
        self.info.set("Calculando…")

    @staticmethod
    def _bar(ax, data, title, color):
        ax.bar(np.arange(256), data, width=1.0, color=color)
        ax.set_title(title)
        ax.set_xlim(-1, 256)
        ax.set_xlabel("Intensidade")
        ax.set_ylabel("Frequência normalizada")

    def plot(self, tipo, original, current):
        """Um canal: original x atual. Todos: R, G, B e I da imagem atual (2x2)."""
        self.fig.clear()
        if tipo == HIST_ALL:
            for k, key in enumerate(CHANNEL_TYPES, start=1):
                self._bar(
                    self.fig.add_subplot(2, 2, k),
                    current[key],
                    f"{key} — imagem atual",
                    self.COLORS[key],
                )
        else:
            color = self.COLORS[tipo]
            self._bar(self.fig.add_subplot(1, 2, 1), original, f"Original — {tipo}", color)
            self._bar(
                self.fig.add_subplot(1, 2, 2), current, f"Imagem atual — {tipo}", color
            )
        self.canvas.draw_idle()
        self.info.set(
            "Atualizado a cada etapa do histórico enquanto esta janela estiver aberta."
        )

#Janela do espectro de Fourier
class SpectrumWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Espectro de Fourier")
        self.geometry("980x520")

        self.fig = Figure(figsize=(9, 4.6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.info = tk.StringVar(value="Calculando…")
        ttk.Label(self, textvariable=self.info, anchor="w", padding=(6, 2)).pack(fill="x")

    def set_loading(self):
        self.info.set("Calculando…")

    def plot(self, gray, spectrum, padded_shape):
        self.fig.clear()
        ax_img = self.fig.add_subplot(1, 2, 1)
        ax_img.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title("Imagem atual (tons de cinza)")
        ax_img.axis("off")

        ax_spec = self.fig.add_subplot(1, 2, 2)
        ax_spec.imshow(spectrum, cmap="gray", vmin=0, vmax=255)
        ph, pw = padded_shape
        ax_spec.set_title("Espectro de Fourier — log(1 + |F|), centralizado")
        ax_spec.set_xlabel(f"v (colunas) — espectro {pw}x{ph}")
        ax_spec.set_ylabel("u (linhas)")
        self.canvas.draw_idle()
        gh, gw = gray.shape
        padded = (ph, pw) != (gh, gw)
        self.info.set(
            f"Imagem {gw}x{gh}"
            + (f", com zero padding até {pw}x{ph} (potência de 2)" if padded else "")
            + ". A frequência zero (componente DC) fica no centro. "
            "Atualizado a cada etapa do histórico enquanto esta janela estiver aberta."
        )