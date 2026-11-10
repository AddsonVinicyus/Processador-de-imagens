import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from color_adjust import ColorAdjustWindow
from spectrum_editor import SpectrumEditorWindow

from ui.models import HistoryEntry
from ui.util.image_utils import (
    CHANNEL_TYPES,
    HIST_ALL,
    HIST_TYPES,
    normalized_histogram,
    to_gray,
)
from ui.operations_catalog import TABS, describe, run_operation
from ui.widgets import HistogramWindow, OpTab, SpectrumWindow
from processing import filtering, frequency_interface

MAX_HISTORY = 40


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Processador de Imagens")
        self.geometry("1520x800")
        self.minsize(1250, 620)

        self.history = []
        self.pos = -1
        self.busy = False
        self.queue = queue.Queue()
        self.tabs = []

        self.hist_window = None
        self.hist_type = HIST_TYPES[0]
        self.hist_token = 0
        self.orig_hist_cache = {}
        self.color_window = None
        self.spec_window = None
        self.spec_token = 0

        self._build()
        self.bind_all("<Control-z>", lambda e: self.grab_current() is None and self.undo())
        self.bind_all("<Control-y>", lambda e: self.grab_current() is None and self.redo())
        self.after(80, self._pump)

    @property
    def current(self):
        return self.history[self.pos].image if self.history else None

    @property
    def original(self):
        return self.history[0].image if self.history else None

    def _reset_history(self, img):
        self.history = [HistoryEntry("Original", img)]
        self.pos = 0

    def _push(self, title, img, raw=None):
        del self.history[self.pos + 1 :]
        self.history.append(HistoryEntry(title, img, raw))
        if len(self.history) > MAX_HISTORY:
            del self.history[1]
        self.pos = len(self.history) - 1

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Style(self).configure("TNotebook.Tab", padding=(6, 4))

        self._build_sidebar()
        self._build_center()
        self._build_history_panel()

        status_bar = ttk.Frame(self)
        status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status = tk.StringVar(value="Abra uma imagem para começar.")
        ttk.Label(
            status_bar,
            textvariable=self.status,
            relief="sunken",
            anchor="w",
            padding=(6, 2),
        ).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=130)
        self.progress.pack(side="right", padx=6)

    def _build_sidebar(self):
        side = ttk.Frame(self, padding=10)
        side.grid(row=0, column=0, sticky="ns")

        file_box = ttk.LabelFrame(side, text="Imagem", padding=8)
        file_box.pack(fill="x")
        ttk.Button(file_box, text="Abrir imagem…", command=self.open_image).pack(fill="x")
        ttk.Button(
            file_box, text="Salvar imagem exibida…", command=self.save_image
        ).pack(fill="x", pady=(4, 0))
        row = ttk.Frame(file_box)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Lado máx. ao abrir (px):").pack(side="left")
        self.max_side = tk.IntVar(value=600)
        ttk.Spinbox(
            row, from_=0, to=8000, increment=100, width=6, textvariable=self.max_side
        ).pack(side="right")
        ttk.Label(
            file_box,
            text="0 = tamanho original (filtros podem demorar)",
            foreground="gray",
        ).pack(anchor="w")

        ops_box = ttk.LabelFrame(side, text="Processamentos", padding=6)
        ops_box.pack(fill="x", pady=(10, 0))
        notebook = ttk.Notebook(ops_box, width=430)
        notebook.pack(fill="x")
        for title, (ops, channel_option) in TABS.items():
            tab = OpTab(notebook, ops, self.apply_operation, channel_option)
            notebook.add(tab, text=title)
            self.tabs.append(tab)

    def _build_center(self):
        self.img_frame = ttk.LabelFrame(self, text="Imagem")
        self.img_frame.grid(row=0, column=1, sticky="nsew", pady=10)
        self.img_box = tk.Frame(self.img_frame)
        self.img_box.pack(fill="both", expand=True)
        self.img_label = tk.Label(self.img_box)
        self.img_label.place(relx=0.5, rely=0.5, anchor="center")
        self.img_box.bind("<Configure>", lambda e: self._show_image())

    def _build_history_panel(self):
        panel = ttk.LabelFrame(self, text="Histórico de processamento", padding=8)
        panel.grid(row=0, column=2, sticky="ns", padx=10, pady=10)

        buttons = ttk.Frame(panel)
        buttons.pack(fill="x")
        self.btn_undo = ttk.Button(buttons, text="↶ Desfazer", command=self.undo)
        self.btn_undo.pack(side="left", expand=True, fill="x")
        self.btn_redo = ttk.Button(buttons, text="↷ Refazer", command=self.redo)
        self.btn_redo.pack(side="left", expand=True, fill="x", padx=(4, 0))

        box = ttk.Frame(panel)
        box.pack(fill="both", expand=True, pady=6)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.hist_list = tk.Listbox(
            box, width=34, exportselection=False, activestyle="none", selectmode="browse"
        )
        self.hist_list.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(box, orient="vertical", command=self.hist_list.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(box, orient="horizontal", command=self.hist_list.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.hist_list.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.hist_list.bind("<<ListboxSelect>>", lambda e: self._on_history_select())
        self._list_fg = self.hist_list.cget("fg")

        self.btn_first = ttk.Button(
            panel, text="Voltar ao original", command=self.go_to_original
        )
        self.btn_first.pack(fill="x")

        self.hist_detail = tk.StringVar(value="")
        ttk.Label(
            panel, textvariable=self.hist_detail, wraplength=250, justify="left"
        ).pack(fill="x", pady=(8, 0))
        ttk.Label(
            panel,
            foreground="gray",
            wraplength=250,
            justify="left",
            text="Clique em uma etapa para reverter a imagem até ela. "
            "Um novo processamento descarta as etapas seguintes.",
        ).pack(fill="x", pady=(8, 0))

        self._update_history_buttons()

    def _show_image(self):
        img = self.current
        if img is None:
            self.img_label.configure(image="")
            self.img_label.image = None
            return
        w = max(self.img_box.winfo_width() - 10, 50)
        h = max(self.img_box.winfo_height() - 10, 50)
        im = Image.fromarray(img)
        im.thumbnail((w, h))
        photo = ImageTk.PhotoImage(im)
        self.img_label.configure(image=photo)
        self.img_label.image = photo

    def preview_image(self, img):
        """Atualiza temporariamente a exibição principal com a pré-visualização do ajuste."""
        self.img_frame.configure(text="Imagem — Pré-visualização (Ajuste de Cores)")
        w = max(self.img_box.winfo_width() - 10, 50)
        h = max(self.img_box.winfo_height() - 10, 50)
        im = Image.fromarray(img)
        im.thumbnail((w, h))
        photo = ImageTk.PhotoImage(im)
        self.img_label.configure(image=photo)
        self.img_label.image = photo

    def end_preview(self):
        """Encerra a pré-visualização e volta a mostrar a etapa atual do histórico."""
        self._update_view()

    def _update_view(self):
        entry = self.history[self.pos]
        self.img_frame.configure(text=f"Imagem — {entry.title}")
        self._show_image()
        h, w = entry.image.shape[:2]
        mode = "tons de cinza" if entry.image.ndim == 2 else "colorido"
        text = f"Etapa {self.pos + 1}/{len(self.history)}  —  {w}x{h}px  —  {mode}"
        if entry.raw is not None:
            text += (
                f"  —  valores brutos de {entry.raw.min():.1f} a {entry.raw.max():.1f}, "
                "normalizados apenas para exibição"
            )
        self.status.set(text)
        self._refresh_history_list()

    def _refresh_history_list(self):
        lb = self.hist_list
        lb.delete(0, "end")
        for i, entry in enumerate(self.history):
            lb.insert("end", f"{i + 1}. {entry.title}")
            lb.itemconfig(i, foreground="gray55" if i > self.pos else self._list_fg)
        if self.history:
            lb.selection_clear(0, "end")
            lb.selection_set(self.pos)
            lb.see(self.pos)
            self.hist_detail.set(self.history[self.pos].title)
        else:
            self.hist_detail.set("")
        self._update_history_buttons()

    def _update_history_buttons(self):
        has = bool(self.history) and not self.busy
        self._set_button(self.btn_undo, has and self.pos > 0)
        self._set_button(self.btn_redo, has and self.pos < len(self.history) - 1)
        self._set_button(self.btn_first, has and self.pos > 0)

    @staticmethod
    def _set_button(button, enabled):
        button.state(["!disabled"] if enabled else ["disabled"])

    def _on_history_select(self):
        selection = self.hist_list.curselection()
        if selection:
            self._goto(selection[0])

    def _goto(self, index):
        if self.busy or not (0 <= index < len(self.history)) or index == self.pos:
            self._refresh_history_list()
            return
        self.pos = index
        self._update_view()
        self._refresh_analysis()

    def undo(self):
        if self.history and self.pos > 0:
            self._goto(self.pos - 1)

    def redo(self):
        if self.history and self.pos < len(self.history) - 1:
            self._goto(self.pos + 1)

    def go_to_original(self):
        if self.history:
            self._goto(0)

    def _set_busy(self, busy):
        self.busy = busy
        for tab in self.tabs:
            tab.set_enabled(not busy)
        self._update_history_buttons()
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _start(self, title, func, reset=False):
        self._set_busy(True)
        self.status.set(f"Processando: {title}…")

        def work():
            try:
                self.queue.put(("image", title, func(), reset))
            except Exception as exc:
                self.queue.put(("error", title, exc))

        threading.Thread(target=work, daemon=True).start()

    def _pump(self):
        try:
            while True:
                self._handle(self.queue.get_nowait())
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _handle(self, msg):
        kind = msg[0]
        if kind == "image":
            _, title, img, reset = msg
            raw = None
            if img.dtype != np.uint8:
                raw = img
                img = filtering.normalize_for_display(raw)
                title += " [exibição normalizada]"
            if reset:
                self._reset_history(img)
            else:
                self._push(title, img, raw)
            self._set_busy(False)
            self._update_view()
            self._refresh_analysis()
        elif kind == "error":
            _, title, exc = msg
            self._set_busy(False)
            self.status.set(f"Erro em: {title}")
            messagebox.showerror("Erro", str(exc))
        elif kind == "hist":
            _, token, tipo, original, current = msg
            if token == self.hist_token and self._alive(self.hist_window):
                self.hist_window.plot(tipo, original, current)
        elif kind == "hist_error":
            _, token, exc = msg
            if token == self.hist_token:
                self.status.set(f"Erro ao calcular o histograma: {exc}")
        elif kind == "editor_data":
            _, data, is_color = msg
            self._set_busy(False)
            self._update_view()
            SpectrumEditorWindow(self, data, is_color, self._commit_external)
        elif kind == "spectrum":
            _, token, gray, spectrum, padded_shape = msg
            if token == self.spec_token and self._alive(self.spec_window):
                self.spec_window.plot(gray, spectrum, padded_shape)
        elif kind == "spectrum_error":
            _, token, exc = msg
            if token == self.spec_token:
                self.status.set(f"Erro ao calcular o espectro: {exc}")

    @staticmethod
    def _alive(window):
        return window is not None and bool(window.winfo_exists())

    def show_histogram(self, tipo="Tons de cinza"):
        if self.current is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        self.hist_type = tipo
        if self._alive(self.hist_window):
            self.hist_window.set_type(tipo)
            self.hist_window.lift()
        else:
            self.hist_window = HistogramWindow(self, tipo, self.show_histogram)
        self._refresh_histogram()

    def _refresh_histogram(self):
        if not self._alive(self.hist_window) or self.current is None:
            return
        self.hist_token += 1
        token, tipo = self.hist_token, self.hist_type
        original, current, cache = self.original, self.current, self.orig_hist_cache
        self.hist_window.set_loading()

        def work():
            try:
                if tipo == HIST_ALL:
                    data = {
                        key: normalized_histogram(current, key) for key in CHANNEL_TYPES
                    }
                    self.queue.put(("hist", token, tipo, None, data))
                else:
                    if tipo not in cache:
                        cache[tipo] = normalized_histogram(original, tipo)
                    self.queue.put(
                        (
                            "hist",
                            token,
                            tipo,
                            cache[tipo],
                            normalized_histogram(current, tipo),
                        )
                    )
            except Exception as exc:
                self.queue.put(("hist_error", token, exc))

        threading.Thread(target=work, daemon=True).start()

    def show_spectrum(self):
        if self.current is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        if self._alive(self.spec_window):
            self.spec_window.lift()
        else:
            self.spec_window = SpectrumWindow(self)
        self._refresh_spectrum()

    def _refresh_spectrum(self):
        if not self._alive(self.spec_window) or self.current is None:
            return
        self.spec_token += 1
        token = self.spec_token
        gray = to_gray(self.current)
        self.spec_window.set_loading()

        def work():
            try:
                spectrum, shape = frequency_interface.fourier_spectrum_image(gray)
                self.queue.put(("spectrum", token, gray, spectrum, shape))
            except Exception as exc:
                self.queue.put(("spectrum_error", token, exc))

        threading.Thread(target=work, daemon=True).start()

    def _refresh_analysis(self):
        self._refresh_histogram()
        self._refresh_spectrum()

    def open_spectrum_editor(self):
        if self.current is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        if self.busy:
            return
        image = self.current
        self._set_busy(True)
        self.status.set("Calculando o espectro para edição…")

        def work():
            try:
                data = frequency_interface.prepare_editor_data(image, to_gray(image))
                self.queue.put(("editor_data", data, image.ndim == 3))
            except Exception as exc:
                self.queue.put(("error", "Editor de espectro", exc))

        threading.Thread(target=work, daemon=True).start()

    def open_color_adjust(self):
        if self.current is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        if self._alive(self.color_window):
            self.color_window.lift()
            return
        self.color_window = ColorAdjustWindow(self, self.current, self._commit_external)

    def _commit_external(self, result, title):
        self._push(title, result)
        self._update_view()
        self._refresh_analysis()

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            im = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Erro ao abrir", str(exc))
            return

        try:
            limit = int(self.max_side.get())
        except (tk.TclError, ValueError):
            limit = 0
        if limit > 0 and max(im.size) > limit:
            im.thumbnail((limit, limit), Image.LANCZOS)

        array = np.array(im, dtype=np.uint8)
        self.orig_hist_cache = {}
        self._start("Original", lambda: array, reset=True)

    def apply_operation(self, name, op, params, per_channel):
        if self.current is None:
            messagebox.showinfo("Sem imagem", "Abra uma imagem primeiro.")
            return
        if op.kind == "action":
            op.fn(self, **params)
            return
        if self.busy:
            return
        title = describe(name, op, params)
        if op.kind == "filter" and per_channel and self.current.ndim == 3:
            title += " [por canal RGB]"
        base = self.current.copy()
        self._start(title, lambda: run_operation(op, base, params, per_channel))

    def save_image(self):
        if self.current is None:
            messagebox.showinfo("Nada para salvar", "Abra uma imagem primeiro.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if path:
            Image.fromarray(self.current).save(path)
            self.status.set(f"Salvo em {path}")


if __name__ == "__main__":
    App().mainloop()