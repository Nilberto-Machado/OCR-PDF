from __future__ import annotations

import argparse
import importlib
import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from tkinter import END, BOTH, LEFT, RIGHT, X, BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from typing import Callable

import pymupdf as fitz


APP_NAME = "PDF OCR"
DEFAULT_LANGUAGE = "por"
DEFAULT_WORKERS = 12
MAX_WORKERS = 20
GPU_OPTION = "GPU AMD (DirectML)"
PROCESSING_OPTIONS = tuple(str(value) for value in range(1, MAX_WORKERS + 1)) + (GPU_OPTION,)


class RapidOcrReader:
    def __init__(self, engine) -> None:
        self.engine = engine

    def readtext(self, image: bytes, detail: int = 1):
        result = self.engine(image)
        if result.boxes is None:
            return []
        return list(zip(result.boxes, result.txts, result.scores))


def create_gpu_reader(language: str):
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        rapidocr = importlib.import_module("rapidocr")
    except ImportError as error:
        raise RuntimeError(
            "O modo GPU AMD requer RapidOCR e ONNX Runtime DirectML. "
            "Instale requirements-gpu.txt."
        ) from error

    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError(
            "O DirectML não está disponível. Atualize o driver AMD e instale "
            "onnxruntime-directml em vez de onnxruntime."
        )

    supported_languages = {"por", "eng", "spa"}
    unsupported = set(language.split("+")) - supported_languages
    if unsupported:
        raise RuntimeError(
            f'O idioma "{sorted(unsupported)[0]}" não é compatível com o modo GPU.'
        )

    engine = rapidocr.RapidOCR(
        params={
            "EngineConfig.onnxruntime.use_dml": True,
            "Rec.ocr_version": rapidocr.OCRVersion.PPOCRV5,
            "Rec.lang_type": rapidocr.LangRec.LATIN,
            "Rec.model_type": rapidocr.ModelType.MOBILE,
        }
    )
    return RapidOcrReader(engine)


def find_tessdata() -> Path:
    candidates = [
        Path(__file__).with_name("tessdata"),
        os.environ.get("TESSDATA_PREFIX"),
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR" / "tessdata",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tessdata",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise RuntimeError(
        "Tesseract OCR não encontrado. Instale-o e defina TESSDATA_PREFIX para a pasta tessdata."
    )


def available_languages(tessdata: Path) -> set[str]:
    return {path.stem for path in tessdata.glob("*.traineddata")}


def install_hint() -> str:
    return (
        "Instale o Tesseract OCR para Windows e marque o idioma Portuguese durante a instalação.\n\n"
        "Depois, reinicie este programa."
    )


def process_page(
    source: Path,
    index: int,
    language: str,
    dpi: int,
    force_ocr: bool,
    tessdata: Path,
) -> tuple[int, bytes | None]:
    with fitz.open(source) as document:
        page = document[index]
        has_images = bool(page.get_images(full=True))
        has_searchable_text = len(page.get_text("text").strip()) >= 20
        if not force_ocr and has_searchable_text and not has_images:
            return index, None

        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return index, pixmap.pdfocr_tobytes(language=language, tessdata=str(tessdata))


def process_page_gpu(
    document: fitz.Document,
    index: int,
    dpi: int,
    force_ocr: bool,
    reader,
) -> bytes | None:
    source_page = document[index]
    has_images = bool(source_page.get_images(full=True))
    has_searchable_text = len(source_page.get_text("text").strip()) >= 20
    if not force_ocr and has_searchable_text and not has_images:
        return None

    pixmap = source_page.get_pixmap(dpi=dpi, alpha=False)
    recognized = reader.readtext(pixmap.tobytes("png"), detail=1)

    page_pdf = fitz.open()
    output_page = page_pdf.new_page(width=source_page.rect.width, height=source_page.rect.height)
    output_page.insert_image(output_page.rect, pixmap=pixmap)
    scale_x = output_page.rect.width / pixmap.width
    scale_y = output_page.rect.height / pixmap.height

    for box, text, confidence in recognized:
        if not text.strip() or confidence < 0.15:
            continue
        left = float(min(point[0] for point in box) * scale_x)
        right = float(max(point[0] for point in box) * scale_x)
        top = float(min(point[1] for point in box) * scale_y)
        bottom = float(max(point[1] for point in box) * scale_y)
        height = max(bottom - top, 1)
        width = max(right - left, 1)
        natural_width = max(fitz.get_text_length(text, fontname="helv", fontsize=1), 0.01)
        font_size = max(1, min(height * 0.8, width / natural_width))
        output_page.insert_text(
            (left, min(bottom, output_page.rect.height)),
            text,
            fontname="helv",
            fontsize=font_size,
            render_mode=3,
        )

    page_data = page_pdf.tobytes(garbage=4, deflate=True)
    page_pdf.close()
    return page_data


def ocr_pdf(
    source: Path,
    destination: Path,
    language: str = DEFAULT_LANGUAGE,
    dpi: int = 300,
    force_ocr: bool = False,
    progress: Callable[[int, int], None] | None = None,
    workers: int = DEFAULT_WORKERS,
    use_gpu: bool = False,
) -> None:
    tessdata = None
    gpu_reader = None
    if use_gpu:
        gpu_reader = create_gpu_reader(language)
    else:
        tessdata = find_tessdata()
        languages = available_languages(tessdata)
        if language not in languages:
            raise RuntimeError(
                f'O idioma "{language}" não está instalado no Tesseract. Idiomas disponíveis: '
                + ", ".join(sorted(languages))
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ocr.tmp.pdf")
    output = fitz.open()

    try:
        with fitz.open(source) as document:
            total = document.page_count
            if use_gpu:
                for index in range(total):
                    page_data = process_page_gpu(
                        document, index, dpi, force_ocr, gpu_reader
                    )
                    if page_data is None:
                        output.insert_pdf(document, from_page=index, to_page=index)
                    else:
                        page_pdf = fitz.open("pdf", page_data)
                        output.insert_pdf(page_pdf)
                        page_pdf.close()
                    if progress:
                        progress(index + 1, total)
            else:
                workers = max(1, min(workers, total, MAX_WORKERS))
                window_size = workers * 2
                pending_pages: dict[int, bytes | None] = {}
                next_to_submit = 0
                next_to_write = 0

                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    while next_to_write < total:
                        while next_to_submit < total and next_to_submit < next_to_write + window_size:
                            future = executor.submit(
                                process_page,
                                source,
                                next_to_submit,
                                language,
                                dpi,
                                force_ocr,
                                tessdata,
                            )
                            futures[future] = next_to_submit
                            next_to_submit += 1

                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            index, page_data = future.result()
                            pending_pages[index] = page_data
                            del futures[future]

                        while next_to_write in pending_pages:
                            page_data = pending_pages.pop(next_to_write)
                            if page_data is None:
                                output.insert_pdf(document, from_page=next_to_write, to_page=next_to_write)
                            else:
                                page_pdf = fitz.open("pdf", page_data)
                                output.insert_pdf(page_pdf)
                                page_pdf.close()
                            next_to_write += 1
                            if progress:
                                progress(next_to_write, total)

        output.save(temporary, garbage=4, deflate=True)
        output.close()
        temporary.replace(destination)
    except Exception:
        output.close()
        temporary.unlink(missing_ok=True)
        raise


class OcrApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("760x520")
        self.root.minsize(640, 440)
        self.files: list[Path] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.output_dir = StringVar(value=str(Path.home() / "Documents" / "PDFs com OCR"))
        self.language = StringVar(value=DEFAULT_LANGUAGE)
        self.dpi = StringVar(value="300")
        self.workers = StringVar(value=str(DEFAULT_WORKERS))
        self.force_ocr = BooleanVar(value=False)
        self.status = StringVar(value="Adicione um ou mais arquivos PDF.")
        self._build_ui()
        self.root.after(100, self._consume_events)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill=BOTH, expand=True)

        ttk.Label(frame, text="Arquivos PDF", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=BOTH, expand=True, pady=(8, 10))
        self.file_list = __import__("tkinter").Listbox(list_frame, selectmode="extended")
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, command=self.file_list.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        self.file_list.configure(yscrollcommand=scrollbar.set)

        file_buttons = ttk.Frame(frame)
        file_buttons.pack(fill=X, pady=(0, 14))
        ttk.Button(file_buttons, text="Adicionar PDFs", command=self._add_files).pack(side=LEFT)
        ttk.Button(file_buttons, text="Remover selecionados", command=self._remove_selected).pack(side=LEFT, padx=8)
        ttk.Button(file_buttons, text="Limpar", command=self._clear_files).pack(side=LEFT)

        options = ttk.LabelFrame(frame, text="Opções", padding=12)
        options.pack(fill=X)
        ttk.Label(options, text="Pasta de saída").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.output_dir).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        ttk.Button(options, text="Escolher...", command=self._choose_output).grid(row=1, column=3, padx=(8, 0), pady=(4, 10))
        ttk.Label(options, text="Idioma").grid(row=2, column=0, sticky="w")
        ttk.Combobox(options, textvariable=self.language, values=("por", "eng", "spa", "por+eng"), width=12).grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Label(options, text="Qualidade (DPI)").grid(row=2, column=1, sticky="w", padx=(12, 0))
        ttk.Combobox(options, textvariable=self.dpi, values=("200", "300", "400"), width=8, state="readonly").grid(row=3, column=1, sticky="w", padx=(12, 0), pady=(4, 0))
        ttk.Label(options, text="Processamento").grid(row=2, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(options, textvariable=self.workers, values=PROCESSING_OPTIONS, width=20, state="readonly").grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(4, 0))
        ttk.Checkbutton(options, text="Aplicar OCR mesmo em páginas que já têm texto", variable=self.force_ocr).grid(row=4, column=0, columnspan=4, sticky="w", pady=(10, 0))
        options.columnconfigure(0, weight=1)
        options.columnconfigure(2, weight=1)

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill=X, pady=(16, 6))
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        self.start_button = ttk.Button(frame, text="Executar OCR", command=self._start)
        self.start_button.pack(anchor="e", pady=(10, 0))

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Selecionar PDFs", filetypes=(("Arquivos PDF", "*.pdf"),))
        for raw_path in paths:
            path = Path(raw_path)
            if path not in self.files:
                self.files.append(path)
                self.file_list.insert(END, str(path))

    def _remove_selected(self) -> None:
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)
            del self.files[index]

    def _clear_files(self) -> None:
        self.files.clear()
        self.file_list.delete(0, END)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Escolher pasta de saída")
        if path:
            self.output_dir.set(path)

    def _start(self) -> None:
        if not self.files:
            messagebox.showwarning(APP_NAME, "Adicione pelo menos um arquivo PDF.")
            return
        try:
            dpi = int(self.dpi.get())
            use_gpu = self.workers.get() == GPU_OPTION
            workers = DEFAULT_WORKERS if use_gpu else int(self.workers.get())
        except ValueError:
            messagebox.showerror(APP_NAME, "Informe valores válidos de DPI e processos.")
            return
        self.start_button.configure(state="disabled")
        self.progress.configure(value=0)
        settings = (
            tuple(self.files),
            Path(self.output_dir.get()),
            self.language.get(),
            self.force_ocr.get(),
        )
        threading.Thread(
            target=self._process, args=(dpi, workers, use_gpu, *settings), daemon=True
        ).start()

    def _process(
        self,
        dpi: int,
        workers: int,
        use_gpu: bool,
        files: tuple[Path, ...],
        output_dir: Path,
        language: str,
        force_ocr: bool,
    ) -> None:
        try:
            for file_index, source in enumerate(files):
                destination = output_dir / f"{source.stem}_OCR.pdf"

                def report(page: int, total: int) -> None:
                    overall = ((file_index + page / total) / len(files)) * 100
                    self.events.put(("progress", (overall, source.name, page, total)))

                ocr_pdf(
                    source, destination, language, dpi, force_ocr, report, workers, use_gpu
                )
            self.events.put(("done", output_dir))
        except Exception as error:
            self.events.put(("error", error))

    def _consume_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "progress":
                    percent, name, page, total = value
                    self.progress.configure(value=percent)
                    self.status.set(f"{name}: página {page} de {total}")
                elif event == "done":
                    self.start_button.configure(state="normal")
                    self.status.set("OCR concluído.")
                    messagebox.showinfo(APP_NAME, f"Arquivos salvos em:\n{value}")
                elif event == "error":
                    self.start_button.configure(state="normal")
                    self.status.set("Não foi possível concluir o OCR.")
                    hint = f"\n\n{install_hint()}" if "Tesseract" in str(value) else ""
                    messagebox.showerror(APP_NAME, f"{value}{hint}")
        except queue.Empty:
            pass
        self.root.after(100, self._consume_events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte PDFs em PDFs pesquisáveis usando OCR.")
    parser.add_argument("files", nargs="*", type=Path, help="Arquivos PDF de entrada")
    parser.add_argument("-o", "--output", type=Path, help="Pasta de saída")
    parser.add_argument("-l", "--language", default=DEFAULT_LANGUAGE, help="Idioma do Tesseract")
    parser.add_argument("--dpi", type=int, default=300, choices=(200, 300, 400))
    parser.add_argument("-j", "--workers", type=int, default=DEFAULT_WORKERS, choices=range(1, MAX_WORKERS + 1))
    parser.add_argument("--gpu", action="store_true", help="Usa RapidOCR com GPU via DirectML")
    parser.add_argument("--force", action="store_true", help="Aplica OCR também em páginas com texto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.files:
        output_dir = args.output or Path.cwd() / "saida_ocr"
        for source in args.files:
            destination = output_dir / f"{source.stem}_OCR.pdf"
            print(f"Processando {source}...")
            ocr_pdf(
                source,
                destination,
                args.language,
                args.dpi,
                args.force,
                workers=args.workers,
                use_gpu=args.gpu,
            )
            print(f"Salvo em {destination}")
        return

    root = Tk()
    OcrApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()