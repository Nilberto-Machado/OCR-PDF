from __future__ import annotations

import argparse
import importlib
import math
import os
import queue
import subprocess
import threading
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from tkinter import END, BOTH, LEFT, RIGHT, X, BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from typing import Callable

import pymupdf as fitz


APP_NAME = "PDF OCR"
DEFAULT_LANGUAGE = "por"
AVAILABLE_CPUS = (
    os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
) or 1
MAX_WORKERS = max(1, AVAILABLE_CPUS)
DEFAULT_WORKERS = min(12, MAX_WORKERS)
MAX_ENHANCED_WORKERS = min(20, MAX_WORKERS)
GPU_OPTION = "GPU AMD (DirectML)"
PROCESSING_OPTIONS = tuple(str(value) for value in range(1, MAX_WORKERS + 1)) + (GPU_OPTION,)


class OcrCancelled(Exception):
    pass


OcrWord = tuple[float, float, float, float, str]
OcrLayer = tuple[float, float, list[OcrWord]]
Orientation = tuple[bool, int]
DetectedOrientation = tuple[Orientation, float]
VisualCorrection = tuple[bytes, float, float]
PageResult = tuple[int, OcrLayer | None, VisualCorrection | None]
ORIENTATIONS: tuple[Orientation, ...] = tuple(
    (mirrored, rotation)
    for mirrored in (False, True)
    for rotation in (0, 90, 180, 270)
)
NORMAL_ORIENTATION: Orientation = (False, 0)
NORMAL_DETECTED_ORIENTATION: DetectedOrientation = (NORMAL_ORIENTATION, 0.0)
COMMON_WORDS = {
    "por": {
        "ao", "aos", "as", "com", "da", "das", "de", "do", "dos", "e",
        "em", "esta", "este", "na", "nas", "no", "nos", "ou", "para", "por",
        "que", "se", "sem", "uma", "um",
    },
    "eng": {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of",
        "on", "or", "that", "the", "this", "to", "with", "without",
    },
    "spa": {
        "al", "con", "de", "del", "el", "en", "esta", "este", "la", "las",
        "los", "o", "para", "por", "que", "se", "sin", "un", "una", "y",
    },
}


def transform_image(image, orientation: Orientation, cv2):
    mirrored, rotation = orientation
    transformed = cv2.flip(image, 1) if mirrored else image
    rotations = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotations[rotation]
    return cv2.rotate(transformed, rotate_code) if rotate_code is not None else transformed


def rotate_image(image, angle: float, cv2):
    if abs(angle) < 0.05:
        return image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = int(height * sine + width * cosine)
    output_height = int(height * cosine + width * sine)
    matrix[0, 2] += output_width / 2 - center[0]
    matrix[1, 2] += output_height / 2 - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def estimate_text_axis_angle(image, cv2) -> float | None:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    shortest_side = min(image.shape[:2])
    detected = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 360,
        threshold=max(20, shortest_side // 25),
        minLineLength=max(20, shortest_side // 18),
        maxLineGap=max(5, shortest_side // 80),
    )
    if detected is None:
        return None

    observations: list[tuple[float, float]] = []
    angle_bins: dict[int, float] = {}
    for x1, y1, x2, y2 in detected.reshape(-1, 4):
        delta_x = int(x2) - int(x1)
        delta_y = int(y2) - int(y1)
        angle = math.degrees(math.atan2(delta_y, delta_x)) % 180
        weight = math.hypot(delta_x, delta_y)
        bucket = round(angle / 3) % 60
        observations.append((angle, weight))
        angle_bins[bucket] = angle_bins.get(bucket, 0) + weight

    dominant_angle = max(angle_bins, key=angle_bins.get) * 3
    nearby = [
        (angle, weight)
        for angle, weight in observations
        if abs((angle - dominant_angle + 90) % 180 - 90) <= 4.5
    ]
    sine = sum(
        math.sin(math.radians(2 * angle)) * weight for angle, weight in nearby
    )
    cosine = sum(
        math.cos(math.radians(2 * angle)) * weight for angle, weight in nearby
    )
    return math.degrees(math.atan2(sine, cosine)) / 2


def estimate_skew_correction(image, cv2) -> float:
    axis_angle = estimate_text_axis_angle(image, cv2)
    if axis_angle is None:
        return 0.0
    while axis_angle > 45:
        axis_angle -= 90
    while axis_angle <= -45:
        axis_angle += 90
    correction = axis_angle
    return 0.0 if abs(correction) < 0.4 else correction


def language_word_bonus(texts: list[str], language: str) -> int:
    common_words = set().union(
        *(COMMON_WORDS.get(code, set()) for code in language.split("+"))
    )
    matches = 0
    for text in texts:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        word = "".join(character for character in normalized.lower() if character.isalpha())
        matches += word in common_words
    return matches * 20


def ocr_layer_score(layer: OcrLayer, language: str = "") -> int:
    words = layer[2]
    useful_characters = sum(
        character.isalnum() for word in words for character in word[4]
    )
    substantial_words = sum(len(word[4].strip()) >= 3 for word in words)
    return (
        useful_characters
        + substantial_words * 3
        + language_word_bonus([word[4] for word in words], language)
    )


def select_orientation(
    scores: dict[Orientation, int], original_score: int | None = None
) -> Orientation:
    if original_score is None:
        original_score = scores.get(NORMAL_ORIENTATION, 0)
    best_orientation = max(ORIENTATIONS, key=lambda item: scores.get(item, 0))
    best_score = scores.get(best_orientation, 0)
    required_gain = max(8, round(original_score * 0.35))
    if best_score < 12 or best_score < original_score + required_gain:
        return NORMAL_ORIENTATION
    return best_orientation


def pixmap_to_image(pixmap: fitz.Pixmap):
    cv2 = importlib.import_module("cv2")
    numpy = importlib.import_module("numpy")
    image_data = numpy.frombuffer(pixmap.tobytes("png"), dtype=numpy.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Não foi possível analisar a orientação da página.")
    return image, cv2


def image_to_pixmap(image, dpi: int, cv2) -> fitz.Pixmap:
    encoded, png_data = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError("Não foi possível transformar a orientação da página.")
    result = fitz.Pixmap(png_data.tobytes())
    result.set_dpi(dpi, dpi)
    return result


def detect_orientation(
    pixmap: fitz.Pixmap,
    dpi: int,
    score_image: Callable[[fitz.Pixmap], int],
) -> DetectedOrientation:
    image, cv2 = pixmap_to_image(pixmap)
    longest_side = max(image.shape[:2])
    if longest_side > 1600:
        scale = 1600 / longest_side
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    original_score = score_image(image_to_pixmap(image, dpi, cv2))
    corrections: dict[Orientation, float] = {}
    scores: dict[Orientation, int] = {}
    for orientation in ORIENTATIONS:
        transformed = transform_image(image, orientation, cv2)
        candidate_axis = estimate_text_axis_angle(transformed, cv2)
        correction = estimate_skew_correction(transformed, cv2)
        if candidate_axis is not None and abs(candidate_axis) > 46:
            corrections[orientation] = correction
            scores[orientation] = 0
            continue

        correction_options = [correction]
        if candidate_axis is not None and abs(abs(candidate_axis) - 45) <= 1:
            correction_options.append(candidate_axis)

        best_correction = correction
        best_candidate_score = 0
        for correction_option in correction_options:
            candidate = rotate_image(transformed, correction_option, cv2)
            corrected_axis = estimate_text_axis_angle(candidate, cv2)
            if corrected_axis is not None and abs(corrected_axis) > 15:
                candidate_score = 0
            else:
                candidate_score = score_image(
                    image_to_pixmap(candidate, dpi, cv2)
                )
            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_correction = correction_option
        corrections[orientation] = best_correction
        scores[orientation] = best_candidate_score

    best_score = max(scores.values())
    normal_score = scores[NORMAL_ORIENTATION]
    text_axis = estimate_text_axis_angle(image, cv2)
    near_horizontal = text_axis is not None and abs(text_axis) <= 30
    close_to_best = best_score - normal_score <= max(8, round(best_score * 0.08))
    preserves_reading = normal_score >= max(12, round(original_score * 0.9))
    if near_horizontal and close_to_best and preserves_reading:
        return NORMAL_ORIENTATION, corrections[NORMAL_ORIENTATION]

    best_orientation = max(ORIENTATIONS, key=lambda item: scores[item])
    if not near_horizontal and scores[best_orientation] >= max(
        12, round(original_score * 0.9)
    ):
        return best_orientation, corrections[best_orientation]

    selected = select_orientation(scores, original_score)
    selected_score = scores[selected]
    required_gain = max(8, round(original_score * 0.35))
    if selected_score < 12 or selected_score < original_score + required_gain:
        return NORMAL_DETECTED_ORIENTATION
    return selected, corrections[selected]


def apply_orientation(
    pixmap: fitz.Pixmap, orientation: DetectedOrientation, dpi: int
) -> fitz.Pixmap:
    if orientation == NORMAL_DETECTED_ORIENTATION:
        return pixmap
    image, cv2 = pixmap_to_image(pixmap)
    coarse_orientation, skew_correction = orientation
    transformed = transform_image(image, coarse_orientation, cv2)
    corrected = rotate_image(transformed, skew_correction, cv2)
    return image_to_pixmap(corrected, dpi, cv2)


def rapid_orientation_score(reader, candidate: fitz.Pixmap, language: str) -> int:
    recognized = reader.readtext(candidate.tobytes("png"), detail=1)
    confidence_score = sum(
        int(sum(character.isalnum() for character in text) * confidence * 10)
        for _, text, confidence in recognized
    )
    return confidence_score + language_word_bonus(
        [text for _, text, _ in recognized], language
    )


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


def preprocess_for_ocr(pixmap: fitz.Pixmap, dpi: int) -> fitz.Pixmap:
    try:
        cv2 = importlib.import_module("cv2")
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError(
            "O OCR aprimorado requer OpenCV. Instale novamente requirements.txt."
        ) from error

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    image_data = numpy.frombuffer(pixmap.tobytes("png"), dtype=numpy.uint8)
    try:
        grayscale = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise RuntimeError(
                "Não foi possível preparar a página para o OCR aprimorado."
            )
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            grayscale
        )
        denoised = cv2.medianBlur(contrast, 3)
        processed_rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        encoded, png_data = cv2.imencode(".png", processed_rgb)
        if not encoded:
            raise RuntimeError("Não foi possível concluir o pré-processamento da página.")
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            "O OpenCV não conseguiu processar a página. Reduza o DPI ou desative "
            "o OCR aprimorado."
        ) from error

    result = fitz.Pixmap(png_data.tobytes())
    result.set_dpi(dpi, dpi)
    return result


def extract_ocr_layer(page_data: bytes) -> OcrLayer:
    with fitz.open("pdf", page_data) as document:
        page = document[0]
        words = [
            (float(word[0]), float(word[1]), float(word[2]), float(word[3]), word[4])
            for word in page.get_text("words")
            if word[4].strip()
        ]
        return page.rect.width, page.rect.height, words


def insert_ocr_layer(page: fitz.Page, layer: OcrLayer) -> None:
    layer_width, layer_height, words = layer
    scale_x = page.rect.width / layer_width
    scale_y = page.rect.height / layer_height
    for x0, y0, x1, y1, text in words:
        left = x0 * scale_x
        right = x1 * scale_x
        top = y0 * scale_y
        bottom = y1 * scale_y
        height = max(bottom - top, 1)
        width = max(right - left, 1)
        natural_width = max(
            fitz.get_text_length(text, fontname="helv", fontsize=1), 0.01
        )
        font_size = max(1, min(height * 0.8, width / natural_width))
        page.insert_text(
            (left, min(bottom, page.rect.height)),
            text,
            fontname="helv",
            fontsize=font_size,
            render_mode=3,
        )


def _process_page(
    source: Path,
    index: int,
    language: str,
    dpi: int,
    force_ocr: bool,
    tessdata: Path,
    enhance_accuracy: bool,
    auto_orient: bool,
) -> PageResult:
    with fitz.open(source) as document:
        page = document[index]
        has_images = bool(page.get_images(full=True))
        has_searchable_text = len(page.get_text("text").strip()) >= 20
        if not force_ocr and has_searchable_text and not has_images:
            return index, None, None

        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        ocr_pixmap = preprocess_for_ocr(pixmap, dpi) if enhance_accuracy else pixmap
        orientation = NORMAL_DETECTED_ORIENTATION
        if auto_orient:
            orientation = detect_orientation(
                ocr_pixmap,
                dpi,
                lambda candidate: ocr_layer_score(
                    extract_ocr_layer(
                        candidate.pdfocr_tobytes(
                            language=language, tessdata=str(tessdata)
                        )
                    ),
                    language,
                ),
            )
            ocr_pixmap = apply_orientation(ocr_pixmap, orientation, dpi)
        page_data = ocr_pixmap.pdfocr_tobytes(
            language=language, tessdata=str(tessdata)
        )
        layer = extract_ocr_layer(page_data)
        correction = None
        if orientation != NORMAL_DETECTED_ORIENTATION:
            corrected_visual = apply_orientation(pixmap, orientation, dpi)
            correction = corrected_visual.tobytes("png"), layer[0], layer[1]
        return index, layer, correction


def process_page(
    source: Path,
    index: int,
    language: str,
    dpi: int,
    force_ocr: bool,
    tessdata: Path,
    enhance_accuracy: bool,
    auto_orient: bool,
) -> PageResult:
    try:
        return _process_page(
            source,
            index,
            language,
            dpi,
            force_ocr,
            tessdata,
            enhance_accuracy,
            auto_orient,
        )
    except Exception as error:
        raise RuntimeError(f"Falha na página {index + 1}: {error}") from None


def process_page_gpu(
    document: fitz.Document,
    index: int,
    language: str,
    dpi: int,
    force_ocr: bool,
    reader,
    enhance_accuracy: bool,
    auto_orient: bool,
) -> tuple[OcrLayer | None, VisualCorrection | None]:
    source_page = document[index]
    has_images = bool(source_page.get_images(full=True))
    has_searchable_text = len(source_page.get_text("text").strip()) >= 20
    if not force_ocr and has_searchable_text and not has_images:
        return None, None

    pixmap = source_page.get_pixmap(dpi=dpi, alpha=False)
    ocr_pixmap = preprocess_for_ocr(pixmap, dpi) if enhance_accuracy else pixmap
    orientation = NORMAL_DETECTED_ORIENTATION
    if auto_orient:
        orientation = detect_orientation(
            ocr_pixmap,
            dpi,
            lambda candidate: rapid_orientation_score(reader, candidate, language),
        )
        ocr_pixmap = apply_orientation(ocr_pixmap, orientation, dpi)
    recognized = reader.readtext(ocr_pixmap.tobytes("png"), detail=1)

    page_width = source_page.rect.width
    page_height = source_page.rect.height
    if orientation != NORMAL_DETECTED_ORIENTATION:
        page_width = ocr_pixmap.width * 72 / dpi
        page_height = ocr_pixmap.height * 72 / dpi
    scale_x = page_width / ocr_pixmap.width
    scale_y = page_height / ocr_pixmap.height
    words: list[OcrWord] = []

    for box, text, confidence in recognized:
        if not text.strip() or confidence < 0.15:
            continue
        left = float(min(point[0] for point in box) * scale_x)
        right = float(max(point[0] for point in box) * scale_x)
        top = float(min(point[1] for point in box) * scale_y)
        bottom = float(max(point[1] for point in box) * scale_y)
        words.append((left, top, right, bottom, text))

    correction = None
    if orientation != NORMAL_DETECTED_ORIENTATION:
        corrected_visual = apply_orientation(pixmap, orientation, dpi)
        correction = corrected_visual.tobytes("png"), page_width, page_height
    return (page_width, page_height, words), correction


def append_processed_page(
    output: fitz.Document,
    source: fitz.Document,
    index: int,
    layer: OcrLayer | None,
    correction: VisualCorrection | None,
) -> None:
    if correction is None:
        output.insert_pdf(source, from_page=index, to_page=index, links=False)
    else:
        image, width, height = correction
        page = output.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=image)
    if layer is not None:
        insert_ocr_layer(output[-1], layer)


def copy_document_navigation(source: fitz.Document, destination: fitz.Document) -> None:
    toc = source.get_toc(simple=False)
    for item in toc:
        toc_destination = item[3]
        if (
            toc_destination.get("kind") == fitz.LINK_NAMED
            and toc_destination.get("page", -1) >= 0
        ):
            toc_destination["kind"] = fitz.LINK_GOTO
            toc_destination.pop("nameddest", None)
    if toc:
        destination.set_toc(toc)

    for page_index in range(source.page_count):
        destination_page = destination[page_index]
        for source_link in source[page_index].get_links():
            link = {
                key: value
                for key, value in source_link.items()
                if key not in {"xref", "id"} and value is not None
            }
            if link["kind"] == fitz.LINK_NAMED and link.get("page", -1) >= 0:
                link["kind"] = fitz.LINK_GOTO
                link.pop("nameddest", None)
            destination_page.insert_link(link)


def ocr_pdf(
    source: Path,
    destination: Path,
    language: str = DEFAULT_LANGUAGE,
    dpi: int = 300,
    force_ocr: bool = False,
    progress: Callable[[int, int], None] | None = None,
    workers: int = DEFAULT_WORKERS,
    use_gpu: bool = False,
    enhance_accuracy: bool = False,
    cancelled: Callable[[], bool] | None = None,
    auto_orient: bool = True,
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
    pages_temporary = destination.with_suffix(".ocr.pages.tmp.pdf")
    temporary = destination.with_suffix(".ocr.tmp.pdf")
    output = fitz.open()

    try:
        if cancelled and cancelled():
            raise OcrCancelled()
        with fitz.open(source) as document:
            total = document.page_count
            if use_gpu:
                for index in range(total):
                    if cancelled and cancelled():
                        raise OcrCancelled()
                    page_data, correction = process_page_gpu(
                        document,
                        index,
                        language,
                        dpi,
                        force_ocr,
                        gpu_reader,
                        enhance_accuracy,
                        auto_orient,
                    )
                    append_processed_page(
                        output, document, index, page_data, correction
                    )
                    if progress:
                        progress(index + 1, total)
            else:
                workers = max(1, min(workers, total, MAX_WORKERS))
                if enhance_accuracy:
                    workers = min(workers, MAX_ENHANCED_WORKERS)
                window_size = workers
                pending_pages: dict[
                    int, tuple[OcrLayer | None, VisualCorrection | None]
                ] = {}
                next_to_submit = 0
                next_to_write = 0

                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    while next_to_write < total:
                        if cancelled and cancelled():
                            raise OcrCancelled()
                        while next_to_submit < total and next_to_submit < next_to_write + window_size:
                            future = executor.submit(
                                process_page,
                                source,
                                next_to_submit,
                                language,
                                dpi,
                                force_ocr,
                                tessdata,
                                enhance_accuracy,
                                auto_orient,
                            )
                            futures[future] = next_to_submit
                            next_to_submit += 1

                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        if cancelled and cancelled():
                            raise OcrCancelled()
                        for future in completed:
                            index, page_data, correction = future.result()
                            pending_pages[index] = page_data, correction
                            del futures[future]

                        while next_to_write in pending_pages:
                            page_data, correction = pending_pages.pop(next_to_write)
                            append_processed_page(
                                output,
                                document,
                                next_to_write,
                                page_data,
                                correction,
                            )
                            next_to_write += 1
                            if progress:
                                progress(next_to_write, total)

        if cancelled and cancelled():
            raise OcrCancelled()
        output.save(pages_temporary, garbage=4, deflate=True)
        output.close()

        if cancelled and cancelled():
            raise OcrCancelled()
        with fitz.open(source) as source_document, fitz.open(
            pages_temporary
        ) as navigable_output:
            copy_document_navigation(source_document, navigable_output)
            navigable_output.save(temporary, garbage=1, deflate=True)

        pages_temporary.unlink(missing_ok=True)
        temporary.replace(destination)
    except Exception:
        output.close()
        pages_temporary.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        raise


class OcrApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("900x650")
        self.root.minsize(720, 560)
        self.files: list[Path] = []
        self.generated_files: list[Path] = []
        self.last_output_dir: Path | None = None
        self.cancel_event = threading.Event()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.output_dir = StringVar(value=str(Path.home() / "Documents" / "PDFs com OCR"))
        self.language = StringVar(value=DEFAULT_LANGUAGE)
        self.dpi = StringVar(value="300")
        self.workers = StringVar(value=str(DEFAULT_WORKERS))
        self.force_ocr = BooleanVar(value=False)
        self.enhance_accuracy = BooleanVar(value=False)
        self.auto_orient = BooleanVar(value=True)
        self.status = StringVar(value="Adicione um ou mais arquivos PDF.")
        self.file_count = StringVar(value="Nenhum arquivo selecionado")
        self._build_ui()
        self.root.after(100, self._consume_events)

    def _configure_styles(self) -> None:
        colors = {
            "background": "#F4F6F8",
            "surface": "#FFFFFF",
            "ink": "#172B3A",
            "muted": "#607384",
            "accent": "#087E8B",
            "accent_active": "#066A75",
            "border": "#D9E1E7",
            "header": "#102A43",
        }
        self.colors = colors
        self.root.configure(background=colors["background"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=colors["background"])
        style.configure("Surface.TFrame", background=colors["surface"])
        style.configure(
            "TLabel",
            background=colors["background"],
            foreground=colors["ink"],
            font=("Segoe UI", 10),
        )
        style.configure("Header.TFrame", background=colors["header"])
        style.configure(
            "Title.TLabel",
            background=colors["header"],
            foreground="#FFFFFF",
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "Subtitle.TLabel",
            background=colors["header"],
            foreground="#C9D8E6",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=colors["surface"],
            foreground=colors["ink"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Muted.TLabel",
            background=colors["surface"],
            foreground=colors["muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "Card.TLabel",
            background=colors["surface"],
            foreground=colors["ink"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "TButton",
            padding=(12, 7),
            background="#E8EDF1",
            foreground=colors["ink"],
            bordercolor=colors["border"],
            font=("Segoe UI Semibold", 9),
        )
        style.map("TButton", background=[("active", "#DDE5EA")])
        style.configure(
            "Accent.TButton",
            padding=(18, 9),
            background=colors["accent"],
            foreground="#FFFFFF",
            bordercolor=colors["accent"],
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", colors["accent_active"]), ("disabled", "#9CB8BC")],
            foreground=[("disabled", "#EDF3F4")],
        )
        style.configure(
            "Cancel.TButton",
            padding=(12, 9),
            background="#C44536",
            foreground="#FFFFFF",
            bordercolor="#C44536",
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Cancel.TButton",
            background=[("active", "#A9362A"), ("disabled", "#D8B2AD")],
            foreground=[("disabled", "#F8EEEE")],
        )
        style.configure(
            "Card.TLabelframe",
            background=colors["surface"],
            bordercolor=colors["border"],
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=colors["surface"],
            foreground=colors["ink"],
            font=("Segoe UI Semibold", 11),
        )
        style.configure("TEntry", padding=7, fieldbackground="#FFFFFF", bordercolor=colors["border"])
        style.configure("TCombobox", padding=6, fieldbackground="#FFFFFF", bordercolor=colors["border"])
        style.configure("TCheckbutton", background=colors["surface"], foreground=colors["ink"])
        style.configure(
            "OCR.Horizontal.TProgressbar",
            troughcolor="#DCE5EA",
            background=colors["accent"],
            bordercolor="#DCE5EA",
            thickness=8,
        )

    def _build_ui(self) -> None:
        self._configure_styles()

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(28, 20))
        header.pack(fill=X)
        ttk.Label(header, text="PDF OCR", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Transforme documentos digitalizados em PDFs pesquisáveis sem perder a navegação.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        frame = ttk.Frame(self.root, padding=(28, 22, 28, 20))
        frame.pack(fill=BOTH, expand=True)

        files_card = ttk.Frame(frame, style="Surface.TFrame", padding=16)
        files_card.pack(fill=BOTH, expand=True)
        files_header = ttk.Frame(files_card, style="Surface.TFrame")
        files_header.pack(fill=X, pady=(0, 10))
        ttk.Label(files_header, text="Documentos", style="Section.TLabel").pack(side=LEFT)
        ttk.Label(files_header, textvariable=self.file_count, style="Muted.TLabel").pack(side=RIGHT)

        list_frame = ttk.Frame(files_card, style="Surface.TFrame")
        list_frame.pack(fill=BOTH, expand=True)
        self.file_list = __import__("tkinter").Listbox(
            list_frame,
            selectmode="extended",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
            background="#FAFBFC",
            foreground=self.colors["ink"],
            selectbackground=self.colors["accent"],
            selectforeground="#FFFFFF",
            font=("Segoe UI", 10),
            activestyle="none",
        )
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, command=self.file_list.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        self.file_list.configure(yscrollcommand=scrollbar.set)

        file_buttons = ttk.Frame(files_card, style="Surface.TFrame")
        file_buttons.pack(fill=X, pady=(12, 0))
        ttk.Button(file_buttons, text="+ Adicionar PDFs", command=self._add_files).pack(side=LEFT)
        ttk.Button(file_buttons, text="Remover selecionados", command=self._remove_selected).pack(side=LEFT, padx=8)
        ttk.Button(file_buttons, text="Limpar", command=self._clear_files).pack(side=LEFT)

        options = ttk.LabelFrame(frame, text="Configurações", style="Card.TLabelframe", padding=16)
        options.pack(fill=X, pady=(16, 0))
        ttk.Label(options, text="Pasta de saída", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.output_dir).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 13))
        ttk.Button(options, text="Escolher pasta", command=self._choose_output).grid(row=1, column=3, padx=(8, 0), pady=(5, 13))
        ttk.Label(options, text="Idioma", style="Card.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Combobox(options, textvariable=self.language, values=("por", "eng", "spa", "por+eng"), width=14).grid(row=3, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(options, text="Qualidade (DPI)", style="Card.TLabel").grid(row=2, column=1, sticky="w", padx=(12, 0))
        ttk.Combobox(options, textvariable=self.dpi, values=("200", "300", "400"), state="readonly").grid(row=3, column=1, sticky="ew", padx=(12, 0), pady=(5, 0))
        ttk.Label(
            options,
            text=f"CPUs / GPU ({AVAILABLE_CPUS} CPUs disponíveis)",
            style="Card.TLabel",
        ).grid(row=2, column=2, sticky="w", padx=(12, 0))
        ttk.Combobox(options, textvariable=self.workers, values=PROCESSING_OPTIONS, state="readonly").grid(row=3, column=2, columnspan=2, sticky="ew", padx=(12, 0), pady=(5, 0))
        ttk.Checkbutton(options, text="Refazer OCR em páginas que já possuem texto", variable=self.force_ocr).grid(row=4, column=0, columnspan=4, sticky="w", pady=(13, 0))
        ttk.Checkbutton(
            options,
            text="OCR aprimorado com OpenCV (contraste e redução de ruído)",
            variable=self.enhance_accuracy,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            options,
            text="Corrigir inclinação (0° a 360°), rotação e espelhamento",
            variable=self.auto_orient,
        ).grid(row=6, column=0, columnspan=4, sticky="w", pady=(7, 0))
        options.columnconfigure(0, weight=1)
        options.columnconfigure(1, weight=1)
        options.columnconfigure(2, weight=1)

        footer = ttk.Frame(frame)
        footer.pack(fill=X, pady=(18, 0))
        progress_area = ttk.Frame(footer)
        progress_area.pack(side=LEFT, fill=X, expand=True, padx=(0, 18))
        self.progress = ttk.Progressbar(
            progress_area, mode="determinate", style="OCR.Horizontal.TProgressbar"
        )
        self.progress.pack(fill=X, pady=(0, 7))
        ttk.Label(progress_area, textvariable=self.status).pack(anchor="w")
        actions = ttk.Frame(footer)
        actions.pack(side=RIGHT, anchor="s")
        self.open_folder_button = ttk.Button(
            actions,
            text="Abrir pasta",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_folder_button.pack(side=LEFT)
        self.open_pdf_button = ttk.Button(
            actions,
            text="Abrir último PDF",
            command=self._open_generated_pdf,
            state="disabled",
        )
        self.open_pdf_button.pack(side=LEFT, padx=8)
        self.cancel_button = ttk.Button(
            actions,
            text="Cancelar",
            command=self._cancel,
            style="Cancel.TButton",
            state="disabled",
        )
        self.cancel_button.pack(side=LEFT, padx=(0, 8))
        self.start_button = ttk.Button(
            actions, text="Executar OCR", command=self._start, style="Accent.TButton"
        )
        self.start_button.pack(side=LEFT)

    def _update_file_count(self) -> None:
        count = len(self.files)
        if count == 0:
            self.file_count.set("Nenhum arquivo selecionado")
        elif count == 1:
            self.file_count.set("1 arquivo selecionado")
        else:
            self.file_count.set(f"{count} arquivos selecionados")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Selecionar PDFs", filetypes=(("Arquivos PDF", "*.pdf"),))
        for raw_path in paths:
            path = Path(raw_path)
            if path not in self.files:
                self.files.append(path)
                self.file_list.insert(END, str(path))
        self._update_file_count()

    def _remove_selected(self) -> None:
        for index in reversed(self.file_list.curselection()):
            self.file_list.delete(index)
            del self.files[index]
        self._update_file_count()

    def _clear_files(self) -> None:
        self.files.clear()
        self.file_list.delete(0, END)
        self._update_file_count()

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Escolher pasta de saída")
        if path:
            self.output_dir.set(path)

    def _open_output_folder(self) -> None:
        if self.last_output_dir is None or not self.last_output_dir.is_dir():
            messagebox.showwarning(APP_NAME, "A pasta de saída ainda não está disponível.")
            return
        try:
            subprocess.Popen(
                ["explorer.exe", str(self.last_output_dir.resolve())]
            )
        except OSError as error:
            messagebox.showerror(APP_NAME, f"Não foi possível abrir a pasta:\n{error}")

    def _open_generated_pdf(self) -> None:
        if not self.generated_files or not self.generated_files[-1].is_file():
            messagebox.showwarning(APP_NAME, "O PDF gerado ainda não está disponível.")
            return
        try:
            os.startfile(self.generated_files[-1])
        except OSError as error:
            messagebox.showerror(APP_NAME, f"Não foi possível abrir o PDF:\n{error}")

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Cancelando após concluir as páginas em andamento...")

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
        self.cancel_event.clear()
        self.cancel_button.configure(state="normal")
        self.open_folder_button.configure(state="disabled")
        self.open_pdf_button.configure(state="disabled")
        self.generated_files.clear()
        self.last_output_dir = None
        self.progress.configure(value=0)
        if self.enhance_accuracy.get() and workers > MAX_ENHANCED_WORKERS:
            self.status.set(
                f"OCR aprimorado: usando até {MAX_ENHANCED_WORKERS} processos para controlar a memória."
            )
        else:
            self.status.set("Preparando o processamento...")
        settings = (
            tuple(self.files),
            Path(self.output_dir.get()),
            self.language.get(),
            self.force_ocr.get(),
            self.enhance_accuracy.get(),
            self.auto_orient.get(),
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
        enhance_accuracy: bool,
        auto_orient: bool,
    ) -> None:
        generated_files: list[Path] = []
        try:
            for file_index, source in enumerate(files):
                destination = output_dir / f"{source.stem}_OCR.pdf"

                def report(page: int, total: int) -> None:
                    overall = ((file_index + page / total) / len(files)) * 100
                    self.events.put(("progress", (overall, source.name, page, total)))

                ocr_pdf(
                    source,
                    destination,
                    language,
                    dpi,
                    force_ocr,
                    report,
                    workers,
                    use_gpu,
                    enhance_accuracy,
                    self.cancel_event.is_set,
                    auto_orient,
                )
                generated_files.append(destination)
            self.events.put(("done", (output_dir, tuple(generated_files))))
        except OcrCancelled:
            self.events.put(("cancelled", (output_dir, tuple(generated_files))))
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
                    output_dir, generated_files = value
                    self.last_output_dir = output_dir
                    self.generated_files = list(generated_files)
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.open_folder_button.configure(state="normal")
                    self.open_pdf_button.configure(state="normal")
                    self.status.set("OCR concluído.")
                    messagebox.showinfo(APP_NAME, f"Arquivos salvos em:\n{output_dir}")
                elif event == "cancelled":
                    output_dir, generated_files = value
                    self.last_output_dir = output_dir if generated_files else None
                    self.generated_files = list(generated_files)
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if generated_files:
                        self.open_folder_button.configure(state="normal")
                        self.open_pdf_button.configure(state="normal")
                    self.status.set("Processamento cancelado.")
                elif event == "error":
                    self.start_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
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
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Melhora contraste e reduz ruído com OpenCV antes do OCR",
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_false",
        dest="auto_orient",
        help="Desativa a correção automática de rotação e espelhamento",
    )
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
                enhance_accuracy=args.enhance,
                auto_orient=args.auto_orient,
            )
            print(f"Salvo em {destination}")
        return

    root = Tk()
    OcrApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()