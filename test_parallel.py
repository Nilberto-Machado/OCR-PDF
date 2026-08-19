from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

import pymupdf as fitz

from app import ocr_pdf


def build_hybrid_pdf(path: Path, pages: int = 20) -> None:
    document = fitz.open()
    for index in range(pages):
        image_document = fitz.open()
        image_page = image_document.new_page(width=420, height=100)
        image_page.insert_text((25, 60), f"IMAGEM PAGINA {index + 1}", fontsize=24)
        image = image_page.get_pixmap(dpi=150, alpha=False).tobytes("png")
        image_document.close()

        page = document.new_page(width=500, height=300)
        page.insert_text((25, 35), f"TEXTO NATIVO PESQUISAVEL PAGINA {index + 1}", fontsize=12)
        page.insert_image(fitz.Rect(25, 80, 475, 190), stream=image)

    document.save(path)
    document.close()


def main() -> None:
    with TemporaryDirectory() as directory:
        source = Path(directory) / "entrada.pdf"
        destination = Path(directory) / "saida.pdf"
        build_hybrid_pdf(source)
        errors: list[BaseException] = []

        def run_ocr() -> None:
            try:
                ocr_pdf(source, destination, "por", 200, False, workers=20)
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=run_ocr)
        thread.start()
        thread.join()
        assert not errors, errors

        with fitz.open(destination) as result:
            texts = [page.get_text() for page in result]

        assert len(texts) == 20
        for index, text in enumerate(texts):
            assert f"PAGINA {index + 1}" in text
            assert "IMAGEM" in text

    print("20 paginas em ordem, OCR paralelo validado")


if __name__ == "__main__":
    main()