import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

import pymupdf as fitz

from app import ocr_pdf


def build_hybrid_pdf(path: Path, pages: int = 4) -> None:
    document = fitz.open()
    for index in range(pages):
        image_document = fitz.open()
        image_page = image_document.new_page(width=420, height=100)
        image_page.insert_text((25, 60), f"IMAGEM PAGINA {index + 1}", fontsize=24)
        image = image_page.get_pixmap(dpi=150, alpha=False).tobytes("png")
        image_document.close()

        page = document.new_page(width=595, height=842)
        page.insert_text((25, 35), f"TEXTO NATIVO PESQUISAVEL PAGINA {index + 1}", fontsize=12)
        page.insert_image(fitz.Rect(25, 80, 570, 300), stream=image)

    document.set_toc([[1, "Inicio", 1], [1, "Ultima pagina", pages]])
    document[0].insert_link(
        {
            "kind": fitz.LINK_GOTO,
            "from": fitz.Rect(25, 210, 225, 235),
            "page": pages - 1,
            "to": fitz.Point(0, 0),
        }
    )
    document.save(path)
    document.close()


def main() -> None:
    with TemporaryDirectory() as directory:
        stress_test = os.environ.get("OCR_STRESS_TEST") == "1"
        page_count = 20 if stress_test else 4
        dpi = 300 if stress_test else 200
        workers = 20 if stress_test else 4
        source = Path(directory) / "entrada.pdf"
        destination = Path(directory) / "saida.pdf"
        build_hybrid_pdf(source, page_count)
        errors: list[BaseException] = []
        cancel_event = Event()

        def run_ocr() -> None:
            try:
                ocr_pdf(
                    source,
                    destination,
                    "por",
                    dpi,
                    False,
                    workers=workers,
                    enhance_accuracy=True,
                    cancelled=cancel_event.is_set,
                    auto_orient=False,
                )
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=run_ocr)
        thread.start()
        thread.join()
        assert not errors, errors

        with fitz.open(destination) as result:
            texts = [page.get_text() for page in result]
            toc = result.get_toc()
            links = result[0].get_links()
            target_xref = result.page_xref(page_count - 1)
            link_object = result.xref_object(links[0]["xref"], compressed=False)

        assert len(texts) == page_count
        for index, text in enumerate(texts):
            assert f"PAGINA {index + 1}" in text
            assert "IMAGEM" in text
        assert toc == [[1, "Inicio", 1], [1, "Ultima pagina", page_count]]
        assert links and links[0]["page"] == page_count - 1
        assert f"{target_xref} 0 R" in link_object
        assert destination.stat().st_size < source.stat().st_size * 3

        size_ratio = destination.stat().st_size / source.stat().st_size

    print(
        f"{page_count} paginas A4 a {dpi} DPI, OCR aprimorado, "
        "indice e links preservados; "
        f"tamanho final {size_ratio:.2f}x o original"
    )


if __name__ == "__main__":
    main()