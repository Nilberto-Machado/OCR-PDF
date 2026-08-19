import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy
import pymupdf as fitz

from app import (
    NORMAL_ORIENTATION,
    ORIENTATIONS,
    apply_orientation,
    detect_orientation,
    estimate_skew_correction,
    estimate_text_axis_angle,
    image_to_pixmap,
    ocr_pdf,
    pixmap_to_image,
    rotate_image,
    select_orientation,
    transform_image,
)


class OrientationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.upright = numpy.array(
            [
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
            ],
            dtype=numpy.uint8,
        )

    def test_detects_correction_for_every_rotation_and_mirroring(self) -> None:
        for distortion in ORIENTATIONS:
            with self.subTest(distortion=distortion):
                distorted = transform_image(self.upright, distortion, cv2)
                pixmap = image_to_pixmap(distorted, 300, cv2)

                def score(candidate) -> int:
                    image, _ = pixmap_to_image(candidate)
                    return 100 if numpy.array_equal(image, self.upright) else 0

                correction = detect_orientation(pixmap, 300, score)
                corrected = apply_orientation(pixmap, correction, 300)
                corrected_image, _ = pixmap_to_image(corrected)
                self.assertTrue(numpy.array_equal(corrected_image, self.upright))

    def test_keeps_original_when_improvement_is_not_clear(self) -> None:
        scores = {orientation: 10 for orientation in ORIENTATIONS}
        scores[(True, 90)] = 16
        self.assertEqual(select_orientation(scores), NORMAL_ORIENTATION)

    def test_accepts_clear_improvement(self) -> None:
        scores = {orientation: 10 for orientation in ORIENTATIONS}
        scores[(True, 270)] = 30
        self.assertEqual(select_orientation(scores), (True, 270))

    def test_estimates_intermediate_rotation_angles(self) -> None:
        image = numpy.full((500, 900, 3), 255, dtype=numpy.uint8)
        for index, text in enumerate(
            ("DOCUMENTO PARA TESTE", "TEXTO ROTACIONADO", "LEITURA AUTOMATICA")
        ):
            cv2.putText(
                image,
                text,
                (60, 130 + index * 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        for angle in (-40, -25, -10, 10, 25, 40):
            with self.subTest(angle=angle):
                rotated = rotate_image(image, angle, cv2)
                correction = estimate_skew_correction(rotated, cv2)
                self.assertAlmostEqual(angle + correction, 0, delta=1)

    def test_detects_text_rotation_through_the_full_circle(self) -> None:
        image = numpy.full((210, 380, 3), 255, dtype=numpy.uint8)
        for index, text in enumerate(
            ("DOCUMENTO PARA TESTE", "ROTACAO DE ZERO A 360", "LEITURA AUTOMATICA")
        ):
            cv2.putText(
                image,
                text,
                (40, 60 + index * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
        cv2.circle(image, (18, 18), 9, (0, 0, 255), -1)
        cv2.circle(image, (362, 192), 9, (255, 0, 0), -1)

        def score(candidate) -> int:
            candidate_image, _ = pixmap_to_image(candidate)
            red_y, red_x = numpy.where(
                (candidate_image[:, :, 2] > 180)
                & (candidate_image[:, :, 1] < 100)
                & (candidate_image[:, :, 0] < 100)
            )
            blue_y, blue_x = numpy.where(
                (candidate_image[:, :, 0] > 180)
                & (candidate_image[:, :, 1] < 100)
                & (candidate_image[:, :, 2] < 100)
            )
            if not len(red_x) or not len(blue_x):
                return 0
            height, width = candidate_image.shape[:2]
            upright = (
                red_x.mean() < width / 2
                and red_y.mean() < height / 2
                and blue_x.mean() > width / 2
                and blue_y.mean() > height / 2
            )
            return 100 if upright else 20

        for angle in range(0, 360, 15):
            with self.subTest(angle=angle):
                rotated = rotate_image(image, angle, cv2)
                pixmap = image_to_pixmap(rotated, 200, cv2)
                orientation = detect_orientation(pixmap, 200, score)
                corrected = apply_orientation(pixmap, orientation, 200)
                corrected_image, _ = pixmap_to_image(corrected)
                axis = estimate_text_axis_angle(corrected_image, cv2)
                self.assertIsNotNone(axis)
                self.assertAlmostEqual(axis or 0, 0, delta=1)
                self.assertEqual(score(corrected), 100)

    def test_ocr_restores_a_mirrored_rotated_and_skewed_page(self) -> None:
        upright_document = fitz.open()
        upright_page = upright_document.new_page(width=500, height=300)
        lines = (
            "DOCUMENTO PARA TESTE DE ORIENTACAO",
            "ESTE TEXTO DEVE FICAR LEGIVEL",
            "ROTACAO E ESPELHAMENTO AUTOMATICOS",
            "PAGINA CORRIGIDA COM OCR",
        )
        for index, line in enumerate(lines):
            upright_page.insert_text((30, 60 + index * 45), line, fontsize=20)
        upright_pixmap = upright_page.get_pixmap(dpi=200, alpha=False)
        upright_document.close()

        upright_image, _ = pixmap_to_image(upright_pixmap)
        distorted_image = transform_image(upright_image, (True, 90), cv2)
        distorted_image = rotate_image(distorted_image, 23, cv2)
        distorted_pixmap = image_to_pixmap(distorted_image, 200, cv2)

        with TemporaryDirectory() as directory:
            source = Path(directory) / "rotacionado.pdf"
            destination = Path(directory) / "corrigido.pdf"
            source_document = fitz.open()
            source_page = source_document.new_page(
                width=distorted_pixmap.width * 72 / 200,
                height=distorted_pixmap.height * 72 / 200,
            )
            source_page.insert_image(source_page.rect, pixmap=distorted_pixmap)
            source_document.save(source)
            source_document.close()

            ocr_pdf(source, destination, "por", 200, workers=1)

            with fitz.open(destination) as result:
                text = result[0].get_text().upper()
                self.assertGreater(result[0].rect.width, result[0].rect.height)
                self.assertIn("DOCUMENTO", text)
                self.assertIn("LEGIVEL", text)
                self.assertTrue(result[0].search_for("DOCUMENTO"))

    def test_ocr_restores_an_upside_down_portrait_resume(self) -> None:
        upright_document = fitz.open()
        upright_page = upright_document.new_page(width=595, height=842)
        lines = (
            "RESUMO",
            "Engenheiro de Automacao e Controle",
            "FORMACAO",
            "Centro Universitario FEI",
            "EXPERIENCIA PROFISSIONAL",
            "Analista de Automacao Senior",
            "CURSOS E CERTIFICACOES",
            "IDIOMAS",
        )
        for index, line in enumerate(lines):
            upright_page.insert_text((55, 90 + index * 85), line, fontsize=18)
        upright_pixmap = upright_page.get_pixmap(dpi=200, alpha=False)
        upright_document.close()

        upright_image, _ = pixmap_to_image(upright_pixmap)
        inverted_image = rotate_image(upright_image, 180, cv2)
        inverted_pixmap = image_to_pixmap(inverted_image, 200, cv2)

        with TemporaryDirectory() as directory:
            source = Path(directory) / "curriculo_invertido.pdf"
            destination = Path(directory) / "curriculo_corrigido.pdf"
            source_document = fitz.open()
            source_page = source_document.new_page(width=595, height=842)
            source_page.insert_image(source_page.rect, pixmap=inverted_pixmap)
            source_document.save(source)
            source_document.close()

            ocr_pdf(source, destination, "por", 200, workers=1)

            with fitz.open(destination) as result:
                text = result[0].get_text().upper()
                self.assertLess(result[0].rect.width, result[0].rect.height)
                for heading in ("RESUMO", "FORMACAO", "EXPERIENCIA", "IDIOMAS"):
                    self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
