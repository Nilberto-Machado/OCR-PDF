# PDF OCR

Aplicativo local para Windows que transforma PDFs digitalizados ou híbridos em PDFs pesquisáveis, sem alterar os arquivos originais. Oferece interface gráfica e linha de comando, OCR em português e inglês, processamento paralelo por CPU e aceleração opcional em GPU AMD via DirectML.

## Recursos

- Processa vários PDFs e salva novas cópias com o sufixo `_OCR`.
- Mantém páginas digitais intactas quando elas já contêm texto pesquisável e não possuem imagens.
- Aplica OCR automaticamente em páginas digitalizadas ou híbridas.
- Permite escolher idioma, resolução entre 200 e 400 DPI e até 20 processos em paralelo.
- Inclui modelos do Tesseract para português e inglês.
- Oferece aceleração opcional em GPU AMD com RapidOCR e DirectML.

## Preparação no Windows

1. Instale o Tesseract OCR:

   ```powershell
   winget install --id UB-Mannheim.TesseractOCR
   ```

2. Os modelos de português e inglês já estão incluídos na pasta `tessdata` deste projeto.

3. Instale a dependência Python, caso necessário:

   ```powershell
   python -m pip install -r requirements.txt
   ```

## Uso

Dê dois cliques em `iniciar.bat`, adicione os PDFs, escolha a pasta de saída e clique em **Executar OCR**.

Também é possível usar pelo terminal:

```powershell
python app.py documento.pdf -o saida_ocr -l por --dpi 300 --workers 20
```

Por padrão, páginas totalmente digitais são copiadas sem novo OCR. Páginas digitalizadas ou híbridas, com texto e imagens, recebem OCR automaticamente. Use a opção de forçar OCR para reprocessar também páginas digitais sem imagens.

O programa usa doze processos de OCR em paralelo por padrão. Na opção **Processamento**, é possível escolher de 1 a 20 processos. Reduza a quantidade caso precise limitar o uso de memória.

## GPU opcional

O modo GPU usa DirectML no Windows e é compatível com a AMD Radeon RX 580. Instale as dependências opcionais no ambiente do projeto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Mantenha o driver AMD Adrenalin atualizado. O primeiro uso pode baixar o modelo latino PP-OCRv5, utilizado para português, inglês e espanhol.

Na interface, selecione **GPU AMD (DirectML)** em **Processamento**. Pelo terminal, use:

```powershell
.\.venv\Scripts\python.exe app.py documento.pdf -o saida_ocr --gpu
```

O backend GPU usa RapidOCR em uma única fila para evitar disputar a memória de vídeo. Caso o DirectML ou as dependências não estejam disponíveis, o programa mostra uma mensagem explicando o que falta.