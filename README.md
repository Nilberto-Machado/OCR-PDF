# PDF OCR

Aplicativo local para Windows que transforma PDFs digitalizados ou híbridos em PDFs pesquisáveis, sem alterar os arquivos originais. Oferece interface gráfica e linha de comando, OCR em português e inglês, processamento paralelo por CPU e aceleração opcional em GPU AMD via DirectML.

## Recursos

- Processa vários PDFs e salva novas cópias com o sufixo `_OCR`.
- Permite cancelar o processamento sem substituir o PDF por um arquivo parcial.
- Permite abrir a pasta de saída e o último PDF gerado diretamente pela interface.
- Mantém páginas digitais intactas quando elas já contêm texto pesquisável e não possuem imagens.
- Preserva a página original comprimida e adiciona somente a camada OCR invisível, evitando duplicar a imagem em alta resolução.
- Aplica OCR automaticamente em páginas digitalizadas ou híbridas.
- Detecta e corrige textos inclinados em qualquer posição de 0° a 360°, além de páginas espelhadas horizontal/verticalmente.
- Permite escolher idioma, resolução entre 200 e 400 DPI e usar até o total de CPUs disponíveis.
- Oferece OCR aprimorado com OpenCV para aumentar contraste e reduzir ruído antes do reconhecimento.
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

Dê dois cliques em `iniciar.bat`, adicione os PDFs, escolha a pasta de saída e clique em **Executar OCR**. Ao concluir, use **Abrir pasta** ou **Abrir último PDF** para acessar o resultado.

Durante a execução, use **Cancelar** para interromper o lote. As páginas que já estão em processamento terminam antes da limpeza segura dos arquivos temporários.

Também é possível usar pelo terminal:

```powershell
python app.py documento.pdf -o saida_ocr -l por --dpi 300 --workers 20 --enhance
```

Por padrão, páginas totalmente digitais são copiadas sem novo OCR. Páginas digitalizadas ou híbridas, com texto e imagens, recebem OCR automaticamente. Use a opção de forçar OCR para reprocessar também páginas digitais sem imagens.

A opção **Corrigir inclinação (0° a 360°), rotação e espelhamento** compara as orientações possíveis e mede a inclinação residual do texto antes do OCR. Ela corrige qualquer posição ao redor do círculo, incluindo ângulos intermediários, além de espelhamento horizontal ou vertical, e vem ativada por padrão. Quando encontra uma alternativa claramente mais legível, o programa corrige a aparência da página e posiciona a camada pesquisável na orientação restaurada. Na linha de comando, use `--no-auto-orient` para desativar essa análise.

O OCR mantém a aparência original e adiciona texto invisível para pesquisa e seleção. PDFs produzidos por versões anteriores, que rasterizavam novamente cada página e mantinham destinos nomeados inválidos, devem ser gerados novamente para receber a redução de tamanho e a correção dos links internos.

O programa usa até doze processos de OCR em paralelo por padrão. Na opção **CPUs / GPU**, é possível escolher de 1 até o total de CPUs lógicas disponíveis na máquina. Reduza a quantidade caso precise limitar o uso de memória.

Ative **OCR aprimorado com OpenCV** em documentos com baixo contraste, fundo irregular ou ruído de digitalização. Esse modo aplica contraste local e redução de ruído somente à imagem enviada ao mecanismo OCR; a aparência original da página é mantida no PDF final. O processamento adicional aumenta o tempo de execução e usa no máximo vinte processos simultâneos, mesmo quando um número maior de CPUs estiver selecionado.

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