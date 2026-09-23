import { preprocessCanvas } from './vision'
import { useRef, useState } from 'react'
import { FileText, LockKeyhole, Play, RotateCcw, ScanText, ShieldCheck, Square, Upload, X } from 'lucide-react'
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import './App.css'

pdfjsLib.GlobalWorkerOptions.workerSrc = `${pdfWorker}?v=20260819-2`

type QueueFile = { id: string; file: File; data: ArrayBuffer; pages: number; status: 'ready' | 'processing' | 'done' | 'error'; progress: number; outputUrl?: string; mode?: string }
type OcrWorker = Awaited<ReturnType<typeof import('tesseract.js')['createWorker']>>
const languageOptions = [{ value: 'por', label: 'Portugues' }, { value: 'eng', label: 'English' }]

function transformCandidate(canvas: HTMLCanvasElement, horizontal: boolean, vertical: boolean, inverted: boolean): HTMLCanvasElement {
  const transformed = document.createElement('canvas')
  transformed.width = canvas.width; transformed.height = canvas.height
  const context = transformed.getContext('2d')!
  if (inverted) context.filter = 'invert(1)'
  context.translate(horizontal ? canvas.width : 0, vertical ? canvas.height : 0)
  context.scale(horizontal ? -1 : 1, vertical ? -1 : 1)
  context.drawImage(canvas, 0, 0)
  return transformed
}

function ocrQuality(result: { data: { text: string; confidence?: number } }): number {
  return result.data.text.trim().replace(/[^\p{L}\p{N}]/gu, '').length + (result.data.confidence ?? 0)
}

function exactArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
}

function drawOcrLayer(
  page: ReturnType<PDFDocument['getPage']>,
  words: Array<{ text: string; bbox: { x0: number; y0: number; x1: number; y1: number } }>,
  canvasWidth: number,
  canvasHeight: number,
  font: Awaited<ReturnType<PDFDocument['embedFont']>>,
) {
  const pageSize = page.getSize()
  const scaleX = pageSize.width / canvasWidth
  const scaleY = pageSize.height / canvasHeight
  for (const word of words) {
    const text = word.text.trim()
    if (!text) continue
    const x = word.bbox.x0 * scaleX
    const top = word.bbox.y0 * scaleY
    const height = Math.max((word.bbox.y1 - word.bbox.y0) * scaleY, 1)
    page.drawText(text, {
      x,
      y: pageSize.height - top - height,
      size: Math.max(1, height * 0.82),
      font,
      color: rgb(0, 0, 0),
      opacity: 0,
    })
  }
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null)
  const activeWorkerRef = useRef<OcrWorker | null>(null)
  const cancelRequestedRef = useRef(false)
  const [queue, setQueue] = useState<QueueFile[]>([])
  const [language, setLanguage] = useState('por')
  const [dpi, setDpi] = useState(200)
  const [previewUrl, setPreviewUrl] = useState<string>()
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isAddingFiles, setIsAddingFiles] = useState(false)
  const [enhance, setEnhance] = useState(false)
  const [autoOrient, setAutoOrient] = useState(true)
  const [message, setMessage] = useState('Pronto para receber documentos')
  const completedItems = queue.filter((item) => item.status === 'done' && item.outputUrl)
  const allDone = queue.length > 0 && completedItems.length === queue.length && !isProcessing && !isAddingFiles

  const renderPreview = async (data: ArrayBuffer) => {
    const pdf = await pdfjsLib.getDocument({ data: data.slice(0) }).promise
    const page = await pdf.getPage(1)
    const viewport = page.getViewport({ scale: 0.9 })
    const canvas = document.createElement('canvas')
    canvas.width = viewport.width; canvas.height = viewport.height
    await page.render({ canvas, canvasContext: canvas.getContext('2d')!, viewport }).promise
    setPreviewUrl(canvas.toDataURL('image/jpeg', 0.86))
  }

  const addFiles = async (incoming: FileList | File[]) => {
    const pdfs = Array.from(incoming).filter((file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf'))
    if (!pdfs.length) {
      setMessage('Selecione um arquivo PDF válido')
      return
    }
    setIsAddingFiles(true)
    try {
      const next: QueueFile[] = await Promise.all(pdfs.map(async (file) => ({
        id: `${file.name}-${file.lastModified}-${file.size}`,
        file,
        data: await file.arrayBuffer(),
        pages: 0,
        status: 'ready' as const,
        progress: 0,
      })))
      const unique = next.filter((item) => !queue.some((entry) => entry.id === item.id))
      setQueue((current) => [...current, ...next.filter((item) => !current.some((entry) => entry.id === item.id))])
      if (!unique.length) {
        setMessage('Esse PDF já está na fila')
        return
      }
      setMessage(`${unique.length} documento${unique.length !== 1 ? 's' : ''} lendo...`)

      let parsedCount = 0
      for (const item of unique) {
        let parsedData = item.data
        let pageCount = 0
      for (const file of pdfs) {
          if (file !== item.file) continue
        try {
            let pdf
            try {
              pdf = await pdfjsLib.getDocument({ data: parsedData.slice(0) }).promise
            } catch (pdfError) {
              console.warn(`Normalizando ${file.name}`, pdfError)
              const normalized = await PDFDocument.load(parsedData, { ignoreEncryption: true })
              parsedData = exactArrayBuffer(await normalized.save())
              pdf = await pdfjsLib.getDocument({ data: parsedData.slice(0) }).promise
            }
            pageCount = pdf.numPages
            parsedCount += 1
            await pdf.cleanup()
          } catch (error) {
            console.error(`Falha ao abrir ${file.name}`, error)
            setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'error', mode: 'PDF protegido ou inválido' } : entry))
            setMessage(`Não foi possível ler ${file.name}: PDF protegido, inválido ou incompatível`)
          }
        }
        try {
          if (pageCount > 0) await renderPreview(parsedData)
        } catch (error) {
          console.error('Falha ao gerar prévia', error)
        }
        setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, data: parsedData, pages: pageCount } : entry))
      }
      if (parsedCount > 0) setMessage(`${parsedCount} documento${parsedCount !== 1 ? 's' : ''} pronto${parsedCount !== 1 ? 's' : ''} para processamento`)
    } finally {
      setIsAddingFiles(false)
    }
  }

  const processDocument = async (item: QueueFile, worker: OcrWorker, visionOptions: { enhance: boolean; autoOrient: boolean }) => {
    const bytes = item.data
    const source = await pdfjsLib.getDocument({ data: bytes.slice(0) }).promise
    const sourcePdf = await PDFDocument.load(bytes)
    const outputPdf = await PDFDocument.create()
    const font = await outputPdf.embedFont(StandardFonts.Helvetica)

    try {
      for (let pageNumber = 1; pageNumber <= source.numPages; pageNumber += 1) {
        if (cancelRequestedRef.current) throw new Error('OCR_CANCELLED')
        const pdfPage = await source.getPage(pageNumber)
        const scale = dpi / 72
        const viewport = pdfPage.getViewport({ scale })
        const canvas = document.createElement('canvas')
        canvas.width = Math.ceil(viewport.width); canvas.height = Math.ceil(viewport.height)
        await pdfPage.render({ canvas, canvasContext: canvas.getContext('2d')!, viewport }).promise
        let ocrCanvas = canvas
        try {
          ocrCanvas = await preprocessCanvas(canvas, visionOptions)
        } catch (visionError) {
          console.warn('OpenCV.js indisponível; continuando OCR sem pré-processamento geométrico', visionError)
          setMessage('OCR em andamento sem correção automática de orientação')
        }
        let result = await worker.recognize(ocrCanvas)
        const firstQuality = ocrQuality(result)
        if (visionOptions.autoOrient && firstQuality < 65) {
          const candidates = [
            { canvas: transformCandidate(ocrCanvas, true, false, false), label: 'espelhado horizontal' },
            { canvas: transformCandidate(ocrCanvas, false, true, false), label: 'espelhado vertical' },
            { canvas: transformCandidate(ocrCanvas, true, true, false), label: 'rotacionado 180°' },
            { canvas: transformCandidate(ocrCanvas, false, false, true), label: 'cores invertidas' },
            { canvas: transformCandidate(ocrCanvas, true, false, true), label: 'espelhado horizontal e cores invertidas' },
            { canvas: transformCandidate(ocrCanvas, false, true, true), label: 'espelhado vertical e cores invertidas' },
            { canvas: transformCandidate(ocrCanvas, true, true, true), label: 'rotacionado 180° e cores invertidas' },
          ]
          let selectedMode = 'normal'
          for (const candidate of candidates) {
            const candidateResult = await worker.recognize(candidate.canvas)
            if (ocrQuality(candidateResult) > ocrQuality(result)) { result = candidateResult; selectedMode = candidate.label }
            candidate.canvas.width = 1; candidate.canvas.height = 1
          }
          if (selectedMode !== 'normal') {
            setMessage(`Modo detectado automaticamente: ${selectedMode}`)
            setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, mode: selectedMode } : entry))
          }
        }
        const [page] = await outputPdf.copyPages(sourcePdf, [pageNumber - 1])
        outputPdf.addPage(page)
        const outputPage = outputPdf.getPage(outputPdf.getPageCount() - 1)
        const words = result.data.blocks?.flatMap((block) => block.paragraphs.flatMap((paragraph) => paragraph.lines.flatMap((line) => line.words))) ?? []
        drawOcrLayer(outputPage, words, canvas.width, canvas.height, font)
        setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, progress: pageNumber / source.numPages } : entry))
        canvas.width = 1
        canvas.height = 1
        if (ocrCanvas !== canvas) { ocrCanvas.width = 1; ocrCanvas.height = 1 }
        pdfPage.cleanup()
      }
    } finally {
      await source.cleanup()
    }
    const outputBytes = await outputPdf.save()
    const outputBuffer = outputBytes.buffer.slice(outputBytes.byteOffset, outputBytes.byteOffset + outputBytes.byteLength) as ArrayBuffer
    return URL.createObjectURL(new Blob([outputBuffer], { type: 'application/pdf' }))
  }

  const startProcessing = async () => {
    if (isProcessing || !queue.length) return
    cancelRequestedRef.current = false
    setIsProcessing(true)
    const visionOptions = { enhance, autoOrient }
    const { createWorker } = await import('tesseract.js')
    const worker = await createWorker(language, 1, { logger: (event) => { if (event.status === 'recognizing text') setMessage(`OCR da pagina ativa: ${Math.round(event.progress * 100)}%`) } })
    activeWorkerRef.current = worker
    let hadErrors = false
    try {
      const pending = queue.filter((entry) => entry.status === 'ready' && entry.pages > 0)
      if (!pending.length) {
        setMessage('Nenhum PDF válido está pronto para processar')
        setIsProcessing(false)
        return
      }
      for (const item of pending) {
        if (cancelRequestedRef.current) break
        setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'processing' } : entry))
        try {
          const outputUrl = await processDocument(item, worker, visionOptions)
          if (cancelRequestedRef.current) {
            URL.revokeObjectURL(outputUrl)
            setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'ready', progress: 0 } : entry))
            break
          }
          setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'done', progress: 1, outputUrl } : entry))
        } catch (error) {
          if (cancelRequestedRef.current) {
            setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'ready', progress: 0 } : entry))
            break
          }
          hadErrors = true
          console.error(error); setQueue((current) => current.map((entry) => entry.id === item.id ? { ...entry, status: 'error' } : entry)); setMessage('Nao foi possivel concluir um documento')
        }
      }
    } finally {
      if (activeWorkerRef.current === worker) await worker.terminate()
      activeWorkerRef.current = null
      setIsProcessing(false)
    }
    setMessage(cancelRequestedRef.current ? 'Processamento cancelado' : hadErrors ? 'Processamento concluído com arquivos não processados' : 'Processamento concluído no seu dispositivo')
  }

  const cancelProcessing = () => {
    if (!isProcessing || cancelRequestedRef.current) return
    cancelRequestedRef.current = true
    setMessage('Cancelando processamento...')
    const worker = activeWorkerRef.current
    activeWorkerRef.current = null
    if (worker) void worker.terminate()
  }

  const clearQueue = () => { queue.forEach((item) => item.outputUrl && URL.revokeObjectURL(item.outputUrl)); setQueue([]); setPreviewUrl(undefined); setMessage('Pronto para receber documentos') }

  const downloadAll = () => {
    completedItems.forEach((item) => {
      const link = document.createElement('a')
      link.href = item.outputUrl!
      link.download = `${item.file.name.replace(/\.pdf$/i, '')}_OCR.pdf`
      link.click()
    })
  }

  return <main className={`app-shell ${allDone ? 'is-complete' : ''}`}>
    {allDone && <section className="completion-screen"><header className="completion-header"><div className="completion-brand"><img src="/clark_logo.png" alt="Clark Solutions" /><span /><div><strong>PDF OCR</strong><small>Transforme documentos digitalizados em PDFs pesquisáveis</small></div></div><div className="completion-badge"><ShieldCheck size={14} /> Concluído</div></header><div className="completion-body"><div className="completion-hero"><div className="completion-heading"><div className="completion-check"><ShieldCheck size={28} /></div><div><h2>OCR concluído</h2><p>{completedItems.length} arquivo{completedItems.length !== 1 ? 's' : ''} processado{completedItems.length !== 1 ? 's' : ''} · {completedItems.reduce((total, item) => total + item.pages, 0)} páginas pesquisáveis</p></div></div><button className="download-all" onClick={downloadAll}><Upload size={15} /> Baixar todos</button></div><div className="generated-panel"><div className="generated-title"><span /> <strong>Arquivos gerados</strong></div>{completedItems.map((item) => <div className="generated-file" key={item.id}><FileText size={21} /><div className="generated-meta"><strong>{item.file.name.replace(/\.pdf$/i, '')}_OCR.pdf</strong><span>{item.pages} páginas pesquisáveis{item.mode ? ` · ${item.mode}` : ''} · {(item.file.size / 1024 / 1024).toFixed(1)} MB</span></div><div className="generated-actions"><a href={item.outputUrl} target="_blank" rel="noreferrer">Visualizar</a><a className="download-one" href={item.outputUrl} download={`${item.file.name.replace(/\.pdf$/i, '')}_OCR.pdf`}>Baixar</a></div></div>)}</div><button className="new-files-button" onClick={clearQueue}><RotateCcw size={15} /> Processar novos arquivos</button></div><footer className="completion-footer"><span>Clark Solutions · PDF OCR · clark-pdf.fly.dev</span><span>Processamento 100% local</span></footer></section>}
    <header className="topbar"><div className="brand"><img src="/clark_logo.png" alt="Clark Solutions" /><span className="brand-divider" /><div><strong>PDF OCR</strong><small>Transforme documentos digitalizados em PDFs pesquisáveis</small></div></div><nav className="site-nav"><span>Soluções</span><span>Tecnologias</span><span>Serviços</span><strong>Clark Solutions</strong></nav><div className="completion-badge"><LockKeyhole size={14} /> Processamento local</div></header>
    <section className="intro"><div><p className="eyebrow">ENGENHARIA DIGITAL · OCR LOCAL</p><h1>Documentos técnicos,<br /><em>prontos para decisão.</em></h1><p className="lede">Converta PDFs digitalizados em documentos pesquisáveis com um fluxo técnico, preciso e executado integralmente no seu dispositivo.</p></div><div className="engine-card"><span className="pulse" /><div><strong>Clark PDF ativo</strong><span>WebAssembly · processamento local</span></div></div></section>
    <section className="workspace"><div className="main-column"><div className={`dropzone ${isDragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); void addFiles(event.dataTransfer.files) }} onClick={() => inputRef.current?.click()}><input ref={inputRef} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(event) => { if (event.target.files) void addFiles(event.target.files); event.target.value = '' }} /><div className="upload-icon"><Upload size={22} /></div><div><strong>Solte seus PDFs aqui</strong><span>ou clique para escolher arquivos</span></div><kbd>PDF</kbd></div>
      <div className="section-heading"><div><span className="eyebrow">FILA DE PROCESSAMENTO</span><h2>Documentos <small>{queue.length}</small></h2></div>{queue.length > 0 && <button className="text-button" onClick={clearQueue}><RotateCcw size={15} /> Limpar fila</button>}</div>
      <div className="queue-list">{!queue.length && <div className="empty-state"><FileText size={22} /><span>Adicione um PDF para começar</span></div>}{queue.map((item) => <div className="queue-item" key={item.id}><div className="file-icon"><FileText size={18} /></div><div className="file-meta"><strong>{item.file.name}</strong><span>{item.pages} página{item.pages !== 1 ? 's' : ''} · {(item.file.size / 1024 / 1024).toFixed(1)} MB</span>{item.mode && <span className="detected-mode">{item.mode}</span>}{item.status === 'processing' && <div className="progress-track"><i style={{ width: `${Math.max(3, item.progress * 100)}%` }} /></div>}</div><span className={`status ${item.status}`}>{item.status === 'ready' ? (item.pages > 0 ? 'Pronto' : 'Lendo...') : item.status === 'processing' ? `${Math.round(item.progress * 100)}%` : item.status === 'done' ? 'Concluído' : 'Erro'}</span>{item.outputUrl && <a className="download-link" href={item.outputUrl} download={`${item.file.name.replace(/\.pdf$/i, '')}_OCR.pdf`}>Baixar</a>}<button className="icon-button" aria-label={`Remover ${item.file.name}`} onClick={(event) => { event.stopPropagation(); setQueue((current) => current.filter((entry) => entry.id !== item.id)) }}><X size={16} /></button></div>)}</div>
      <div className="action-row"><span className="status-line"><span className={`status-dot ${isProcessing || isAddingFiles ? 'working' : ''}`} />{message}</span><div className="action-buttons">{isProcessing && <button className="cancel-button" onClick={cancelProcessing}><Square size={15} fill="currentColor" /> Cancelar</button>}<button className="primary-button" disabled={!queue.length || isProcessing || isAddingFiles} onClick={() => void startProcessing()}><Play size={16} fill="currentColor" /> {isAddingFiles ? 'Lendo PDF...' : isProcessing ? 'Processando...' : 'Executar OCR'}</button></div></div></div>
      <aside className="side-column"><div className="preview-panel"><div className="panel-title"><span>PRÉVIA</span><span className="page-badge">{queue.length ? 'PÁG. 01' : 'AGUARDANDO'}</span></div><div className="preview-canvas">{previewUrl ? <img src={previewUrl} alt="Prévia da primeira página" /> : <div className="preview-placeholder"><ScanText size={30} /><span>A prévia da primeira<br />página aparecerá aqui</span></div>}</div></div><div className="settings-panel"><div className="panel-title"><span>CONFIGURAÇÃO</span><span className="local-chip"><ShieldCheck size={13} /> LOCAL</span></div><label>Idioma<select value={language} onChange={(event) => setLanguage(event.target.value)}>{languageOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select></label><label>Qualidade<select value={dpi} onChange={(event) => setDpi(Number(event.target.value))}><option value={150}>150 DPI · rápido</option><option value={200}>200 DPI · equilibrado</option><option value={300}>300 DPI · detalhado</option></select></label><div className="vision-options"><label><input type="checkbox" checked={autoOrient} onChange={(event) => setAutoOrient(event.target.checked)} /> Corrigir inclinação, rotação e espelhamento</label><label><input type="checkbox" checked={enhance} onChange={(event) => setEnhance(event.target.checked)} /> Melhorar contraste e reduzir ruído</label></div><div className="backend-row"><span>Execução</span><strong><span className="green-dot" /> WebAssembly + OpenCV.js</strong></div></div></aside></section>
    <footer><span>Clark PDF · Clark Solutions</span><span>Processamento 100% local · Nenhum arquivo é enviado</span></footer>
  </main>
}

export default App
