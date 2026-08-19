type OpenCvApi = {
  Mat: new () => any
  onRuntimeInitialized?: () => void
  matFromImageData: (data: ImageData) => any
  imread: (canvas: HTMLCanvasElement) => any
  imshow: (canvas: HTMLCanvasElement, image: any) => void
  cvtColor: (source: any, destination: any, code: number) => void
  equalizeHist: (source: any, destination: any) => void
  GaussianBlur: (source: any, destination: any, size: any, sigmaX: number) => void
  Canny: (source: any, destination: any, threshold1: number, threshold2: number) => void
  HoughLinesP: (source: any, destination: any, rho: number, theta: number, threshold: number, minLineLength: number, maxLineGap: number) => void
  getRotationMatrix2D: (center: any, angle: number, scale: number) => any
  warpAffine: (source: any, destination: any, matrix: any, size: any, interpolation: number, borderMode: number, borderValue: any) => void
  flip: (source: any, destination: any, flipCode: number) => void
  rotate: (source: any, destination: any, rotateCode: number) => void
  Size: new (width: number, height: number) => any
  Point: new (x: number, y: number) => any
  Scalar: new (red: number, green: number, blue: number, alpha?: number) => any
  COLOR_RGBA2GRAY: number
  COLOR_GRAY2RGBA: number
  INTER_CUBIC: number
  BORDER_CONSTANT: number
  ROTATE_90_CLOCKWISE: number
  ROTATE_90_COUNTERCLOCKWISE: number
}

let openCvPromise: Promise<OpenCvApi> | undefined

async function loadOpenCv(): Promise<OpenCvApi> {
  if (!openCvPromise) {
    openCvPromise = import('@techstark/opencv-js').then(async (module) => {
      const candidate = module.default as unknown as OpenCvApi | Promise<OpenCvApi>
      return candidate instanceof Promise ? candidate : candidate.Mat ? candidate : new Promise<OpenCvApi>((resolve) => {
        candidate.onRuntimeInitialized = () => resolve(candidate)
      })
    })
  }
  return openCvPromise
}

function estimateOrientation(image: any, cv: OpenCvApi): { rotation: 0 | 90 | 270; skew: number } {
  const gray = new cv.Mat()
  const edges = new cv.Mat()
  const lines = new cv.Mat()
  cv.cvtColor(image, gray, cv.COLOR_RGBA2GRAY)
  cv.Canny(gray, edges, 50, 150)
  cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 35, Math.max(25, image.cols / 10), 12)
  const angles: number[] = []
  let verticalLines = 0
  for (let index = 0; index < lines.rows; index += 1) {
    const line = lines.data32S.slice(index * 4, index * 4 + 4)
    const angle = Math.atan2(line[3] - line[1], line[2] - line[0]) * 180 / Math.PI
    const normalized = Math.abs(angle)
    if (normalized >= 60 && normalized <= 120) verticalLines += 1
    if (normalized <= 20) angles.push(angle)
  }
  gray.delete(); edges.delete(); lines.delete()
  if (verticalLines > angles.length && verticalLines >= 2) return { rotation: 90, skew: 0 }
  if (!angles.length) return { rotation: 0, skew: 0 }
  const average = angles.reduce((sum, angle) => sum + angle, 0) / angles.length
  return { rotation: 0, skew: Math.abs(average) < 0.4 ? 0 : average }
}

export async function preprocessCanvas(canvas: HTMLCanvasElement, options: { enhance: boolean; autoOrient: boolean }): Promise<HTMLCanvasElement> {
  if (!options.enhance && !options.autoOrient) return canvas
  const cv = await loadOpenCv()
  const source = cv.imread(canvas)
  const working = new cv.Mat()
  let outputCanvas = canvas
  try {
    source.copyTo(working)

    const orientation = options.autoOrient ? estimateOrientation(working, cv) : { rotation: 0 as const, skew: 0 }
    if (orientation.rotation === 90) {
      const rotated = new cv.Mat()
      cv.rotate(working, rotated, cv.ROTATE_90_CLOCKWISE)
      working.delete()
      outputCanvas = document.createElement('canvas')
      outputCanvas.width = canvas.height; outputCanvas.height = canvas.width
      cv.imshow(outputCanvas, rotated)
      rotated.delete()
      return preprocessCanvas(outputCanvas, { ...options, autoOrient: false })
    }

    if (options.enhance) {
      const gray = new cv.Mat()
      const blurred = new cv.Mat()
      cv.cvtColor(working, gray, cv.COLOR_RGBA2GRAY)
      cv.equalizeHist(gray, blurred)
      cv.GaussianBlur(blurred, blurred, new cv.Size(3, 3), 0)
      cv.cvtColor(blurred, working, cv.COLOR_GRAY2RGBA)
      gray.delete(); blurred.delete()
    }

    if (options.autoOrient) {
      const angle = orientation.skew
      if (angle) {
        const center = new cv.Point(working.cols / 2, working.rows / 2)
        const matrix = cv.getRotationMatrix2D(center, angle, 1)
        const rotated = new cv.Mat()
        cv.warpAffine(working, rotated, matrix, new cv.Size(working.cols, working.rows), cv.INTER_CUBIC, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255))
        working.delete()
        outputCanvas = document.createElement('canvas')
        outputCanvas.width = canvas.width; outputCanvas.height = canvas.height
        cv.imshow(outputCanvas, rotated)
        rotated.delete(); matrix.delete(); center.delete()
        return outputCanvas
      }
    }
    cv.imshow(outputCanvas, working)
    return outputCanvas
  } finally {
    source.delete()
    if (!outputCanvas || outputCanvas === canvas) working.delete()
  }
}
