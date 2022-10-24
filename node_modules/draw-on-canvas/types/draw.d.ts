export default class Draw {
  static getPixelArray(ctx: CanvasRenderingContext2D): Uint8ClampedArray;
  static getGreyScalePixelArray(ctx: CanvasRenderingContext2D): Uint8ClampedArray;
  static getPixelMatrix(ctx: CanvasRenderingContext2D): unknown[][];
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private strokeColor: string;
  private drawing: { color: string, strokeWeight: number, points: number[] }[];
  private height: number;
  private width: number;
  constructor(element: HTMLElement, width: number, height: number, opts: { backgroundColor?: string, strokeColor?: string, strokeWeight?: number });

  changeStrokeColor(strokeColor: string): void;
  changeBackgroundColor(backgroundColor: string | CanvasGradient | CanvasPattern): void;
  changeStrokeWeight(strokeWeight: number): void
  getDrawing(): any[];
  downloadPNG(filename?: string): void;
  private setupEventListeners(): void;
  private onMouseMove(event: MouseEvent): void;
  private onMouseDown(): void;
  private onMouseUp(): void;
  getPixelArray(): Uint8ClampedArray;
  getGreyScalePixelArray(): Uint8ClampedArray;
  reset(): void;
  private clearCanvas(): void;
  private draw(): void;
  private drawLinePoint(point: {x: number, y: number}[]): void;
  private drawStroke(stroke: { color: string, strokeWeight: number, points: number[] }[]): void;
}
