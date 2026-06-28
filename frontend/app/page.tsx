"use client";

import { Camera, ImageIcon, Loader2, Play, Upload, Video } from "lucide-react";
import { ChangeEvent, MutableRefObject, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Mode = "image" | "video" | "camera";
type ModelOption = "best2" | "resnet";

type Detection = {
  class_id: number;
  label: string;
  confidence: number;
  box: { x1: number; y1: number; x2: number; y2: number };
  detector_label?: string;
  detector_confidence?: number;
};

type FrameSize = {
  width: number;
  height: number;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CAMERA_CAPTURE_MS = 900;
const VIDEO_CAPTURE_MS = 700;
const CAMERA_MAX_WIDTH = 1280;
const VIDEO_MAX_WIDTH = 1280;
const CAMERA_JPEG_QUALITY = 0.6;
const VIDEO_JPEG_QUALITY = 0.65;
const DEFAULT_IMAGE_SIZE = 1280;
const CAMERA_IMAGE_SIZE = DEFAULT_IMAGE_SIZE;
const VIDEO_IMAGE_SIZE = DEFAULT_IMAGE_SIZE;
const DEFAULT_MAX_DET = 10000;
const CAMERA_MAX_DET = DEFAULT_MAX_DET;
const VIDEO_MAX_DET = DEFAULT_MAX_DET;
const DEFAULT_IOU = 0.45;
const DEFAULT_RESULT_QUALITY = 88;
const CAMERA_RESULT_QUALITY = 72;
const VIDEO_RESULT_QUALITY = 76;

const modes: Array<{ value: Mode; label: string; icon: typeof ImageIcon }> = [
  { value: "image", label: "Image", icon: ImageIcon },
  { value: "video", label: "Video", icon: Video },
  { value: "camera", label: "Camera", icon: Camera }
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("image");
  const [modelOption, setModelOption] = useState<ModelOption>("best2");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(DEFAULT_IOU);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [imageResult, setImageResult] = useState("");
  const [imageFrameSize, setImageFrameSize] = useState<FrameSize | null>(null);
  const [liveVideoUrl, setLiveVideoUrl] = useState("");
  const [liveAnnotatedFrame, setLiveAnnotatedFrame] = useState("");
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraFrameSize, setCameraFrameSize] = useState<FrameSize | null>(null);
  const [videoFrameSize, setVideoFrameSize] = useState<FrameSize | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const cameraIntervalRef = useRef<number | null>(null);
  const videoIntervalRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraBusyRef = useRef(false);
  const videoBusyRef = useRef(false);

  useEffect(() => {
    return () => {
      stopCamera();
      stopLiveVideo();
    };
  }, []);

  useEffect(() => {
    if (cameraOn && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
    }
  }, [cameraOn]);

  function resetResult() {
    stopLiveVideo();
    if (imageResult.startsWith("blob:")) URL.revokeObjectURL(imageResult);
    setError("");
    setImageResult("");
    setImageFrameSize(null);
    setLiveAnnotatedFrame("");
    setDetections([]);
  }

  async function detectImage(file: File) {
    resetResult();
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const params = new URLSearchParams({
        confidence: String(confidence),
        classifier_conf: String(confidence),
        imgsz: String(DEFAULT_IMAGE_SIZE),
        max_det: String(DEFAULT_MAX_DET),
        iou: String(iou),
        quality: String(DEFAULT_RESULT_QUALITY),
        model: modelOption
      });
      const response = await fetch(`${API_URL}/detect/image?${params.toString()}`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setImageResult(data.annotated_image);
      setDetections(data.detections);
    } catch (err) {
      setError(readError(err));
    } finally {
      setLoading(false);
    }
  }

  async function detectVideo(file: File) {
    resetResult();
    setLiveVideoUrl(URL.createObjectURL(file));
  }

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (mode === "image") detectImage(file);
    if (mode === "video") detectVideo(file);
    event.target.value = "";
  }

  async function startCamera() {
    resetResult();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 360 },
        audio: false
      });
      streamRef.current = stream;
      setCameraOn(true);
      cameraIntervalRef.current = window.setInterval(captureCameraFrame, CAMERA_CAPTURE_MS);
    } catch (err) {
      setError(readError(err));
      stopCamera();
    }
  }

  function stopCamera() {
    if (cameraIntervalRef.current) window.clearInterval(cameraIntervalRef.current);
    cameraIntervalRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    cameraBusyRef.current = false;
    setCameraFrameSize(null);
    setCameraOn(false);
  }

  function startVideoFrameLoop() {
    if (videoIntervalRef.current) return;
    captureVideoFrame();
    videoIntervalRef.current = window.setInterval(captureVideoFrame, VIDEO_CAPTURE_MS);
  }

  function stopVideoFrameLoop() {
    if (videoIntervalRef.current) window.clearInterval(videoIntervalRef.current);
    videoIntervalRef.current = null;
    videoBusyRef.current = false;
  }

  function stopLiveVideo() {
    stopVideoFrameLoop();
    if (liveVideoUrl) URL.revokeObjectURL(liveVideoUrl);
    setLiveVideoUrl("");
    setVideoFrameSize(null);
  }

  async function captureVideoFrame() {
    await captureDetectFrame({
      busyRef: videoBusyRef,
      maxWidth: VIDEO_MAX_WIDTH,
      imageSize: VIDEO_IMAGE_SIZE,
      maxDet: VIDEO_MAX_DET,
      resultQuality: VIDEO_RESULT_QUALITY,
      jpegQuality: VIDEO_JPEG_QUALITY,
      setFrameSize: setVideoFrameSize
    });
  }

  async function captureCameraFrame() {
    await captureDetectFrame({
      busyRef: cameraBusyRef,
      maxWidth: CAMERA_MAX_WIDTH,
      imageSize: CAMERA_IMAGE_SIZE,
      maxDet: CAMERA_MAX_DET,
      resultQuality: CAMERA_RESULT_QUALITY,
      jpegQuality: CAMERA_JPEG_QUALITY,
      setFrameSize: setCameraFrameSize
    });
  }

  async function captureDetectFrame({
    busyRef,
    maxWidth,
    imageSize,
    maxDet,
    resultQuality,
    jpegQuality,
    setFrameSize
  }: {
    busyRef: MutableRefObject<boolean>;
    maxWidth: number;
    imageSize: number;
    maxDet: number;
    resultQuality: number;
    jpegQuality: number;
    setFrameSize: (size: FrameSize) => void;
  }) {
    if (busyRef.current || !videoRef.current || !canvasRef.current) return;
    const videoEl = videoRef.current;
    if (videoEl.readyState < 2 || videoEl.paused || videoEl.ended) return;

    const sourceWidth = videoEl.videoWidth || 1;
    const sourceHeight = videoEl.videoHeight || 1;
    const scale = Math.min(1, maxWidth / sourceWidth);
    const frameSize = {
      width: Math.max(1, Math.round(sourceWidth * scale)),
      height: Math.max(1, Math.round(sourceHeight * scale))
    };

    busyRef.current = true;
    const canvas = canvasRef.current;
    canvas.width = frameSize.width;
    canvas.height = frameSize.height;
    setFrameSize(frameSize);
    canvas.getContext("2d")?.drawImage(videoEl, 0, 0, frameSize.width, frameSize.height);

    canvas.toBlob(async (blob) => {
      if (!blob) {
        busyRef.current = false;
        return;
      }
      try {
        const formData = new FormData();
        formData.append("file", new File([blob], "frame.jpg", { type: "image/jpeg" }));
        const params = new URLSearchParams({
          confidence: String(confidence),
          classifier_conf: String(confidence),
          imgsz: String(imageSize),
          max_det: String(maxDet),
          iou: String(iou),
          quality: String(resultQuality),
          model: modelOption
        });
        const response = await fetch(`${API_URL}/detect/image?${params.toString()}`, {
          method: "POST",
          body: formData
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        setDetections(data.detections);
        setLiveAnnotatedFrame(data.annotated_image);
        setError("");
      } catch (err) {
        setError(readError(err));
      } finally {
        busyRef.current = false;
      }
    }, "image/jpeg", jpegQuality);
  }

  function selectMode(nextMode: Mode) {
    stopCamera();
    setMode(nextMode);
    resetResult();
  }

  return (
    <main className="min-h-screen bg-slate-50 p-4 text-slate-950 md:p-6">
      <Card className="mx-auto grid min-h-[calc(100vh-2rem)] max-w-[1360px] overflow-hidden shadow-xl shadow-slate-200/70 md:min-h-[calc(100vh-3rem)]">
        <header className="flex min-h-14 items-center border-b border-border px-5">
          <div className="flex min-w-0 items-baseline gap-3">
            <span className="text-base font-semibold">Fruit Detection</span>
            <span className="truncate text-sm text-muted-foreground">
              {modelOption === "best2" ? "YOLO" : "YOLO + ResNet"}
            </span>
          </div>
        </header>

        <div className="grid min-h-0 grid-cols-1 md:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="border-b border-border bg-white p-4 md:border-b-0 md:border-r">
            <div className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-muted p-1">
              {modes.map((item) => {
                const Icon = item.icon;
                return (
                  <Button
                    key={item.value}
                    type="button"
                    variant={mode === item.value ? "outline" : "ghost"}
                    className={cn("min-w-0 px-2", mode === item.value && "bg-white shadow-sm")}
                    onClick={() => selectMode(item.value)}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span className="truncate">{item.label}</span>
                  </Button>
                );
              })}
            </div>

            <Card className="mt-4 p-4">
              <label className="grid gap-3 text-sm text-muted-foreground">
                <span>Confidence {confidence.toFixed(2)}</span>
                <input
                  className="h-2 w-full accent-primary"
                  type="range"
                  min="0.01"
                  max="0.99"
                  step="0.01"
                  value={confidence}
                  onChange={(event) => setConfidence(Number(event.target.value))}
                />
              </label>
            </Card>

            <Card className="mt-4 p-4">
              <label className="grid gap-3 text-sm text-muted-foreground">
                <span>IoU {iou.toFixed(2)}</span>
                <input
                  className="h-2 w-full accent-primary"
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.01"
                  value={iou}
                  onChange={(event) => setIou(Number(event.target.value))}
                />
              </label>
            </Card>

            <div className="mt-4 grid gap-1 rounded-lg border border-border bg-muted p-1">
              <Button
                type="button"
                variant={modelOption === "best2" ? "outline" : "ghost"}
                className={cn(modelOption === "best2" && "bg-white shadow-sm")}
                onClick={() => {
                  setModelOption("best2");
                  resetResult();
                }}
              >
                YOLO
              </Button>
              <Button
                type="button"
                variant={modelOption === "resnet" ? "outline" : "ghost"}
                className={cn(modelOption === "resnet" && "bg-white shadow-sm")}
                onClick={() => {
                  setModelOption("resnet");
                  resetResult();
                }}
              >
                YOLO + ResNet
              </Button>
            </div>

            {mode !== "camera" ? (
              <label className="mt-4 grid min-h-36 cursor-pointer place-items-center gap-3 rounded-lg border border-dashed border-slate-300 bg-white p-5 text-center text-sm text-muted-foreground transition-colors hover:border-primary hover:bg-green-50 hover:text-green-800">
                <Upload className="h-5 w-5" />
                <span>{mode === "image" ? "Choose an image" : "Choose a video"}</span>
                <input className="hidden" type="file" accept={mode === "image" ? "image/*" : "video/*"} onChange={handleFile} />
              </label>
            ) : (
              <div className="mt-4 grid grid-cols-2 gap-2">
                <Button type="button" onClick={startCamera} disabled={cameraOn}>
                  <Play className="h-4 w-4" />
                  Start camera
                </Button>
                <Button type="button" variant="outline" onClick={stopCamera} disabled={!cameraOn}>
                  Stop camera
                </Button>
              </div>
            )}
          </aside>

          <section className="relative grid min-h-[520px] place-items-center overflow-hidden bg-slate-100 p-4 md:min-h-[680px]">
            <canvas ref={canvasRef} className="hidden" />

            {mode === "camera" && cameraOn && (
              <MediaStage frameSize={cameraFrameSize}>
                <video ref={videoRef} className="hidden" autoPlay playsInline muted />
                <AnnotatedFrame src={liveAnnotatedFrame} />
              </MediaStage>
            )}

            {mode === "video" && liveVideoUrl && (
              <MediaStage frameSize={videoFrameSize}>
                <video
                  ref={videoRef}
                  className="absolute inset-0 h-full w-full object-contain"
                  src={liveVideoUrl}
                  controls
                  autoPlay
                  playsInline
                  onPlay={startVideoFrameLoop}
                  onPause={stopVideoFrameLoop}
                  onEnded={stopVideoFrameLoop}
                  onLoadedMetadata={() => {
                    if (videoRef.current) {
                      setVideoFrameSize({
                        width: videoRef.current.videoWidth || 16,
                        height: videoRef.current.videoHeight || 9
                      });
                    }
                  }}
                />
                <AnnotatedFrame src={liveAnnotatedFrame} />
              </MediaStage>
            )}

            {loading && (
              <Card className="grid min-h-64 w-full max-w-lg place-items-center gap-3 border-dashed p-8 text-center text-muted-foreground">
                <Loader2 className="h-8 w-8 animate-spin" />
                <span>Detecting...</span>
              </Card>
            )}

            {!loading && mode === "image" && imageResult && (
              <MediaStage frameSize={imageFrameSize}>
                <img
                  className="absolute inset-0 h-full w-full object-contain"
                  src={imageResult}
                  alt="Detection result"
                  onLoad={(event) => {
                    setImageFrameSize({
                      width: event.currentTarget.naturalWidth || 16,
                      height: event.currentTarget.naturalHeight || 9
                    });
                  }}
                />
              </MediaStage>
            )}

            {!loading && mode !== "camera" && !imageResult && !liveVideoUrl && (
              <Card className="grid min-h-64 w-full max-w-lg place-items-center gap-3 border-dashed p-8 text-center text-muted-foreground">
                <Upload className="h-8 w-8" />
                <span>Upload a file to view results</span>
              </Card>
            )}

            {!loading && mode === "camera" && !cameraOn && (
              <Card className="grid min-h-64 w-full max-w-lg place-items-center gap-3 border-dashed p-8 text-center text-muted-foreground">
                <Camera className="h-8 w-8" />
                <span>Start the camera for live detection</span>
              </Card>
            )}

            {error && (
              <div className="absolute bottom-4 left-4 right-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}
          </section>
        </div>
      </Card>
    </main>
  );
}

function MediaStage({ children, frameSize }: { children: React.ReactNode; frameSize: FrameSize | null }) {
  const aspect = frameSize ? frameSize.width / frameSize.height : 16 / 9;

  return (
    <div
      className="relative overflow-hidden rounded-lg border border-border bg-black shadow-lg"
      style={{
        aspectRatio: String(aspect),
        width: `min(100%, calc(76vh * ${aspect}))`
      }}
    >
      {children}
    </div>
  );
}

function AnnotatedFrame({ src }: { src: string }) {
  if (!src) return null;
  return <img className="absolute inset-0 h-full w-full object-contain" src={src} alt="Detection frame" />;
}

function readError(err: unknown) {
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}
