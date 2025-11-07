import argparse
import time
from collections import Counter
from pathlib import Path
import threading
import queue
import sys

import cv2
import numpy as np
from ultralytics import YOLO

import torch
import torchvision.transforms as T
from torchvision import models

import serial


# -------------------- Places365 helpers -------------------- #

def load_places365(places_dir: Path, device="cpu"):
    labels_path = places_dir / "categories_places365.txt"
    weights_path = places_dir / "resnet50_places365.pth.tar"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    with open(labels_path, "r") as f:
        classes = [line.strip().split(" ")[0][3:] for line in f]

    model = models.resnet50(num_classes=365)
    sd = torch.load(weights_path, map_location=device)
    sd = sd.get("state_dict", sd)
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    model.to(device).eval()

    tfm = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return model, classes, tfm


@torch.no_grad()
def predict_scene_bgr(frame_bgr: np.ndarray, model, classes, tfm, device="cpu", topk=1):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(rgb).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    p, idx = probs.topk(topk, dim=1)
    return [(classes[i], float(pv)) for i, pv in zip(idx[0].tolist(), p[0].tolist())]


# -------------------- Caption helpers -------------------- #

def humanize_counts(counts: Counter) -> str:
    if not counts:
        return "no notable objects"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    phrases = []
    for name, n in ordered:
        if n == 1:
            qty = "a"
            word = name
        else:
            qty = str(n)
            word = name + ("es" if name.endswith(("s", "x", "z", "ch", "sh")) else "s")
        phrases.append(f"{qty} {word}")
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def make_caption(counts: Counter, scene_hint: str | None = None) -> str:
    body = humanize_counts(counts)
    if scene_hint:
        return f"A {scene_hint} with {body}."
    if body.startswith(("a ", "an ", "no ")):
        return body[0].upper() + body[1:] + ("" if body.endswith(".") else ".")
    return f"{body.capitalize()}."


# -------------------- Main -------------------- #

def main():
    ap = argparse.ArgumentParser(description="YOLOv11 + Places365 → timed scene captions with TTS")

    ap.add_argument("--model", type=str,
                    default=str(Path(__file__).resolve().parent / "yolo11n.pt"),
                    help="Path to YOLO model (.pt, .engine, etc.).")
    ap.add_argument("--source", type=str, default="0",
                    help="Camera index (0/1) or video/RTSP path.")
    ap.add_argument("--conf", type=float, default=0.7,
                    help="YOLO confidence threshold.")      #updated conf from 0.35 to 0.7 for higher confidence in labels
    ap.add_argument("--min_area", type=float, default=0.01,
                    help="Min bbox area ratio vs frame (0.01 = 1%).")
    ap.add_argument("--topk", type=int, default=6,
                    help="Max distinct categories to mention.")
    ap.add_argument("--show", action="store_true",
                    help="Show window with overlay.")
    ap.add_argument("--save", action="store_true",
                    help="Save annotated video to ./runs/caption/*.mp4")

    ap.add_argument("--scene_mode", type=str, default="auto",
                    choices=["manual", "auto"],
                    help="manual: use --scene; auto: infer with Places365.")
    ap.add_argument("--scene", type=str, default="",
                    help="Manual scene hint when scene_mode=manual.")
    ap.add_argument("--scene_every", type=int, default=15,
                    help="Infer scene every N frames in auto mode.")

    ap.add_argument("--places_dir", type=str,
                    default=str(Path(__file__).resolve().parent / "places"),
                    help="Folder containing Places365 weights/labels.")
    ap.add_argument("--device", type=str, default="cpu",
                    help="Torch device for Places365 (cpu or cuda:0).")

    # TTS config
    ap.add_argument("--tts", type=str, default="pyttsx3",
                    choices=["none", "pyttsx3"],
                    help="TTS backend: pyttsx3 (non-blocking) or none.")
    ap.add_argument("--speak_every", type=float, default=6.0,
                    help="Speak the caption at least every N seconds.")
    ap.add_argument("--speak_on_change", action="store_true", default=True,
                    help="Also speak when caption text changes.")
    ap.add_argument("--min_gap", type=float, default=5.0,
                    help="Minimum seconds between spoken captions (hard limit).")

    args = ap.parse_args()

    print(f"[Captioner] model={args.model} | src={args.source} | device={args.device}")
    print(f"[Captioner] tts={args.tts} | speak_every={args.speak_every}s | "
          f"speak_on_change={args.speak_on_change} | min_gap={args.min_gap}s")

    # ---------- Init YOLO ----------
    model = YOLO(args.model)

    # ---------- Init Places365 ----------
    places = None
    if args.scene_mode == "auto":
        pdir = Path(args.places_dir)
        p_model, p_classes, p_tfm = load_places365(pdir, device=args.device)
        places = {
            "model": p_model,
            "classes": p_classes,
            "tfm": p_tfm,
            "device": args.device,
            "cache": ("", 0.0),
        }
        print(f"[Places365] Loaded from {pdir}")

    # ---------- Init TTS (background) ----------
    tts_engine = None
    tts_queue = None
    tts_thread = None

    if args.tts == "pyttsx3":
        try:
            import pyttsx3

            if sys.platform.startswith("linux"):
                drv = "espeak"
            elif sys.platform.startswith("win"):
                drv = "sapi5"
            elif sys.platform == "darwin":
                drv = "nsss"
            else:
                drv = None

            tts_engine = pyttsx3.init(driverName=drv) if drv else pyttsx3.init()
            tts_engine.setProperty("rate", 175)
            tts_engine.setProperty("volume", 1.0)
            print(f"[TTS] pyttsx3 initialized (driver={drv})")

            tts_queue = queue.Queue()

            def tts_worker():
                while True:
                    text = tts_queue.get()
                    if text is None:
                        break
                    try:
                        tts_engine.say(text)
                        tts_engine.runAndWait()
                    except Exception as e:
                        print(f"[WARN] TTS error: {e}")
                    tts_queue.task_done()

            tts_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_thread.start()

        except Exception as e:
            print(f"[WARN] pyttsx3 init failed: {e}")
            tts_engine = None
            tts_queue = None
            tts_thread = None


    # ---------- Init Arduino Serial ----------
    try:
        arduino = serial.Serial('COM4', 9600, timeout=0.1)  # or '/dev/ttyACM0' on Jetson
        print("[Arduino] Serial connected successfully")
    except Exception as e:
        print(f"[WARN] Arduino serial not connected: {e}")
        arduino = None

    # ---------- Open video source ----------
    src = args.source
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src}")

    # Lighten load for Jetson
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Video] {width}x{height} @ {fps:.1f} fps")

    # ---------- Optional video writer ----------
    writer = None
    if args.save:
        out_dir = Path("runs/caption")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"caption_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
        print(f"[INFO] Saving video to {out_path}")

    # ---------- State ----------
    last_caption = ""
    last_speak_time = 0.0
    frame_idx = 0


    # --- Check Arduino button press before main loop ---
    if arduino is not None and arduino.in_waiting > 0:
        line = arduino.readline().decode('utf-8').strip()
        
        if line.startswith("BUTTON_PRESSED:"):
            # Example: Arduino sends "BUTTON_PRESSED: 45.2"
            try:
                distance = float(line.split(":")[1].strip())
                message = f"{distance:.0f} centimeters away."
                print(f"[BUTTON] Speaking distance: {message}")

                # Interrupt TTS queue for priority speech
                if tts_queue is not None:
                    # Optional: clear pending captions
                    with tts_queue.mutex:
                        tts_queue.queue.clear()
                    tts_queue.put_nowait(message)

            except Exception as e:
                print(f"[WARN] Bad button data: {line} ({e})")
    

    # -------------------- Main loop -------------------- #

    while True:

        # --- Check Arduino button press before object detection each loop ---
        if arduino is not None and arduino.in_waiting > 0:
            line = arduino.readline().decode('utf-8').strip()
            
            if line.startswith("BUTTON_PRESSED:"):
                # Example: Arduino sends "BUTTON_PRESSED: 45.2"
                try:
                    distance = float(line.split(":")[1].strip())
                    message = f"{distance:.0f} centimeters away."
                    print(f"[BUTTON] Speaking distance: {message}")

                    # Interrupt TTS queue for priority speech
                    if tts_queue is not None:
                        # Optional: clear pending captions
                        with tts_queue.mutex:
                            tts_queue.queue.clear()
                        tts_queue.put_nowait(message)

                except Exception as e:
                    print(f"[WARN] Bad button data: {line} ({e})")

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        H, W = frame.shape[:2]
        min_area = args.min_area * (W * H)

        # YOLO inference
        results = model.predict(source=frame, verbose=False, conf=args.conf)
        counts = Counter()
        names = model.names if hasattr(model, "names") else {}

        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                if p < args.conf or area < min_area:
                    continue
                label = names.get(c, str(c))
                counts[label] += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {p:.2f}",
                            (int(x1), max(0, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Limit to top-k
        if len(counts) > args.topk:
            counts = Counter(dict(counts.most_common(args.topk)))

        # Scene hint
        scene_hint = args.scene.strip() or None
        if places:
            if frame_idx % max(1, args.scene_every) == 0:
                top = predict_scene_bgr(
                    frame,
                    places["model"],
                    places["classes"],
                    places["tfm"],
                    places["device"],
                    topk=1,
                )[0]
                places["cache"] = top
            label, prob = places["cache"]
            if label:
                scene_hint = label.replace("_", " ")

        # Build caption
        caption = make_caption(counts, scene_hint=scene_hint)

        # Caption overlay
        overlay = frame.copy()
        bar_h = 40
        cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, caption, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show / save
        if args.show:
            cv2.imshow("YOLOv11 Scene Caption", frame)
        if writer is not None:
            writer.write(frame)

        # ---------- TTS timing logic (with min_gap) ----------
        now = time.time()
        dt = now - last_speak_time

        # Timer-based: always allowed when speak_every reached
        speak_due_to_timer = (args.speak_every > 0 and dt >= args.speak_every)

        # Change-based: only if caption changed AND we've respected min_gap
        speak_due_to_change = (
            args.speak_on_change
            and caption != last_caption
            and dt >= args.min_gap
        )

        if (speak_due_to_timer or speak_due_to_change) and args.tts != "none":
            print(caption)
            last_caption = caption
            last_speak_time = now

            if tts_queue is not None:
                try:
                    tts_queue.put_nowait(caption)
                except Exception as e:
                    print(f"[WARN] TTS queue error: {e}")
        # --------------------------------------------------- #

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("r"), ord("R")):
            # Force next loop to speak (still respects min_gap via logic above)
            last_speak_time = 0.0

    # -------------------- Cleanup -------------------- #
    if tts_engine:
        if tts_queue is not None:
            tts_queue.put(None)
        if tts_thread is not None:
            tts_thread.join(timeout=1.0)
        tts_engine.stop()

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
