# app.py — CardBlur (Streamlit + YOLO, no OCR, Render friendly)
import os, io
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import streamlit as st

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"
WEIGHTS_PATH = Path(os.environ.get("WEIGHTS_PATH", str(DEFAULT_WEIGHTS))).resolve()

IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF = float(os.environ.get("CONF", 0.28))
IOU  = float(os.environ.get("IOU", 0.5))

# Upload-only settings (higher recall)
UPLOAD_CONF   = float(os.environ.get("UPLOAD_CONF", 0.15))
UPLOAD_IOU    = float(os.environ.get("UPLOAD_IOU", 0.5))
TILE_SIZE     = int(os.environ.get("TILE_SIZE", 960))
TILE_OVERLAP  = float(os.environ.get("TILE_OVERLAP", 0.20))

# Text post-process (geometry-based)
TEXT_DILATE_FRAC    = float(os.environ.get("TEXT_DILATE_FRAC", 0.010))
TEXT_MERGE_GAP_FRAC = float(os.environ.get("TEXT_MERGE_GAP_FRAC", 0.010))
TEXT_MAX_DOC_FRAC   = float(os.environ.get("TEXT_MAX_DOC_FRAC", 0.50))
TEXT_MIN_H_FRAC     = float(os.environ.get("TEXT_MIN_H_FRAC", 0.012))
TEXT_MAX_H_FRAC     = float(os.environ.get("TEXT_MAX_H_FRAC", 0.28))
TEXT_MIN_AR         = float(os.environ.get("TEXT_MIN_AR", 2.3))
TEXT_MAX_AR         = float(os.environ.get("TEXT_MAX_AR", 40.0))
TEXT_NMS_IOU        = float(os.environ.get("TEXT_NMS_IOU", 0.35))

# Blur strength
MIN_KERNEL   = int(os.environ.get("MIN_KERNEL", 31))
KERNEL_SCALE = float(os.environ.get("KERNEL_SCALE", 0.22))

FORCE_DEVICE = os.environ.get("FORCE_DEVICE", "").strip().lower()  # "", "cpu", "cuda", "mps"

# Labels (match your model)
DOC_LABELS  = {"id", "id_card", "idcard", "passport", "mrz", "serial", "number", "document", "card", "passport_id", "name", "dob", "expiry"}
FACE_LABELS = {"face", "person_face", "head"}
TEXT_LABELS = {"text"}

# Anti-flicker for doc blur
DOC_PAD_FRAC = float(os.environ.get("DOC_PAD_FRAC", 0.08))

# =========================
# Helpers
# =========================
def _lazy_import():
    from ultralytics import YOLO as _YOLO
    import torch as _torch
    return _YOLO, _torch

def pick_device(_torch):
    if FORCE_DEVICE in ("cpu", "cuda", "mps"): return FORCE_DEVICE
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available(): return "mps"
    if _torch.cuda.is_available(): return "cuda"
    return "cpu"

def make_odd(n): return n if n % 2 == 1 else n + 1

def compute_kernel(w, h):
    k = int(max(w, h) * KERNEL_SCALE)
    k = max(k, MIN_KERNEL)
    return make_odd(k)

def blur_region(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2));     y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1: return
    k = compute_kernel(x2 - x1, y2 - y1)
    if k < 3: k = 3
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)

def center(x1, y1, x2, y2): return ((x1 + x2) // 2, (y1 + y2) // 2)

def contains(box, pt):
    x1, y1, x2, y2 = box; x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def pad_box(box, pad_frac, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(W, x2 + px); ny2 = min(H, y2 + py)
    return (nx1, ny1, nx2, ny2)

def expand_px(box, px=4, py=4, W=99999, H=99999):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)

def box_area(b):
    x1,y1,x2,y2=b
    return max(0,x2-x1)*max(0,y2-y1)

def merge_boxes_overlap_or_near(boxes, max_gap_px):
    if not boxes: return []
    boxes = boxes[:]
    merged = []
    used = [False]*len(boxes)

    def near_or_overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
            return True
        h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
        v_overlap = min(ay2, by2) - max(ay1, by1)
        if h_gap <= max_gap_px and v_overlap > 0: return True
        v_gap = max(0, max(by1 - ay2, ay1 - by2))
        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        if v_gap <= max_gap_px and h_overlap > 0: return True
        return False

    for i in range(len(boxes)):
        if used[i]: continue
        cur = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                if near_or_overlap(cur, boxes[j]):
                    x1 = min(cur[0], boxes[j][0]); y1 = min(cur[1], boxes[j][1])
                    x2 = max(cur[2], boxes[j][2]); y2 = max(cur[3], boxes[j][3])
                    cur = (x1,y1,x2,y2)
                    used[j] = True
                    changed = True
        merged.append(cur)
    return merged

def nms_boxes(boxes, scores=None, iou_th=0.5):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.ones(len(boxes), dtype=float) if scores is None else np.array(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ub = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (ua + ub - inter + 1e-6)
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        suppress = []
        for j in rest:
            if _iou(boxes[i], boxes[int(j)]) > iou_th:
                suppress.append(j)
        order = np.array([k for k in rest if k not in suppress])
    return keep

# =========================
# Model wrapper (YOLO only, no OCR)
# =========================
class ImageProcessor:
    def __init__(self):
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")
        YOLO, torch = _lazy_import()
        self.device = pick_device(torch)
        self.model = YOLO(str(WEIGHTS_PATH))
        try: self.model.fuse()
        except Exception: pass
        self.names = getattr(self.model, "names", {})

    def _predict(self, img_rgb, conf, iou):
        try:
            results = self.model.predict(img_rgb, imgsz=IMG_SIZE, conf=conf, iou=iou, device=self.device, verbose=False)
        except Exception as e:
            st.warning(f"Inference error: {e}")
            return []
        out = []
        if not results: return out
        res = results[0]
        def npint(x): return x.cpu().numpy().astype(int)
        def npfloat(x): return x.cpu().numpy().astype(float)
        try:
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                xyxy = npint(res.boxes.xyxy)
                cls  = res.boxes.cls.int().tolist() if hasattr(res.boxes, "cls") else [None]*len(xyxy)
                scr  = npfloat(res.boxes.conf).tolist() if hasattr(res.boxes, "conf") else [1.0]*len(xyxy)
                for coords, c, s in zip(xyxy.tolist(), cls, scr):
                    label = self.names.get(int(c), str(c))
                    out.append((tuple(coords), (label or "").lower(), float(s)))
        except Exception:
            pass
        try:
            if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
                xyxy = npint(res.obb.xyxy)
                cls  = res.obb.cls.int().tolist() if hasattr(res.obb, "cls") else [None]*len(xyxy)
                scr  = npfloat(res.obb.conf).tolist() if hasattr(res.obb, "conf") else [1.0]*len(xyxy)
                for coords, c, s in zip(xyxy.tolist(), cls, scr):
                    label = self.names.get(int(c), str(c))
                    out.append((tuple(coords), (label or "").lower(), float(s)))
        except Exception:
            pass
        return out

    def _predict_tiled(self, img_rgb, conf, iou):
        H, W = img_rgb.shape[:2]
        ts = min(TILE_SIZE, max(H, W))
        step_x = int(ts * (1 - TILE_OVERLAP))
        step_y = int(ts * (1 - TILE_OVERLAP))
        if step_x <= 0 or step_y <= 0:
            return self._predict(img_rgb, conf, iou)
        out = []
        for y in range(0, H, step_y):
            for x in range(0, W, step_x):
                x2 = min(x + ts, W)
                y2 = min(y + ts, H)
                tile = img_rgb[y:y2, x:x2]
                preds = self._predict(tile, conf, iou)
                for (bx1, by1, bx2, by2), label, score in preds:
                    out.append(((bx1 + x, by1 + y, bx2 + x, by2 + y), label, score))
        return out

    def _predict_tta_flip(self, img_rgb, conf, iou):
        H, W = img_rgb.shape[:2]
        flip = cv2.flip(img_rgb, 1)
        preds = self._predict(flip, conf, iou)
        mapped = []
        for (x1, y1, x2, y2), label, score in preds:
            nx1 = W - x2
            nx2 = W - x1
            mapped.append(((nx1, y1, nx2, y2), label, score))
        return mapped

    def _gather_boxes(self, preds):
        doc_boxes, face_boxes, face_scores, text_boxes, text_scores = [], [], [], [], []
        for (x1,y1,x2,y2), label, score in preds:
            if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
            elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2)); face_scores.append(score)
            elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2)); text_scores.append(score)
        if face_boxes:
            keep = nms_boxes(face_boxes, face_scores, iou_th=0.45)
            face_boxes = [face_boxes[i] for i in keep]
        if text_boxes:
            keep = nms_boxes(text_boxes, None, iou_th=TEXT_NMS_IOU)
            text_boxes = [text_boxes[i] for i in keep]
        return doc_boxes, face_boxes, text_boxes

    def _filter_text_geometry(self, text_boxes, W, H):
        out = []
        for (x1,y1,x2,y2) in text_boxes:
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            h_frac = h / H; ar = w / h
            if h_frac < TEXT_MIN_H_FRAC: continue
            if h_frac > TEXT_MAX_H_FRAC: continue
            if ar < TEXT_MIN_AR or ar > TEXT_MAX_AR:
                if not (h_frac < (TEXT_MIN_H_FRAC*1.5) and 1.2 <= ar <= 2.5):
                    continue
            out.append((x1,y1,x2,y2))
        return out

    def _group_by_rows(self, boxes, row_tol_px):
        if not boxes: return []
        ys = [ ( (b[1]+b[3])//2, i ) for i,b in enumerate(boxes) ]
        ys.sort()
        groups, cur, last_y = [], [], None
        for yc, i in ys:
            if last_y is None or abs(yc - last_y) <= row_tol_px:
                cur.append(i)
            else:
                groups.append(cur); cur=[i]
            last_y = yc
        if cur: groups.append(cur)
        return [[boxes[i] for i in g] for g in groups]

    def _conservative_text(self, text_boxes, doc_boxes, W, H):
        if not text_boxes: return []
        text_boxes = self._filter_text_geometry(text_boxes, W, H)
        if not text_boxes: return []

        px = max(2, int(TEXT_DILATE_FRAC * W))
        py = max(2, int(TEXT_DILATE_FRAC * H))
        dil = [expand_px(t, px=px, py=py, W=W, H=H) for t in text_boxes]

        row_tol = max(3, int(0.012 * H))
        row_groups = self._group_by_rows(dil, row_tol_px=row_tol)

        gap = int(TEXT_MERGE_GAP_FRAC * max(W, H))
        merged_all = []
        for grp in row_groups:
            merged_all.extend(merge_boxes_overlap_or_near(grp, max_gap_px=gap))

        if doc_boxes and merged_all:
            safe = []
            for m in merged_all:
                cx, cy = center(*m)
                chosen = None
                for d in doc_boxes:
                    if contains(d, (cx, cy)):
                        chosen = d; break
                if chosen is None and doc_boxes:
                    ious = [iou(m, d) for d in doc_boxes]
                    chosen = doc_boxes[int(np.argmax(ious))] if any(ious) else None
                if chosen and box_area(m) > TEXT_MAX_DOC_FRAC * box_area(chosen):
                    inside = [b for b in dil if contains(chosen, center(*b))]
                    safe.extend(merge_boxes_overlap_or_near(inside, max_gap_px=gap))
                else:
                    safe.append(m)
            merged_all = safe

        if merged_all:
            keep = nms_boxes(merged_all, None, iou_th=0.4)
            merged_all = [merged_all[i] for i in keep]

        return merged_all

    def run_upload(self, bgr_image: np.ndarray, mode: str = "both") -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        preds = []
        preds += self._predict(rgb, UPLOAD_CONF, UPLOAD_IOU)
        preds += self._predict_tiled(rgb, UPLOAD_CONF, UPLOAD_IOU)
        preds += self._predict_tta_flip(rgb, UPLOAD_CONF, UPLOAD_IOU)

        doc_boxes, face_boxes, text_boxes = self._gather_boxes(preds)
        H, W = bgr_image.shape[:2]
        if doc_boxes:
            text_boxes = [t for t in text_boxes if any(contains(d, center(*t)) for d in doc_boxes)]
            face_boxes = [f for f in face_boxes if any(contains(d, center(*f)) for d in doc_boxes)]
        text_boxes = self._conservative_text(text_boxes, doc_boxes, W, H)

        m = (mode or "both").lower()
        if m == "text":
            targets = text_boxes
        elif m == "face":
            targets = face_boxes
        elif m == "both":
            merged = text_boxes + face_boxes
            targets = [merged[i] for i in nms_boxes(merged, None, iou_th=0.3)] if merged else []
        elif m == "doc":
            targets = [pad_box(d, DOC_PAD_FRAC, W, H) for d in doc_boxes]
        else:
            targets = []

        for (x1, y1, x2, y2) in targets:
            blur_region(bgr_image, x1, y1, x2, y2)
        return bgr_image

    def run_live_doc_only(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        preds = self._predict(rgb, CONF, IOU)

        doc_boxes, face_boxes, text_boxes = [], [], []
        for (x1,y1,x2,y2), label, _score in preds:
            if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
            elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2))
            elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2))

        H, W = bgr_image.shape[:2]
        blur_regions = []
        if doc_boxes:
            for d in doc_boxes:
                faces_in = [f for f in face_boxes if contains(d, center(*f))]
                texts_in = [t for t in text_boxes if contains(d, center(*t))]
                if faces_in and texts_in:
                    blur_regions.append(pad_box(d, DOC_PAD_FRAC, W, H))

        for (x1, y1, x2, y2) in merge_boxes_overlap_or_near(blur_regions, max_gap_px=0):
            blur_region(bgr_image, x1, y1, x2, y2)
        return bgr_image

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CardBlur", page_icon="🪪", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>CardBlur</h1>"
    "<p style='color:#888;margin-top:4px'>AI-powered privacy protection for IDs and passports</p>",
    unsafe_allow_html=True,
)

# Lazy-load model into session_state
if "uploader" not in st.session_state:
    try:
        st.session_state.uploader = ImageProcessor()
        st.success(f"Model loaded from {WEIGHTS_PATH}")
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        st.stop()

mode_map = {
    "Blur TEXT only": "text",
    "Blur FACE only": "face",
    "Blur BOTH": "both",
    "Blur WHOLE CARD": "doc",
}
mode = st.radio("Select blur mode for processing:", list(mode_map.keys()), horizontal=True)
mode_key = mode_map[mode]

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera Snapshot"])

# ---- Upload tab ----
with tab1:
    colL, colR = st.columns([1, 1])
    with colL:
        up = st.file_uploader("Choose an image", type=["jpg","jpeg","png","bmp","webp"])
        process_btn = st.button("Process Upload", use_container_width=True, type="primary", disabled=(up is None))
        dl_placeholder = st.empty()
    with colR:
        out_slot = st.empty()

    if process_btn and up is not None:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read image.")
        else:
            proc = st.session_state.uploader.run_upload(bgr.copy(), mode=mode_key)
            # Show result
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            out_slot.image(rgb, caption="Blurred result", use_column_width=True)
            # Download
            ok, buf = cv2.imencode(".jpg", proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                dl_placeholder.download_button(
                    "⬇ Download Blurred Image",
                    data=buf.tobytes(),
                    file_name="cardblur_result.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

# ---- Camera tab ----
with tab2:
    st.write("Click **Take Photo** to capture a frame from your camera (HTTPS required).")
    cam_img = st.camera_input("Camera", label_visibility="collapsed")
    left, right = st.columns([1,1])
    with left:
        if cam_img is not None:
            st.image(cam_img, caption="Original capture", use_column_width=True)
    with right:
        if cam_img is not None:
            data = np.frombuffer(cam_img.getvalue(), np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Could not read camera image.")
            else:
                # For camera, you can choose to use the same mode or the doc-only rule.
                # Here we respect the same mode selection.
                proc = st.session_state.uploader.run_upload(bgr.copy(), mode=mode_key)
                rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"Processed ({mode})", use_column_width=True)
                ok, buf = cv2.imencode(".jpg", proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    st.download_button(
                        "⬇ Download Blurred Snapshot",
                        data=buf.tobytes(),
                        file_name="cardblur_camera.jpg",
                        mime="image/jpeg",
                        use_container_width=True,
                    )

st.markdown(
    "<div style='text-align:center;color:#888;margin-top:16px;'>"
    "Shatha Khawaji • Renad Almutairi • Jury Alsultan • Yara Alsardi<br/>"
    "<span style='font-size:12px'>Built as our AI capstone project</span>"
    "</div>",
    unsafe_allow_html=True
)
