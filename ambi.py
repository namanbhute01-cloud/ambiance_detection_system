# ambi_youtube_bg_preview_youtubeapi_with_camera_test_v2.py
import os
import time
import random
import threading
import cv2
import numpy as np
import yt_dlp
import vlc
from ytmusicapi import YTMusic

# ---------------- CONFIG ----------------
CAMERA_SOURCE = 0
CAPTURE_INTERVAL = 5.0
TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO  = "deploy_age.prototxt"
AGE_MODEL  = "age_net.caffemodel"

AGE_BUCKETS    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MIDPOINTS  = np.array([1,5,10,18,28,40,50,70], dtype=np.float32)

# ---------------- Model Helpers ----------------
def check_models():
    missing = [p for p in (FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL) if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing model files:", missing)
        raise SystemExit(1)

def load_nets():
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    age_net  = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    return face_net, age_net

def detect_faces_dnn(net, frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf > conf_threshold:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2,conf))
    return boxes

# ---------------- Improved Age Prediction ----------------
def predict_age(age_net, face_img):
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, (227,227))
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227),
                                 (78.4263377603,87.7689143744,114.895847746),
                                 swapRB=False, crop=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]

    # Apply softmax for stability
    probs = np.exp(preds - np.max(preds))
    probs /= probs.sum()

    expected_age = float((probs * AGE_MIDPOINTS).sum())

    # 🔹 Add slight correction bias for more realistic estimation
    correction_bias = np.interp(expected_age, [0, 70], [1.2, 1.05])
    expected_age *= correction_bias
    expected_age = np.clip(expected_age, 0, 100)

    return expected_age

def cleanup_temp():
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, f))
        except:
            pass

# ---------------- Age Group Helper ----------------
def age_to_group(age):
    if age <= 12:
        return "kids"
    elif age <= 25:
        return "youth"
    elif age <= 40:
        return "adult"
    elif age <= 60:
        return "mid"
    else:
        return "senior"

# ---------------- YTMusic Dynamic Fetch ----------------
ytmusic = YTMusic()

def fetch_song_url_for_group(group):
    queries = {
        "kids": "popular kids songs",
        "youth": "latest pop hits",
        "adult": "relaxing instrumental music",
        "mid": "soft rock or classic love songs",
        "senior": "retro old hindi songs"
    }
    query = queries.get(group, "popular songs")

    results = ytmusic.search(query, filter="songs")
    if not results:
        print(f"[WARN] No YTMusic results for '{query}'")
        return None, None, None

    song = random.choice(results[:5])  # pick one of top 5
    title = song.get("title", "Unknown Title")
    video_id = song.get("videoId")
    artists = ", ".join([a["name"] for a in song.get("artists", [])])
    url = f"https://www.youtube.com/watch?v={video_id}"
    return url, title, artists

# ---------------- Play Music with Preview ----------------
def play_music_with_preview(avg_age, group):
    try:
        song_url, title, artist = fetch_song_url_for_group(group)
        if not song_url:
            return

        ydl_opts2 = {'quiet': True, 'format': 'bestaudio/best'}
        with yt_dlp.YoutubeDL(ydl_opts2) as ydl2:
            info2 = ydl2.extract_info(song_url, download=False)
            stream_url = info2.get('url')
            duration = info2.get('duration', 0)

        if not stream_url:
            print("[WARN] Could not fetch stream.")
            return

        instance = vlc.Instance("--no-xlib --quiet --intf dummy")
        player = instance.media_player_new()
        media = instance.media_new(stream_url)
        player.set_media(media)
        player.play()

        # --- Display Preview Window ---
        def show_preview():
            start_time = time.time()
            cv2.namedWindow("Now Playing 🎵", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Now Playing 🎵", 1000, 40)
            while True:
                state = player.get_state()
                if state in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                    break
                elapsed = time.time() - start_time
                progress = min(1.0, elapsed / duration) if duration > 0 else 0
                img = np.zeros((130, 500, 3), dtype=np.uint8)
                cv2.putText(img, f"Now Playing:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(img, f"{title[:40]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(img, f"by {artist[:40]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                bar_x = int(progress * 460)
                cv2.rectangle(img, (20,110), (480,120), (60,60,60), -1)
                cv2.rectangle(img, (20,110), (20+bar_x,120), (0,255,0), -1)
                cv2.imshow("Now Playing 🎵", img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow("Now Playing 🎵")

        threading.Thread(target=show_preview, daemon=True).start()

        # Wait until ~95% of song done
        if duration > 0:
            time.sleep(duration * 0.95)
        player.stop()
        print("[INFO] Song finished — resuming detection...")

    except Exception as e:
        print("[ERROR]", e)

# ---------------- MAIN ----------------
def main():
    print("[INFO] Starting ambience system with improved age detection 🎵")
    check_models()
    face_net, age_net = load_nets()
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    print("[TEST] Live camera preview enabled — press 'q' to quit the preview.")

    last_capture = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame. Retrying...")
            time.sleep(1)
            continue

        # --- Live camera preview for testing ---
        preview_frame = frame.copy()
        boxes = detect_faces_dnn(face_net, frame)
        for (x1,y1,x2,y2,_) in boxes:
            cv2.rectangle(preview_frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.imshow("Camera Feed (Testing)", preview_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Preview window closed by user.")
            break

        # --- Detection and music logic ---
        now = time.time()
        if now - last_capture >= CAPTURE_INTERVAL:
            last_capture = now
            print(f"[STEP] {len(boxes)} faces detected.")
            if not boxes:
                cleanup_temp()
                continue

            ages = []
            cleanup_temp()
            timestamp = int(time.time())

            for i, (x1,y1,x2,y2,conf) in enumerate(boxes):
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_path = os.path.join(TEMP_DIR, f"face_{timestamp}_{i}.jpg")
                cv2.imwrite(face_path, face_img)
                print(f"[INFO] Saved face: {face_path}")
                age_pred = predict_age(age_net, face_img)
                if age_pred is not None:
                    ages.append(age_pred)

            if not ages:
                print("[WARN] No ages predicted.")
                continue

            avg_age = float(np.mean(ages))
            group = age_to_group(avg_age)
            print(f"[RESULT] Avg age: {avg_age:.1f} → Group: {group}")

            play_music_with_preview(avg_age, group)

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
