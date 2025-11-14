# ambi_youtube_bg_preview_youtubeapi_with_camera_test_v2.py
import os
import time
import random
import threading
import cv2
import numpy as np
import yt_dlp
import vlc
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from ytmusicapi import YTMusic

# ---------------- CONFIG ----------------
CAMERA_SOURCE = 0
CAPTURE_INTERVAL = 5.0
TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)
# Optional: path to yt-dlp cookies file (set via env var `YTDLP_COOKIES`) or place `cookies.txt` next to script
YTDLP_COOKIES = os.environ.get('YTDLP_COOKIES', 'cookies.txt')
# When set to '1' skip playlist extraction and use song search fallback directly
FORCE_SONG_SEARCH = os.environ.get('YTDLP_FORCE_SONG', '0') == '1'
# Allow playlist extraction (default: disabled to avoid noisy errors). Set to '1' to enable.
YTDLP_ALLOW_PLAYLISTS = os.environ.get('YTDLP_ALLOW_PLAYLISTS', '0') == '1'

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
    # Four brackets: kids, young, adult, senior
    if age <= 12:
        return "kids"
    elif age <= 25:
        return "young"
    elif age <= 60:
        return "adult"
    else:
        return "senior"

# ---------------- Playlist Fetch (fixed URLs) ----------------
def fetch_song_url_for_group(group):
    # Search keywords per age group
    queries = {
        "kids": "Kids Music",
        "young": "Pop Music",
        "adult": "Hindi Music",
        "senior": "Hindi Classic",
    }

    query = queries.get(group, "Popular Music")

    # 1) Try song search first (quiet, reliable)
    try:
        ytmusic = YTMusic()
        song_results = ytmusic.search(query, filter="songs")
        if song_results:
            song = random.choice(song_results[:10])
            video_id = song.get('videoId') or song.get('id') or song.get('browseId')
            title = song.get('title', 'Unknown Title')
            artists = ", ".join([a.get('name') for a in song.get('artists', [])]) if song.get('artists') else song.get('artist') or 'Unknown Artist'
            if video_id:
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                return video_url, title, artists
    except Exception as e:
        print("[WARN] Song search failed:", e)

    # 2) If song search failed and playlists are allowed, try playlists (limited attempts)
    if not YTDLP_ALLOW_PLAYLISTS or FORCE_SONG_SEARCH:
        if FORCE_SONG_SEARCH:
            print("[INFO] FORCE_SONG_SEARCH enabled — skipping playlist extraction.")
        else:
            print("[INFO] Playlist extraction disabled (enable with YTDLP_ALLOW_PLAYLISTS=1).")
        return None, None, None

    try:
        ytmusic = YTMusic()
        results = ytmusic.search(query, filter="playlists")
        if not results:
            print(f"[WARN] No playlist search results for '{query}'")
            return None, None, None

        featured = [r for r in results if 'featured' in r.get('title','').lower()]
        candidates = featured or results

        # Try fewer candidates to avoid many failing requests
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extractor_args": {"youtube": {"player_client": "default"}}}
        if os.path.exists(YTDLP_COOKIES):
            ydl_opts['cookiefile'] = YTDLP_COOKIES
            print(f"[INFO] Using yt-dlp cookies file: {YTDLP_COOKIES}")

        tried = 0
        for choice in candidates[:3]:
            tried += 1
            playlist_id = choice.get('playlistId') or choice.get('browseId')
            if not playlist_id:
                continue
            playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
            print(f"[DEBUG] Trying playlist ({tried}): {playlist_url}")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(playlist_url, download=False)

                entries = info.get('entries') or []
                entries = [e for e in entries if e]
                if not entries:
                    print(f"[WARN] No entries in playlist {playlist_id}")
                    continue

                random.shuffle(entries)
                entry = entries[0]
                video_id = entry.get('id') or entry.get('url')
                if not video_id:
                    print("[WARN] Couldn't determine video id from playlist entry")
                    continue

                video_url = f"https://www.youtube.com/watch?v={video_id}"
                title = entry.get('title', 'Unknown Title')
                artist = entry.get('uploader') or entry.get('artist') or 'Unknown Artist'
                return video_url, title, artist

            except Exception as e:
                print(f"[WARN] Failed to extract playlist {playlist_id}: {e}")
                continue

        print("[WARN] Playlist candidates exhausted; no playable entry found.")
        return None, None, None

    except Exception as e:
        print("[ERROR] Playlist search/extract failed:", e)
        return None, None, None

# ---------------- Play Music with Preview ----------------
def play_music_with_preview(avg_age, group):
    # Implement retry logic: re-fetch and restart playback up to N attempts
    max_retries = int(os.environ.get('YTDLP_PLAYBACK_RETRIES', '3'))

    for attempt in range(1, max_retries + 1):
        try:
            song_url, title, artist = fetch_song_url_for_group(group)
            if not song_url:
                print(f"[WARN] No song URL found for group '{group}' (attempt {attempt}).")
                return

            # Use default youtube player client to reduce extractor warnings
            ydl_opts2 = {
                'quiet': True,
                'format': 'bestaudio/best',
                'extractor_args': {'youtube': {'player_client': 'default'}}
            }

            with yt_dlp.YoutubeDL(ydl_opts2) as ydl2:
                info2 = ydl2.extract_info(song_url, download=False)

            # Try direct url first, then fall back to best audio format
            stream_url = info2.get('url')
            duration = info2.get('duration') or 0

            if not stream_url:
                formats = info2.get('formats') or []
                audio_formats = [f for f in formats if f.get('url') and f.get('acodec') and f.get('acodec') != 'none']
                if audio_formats:
                    audio_formats.sort(key=lambda f: ((f.get('abr') or 0) + (f.get('tbr') or 0)), reverse=True)
                    stream_url = audio_formats[0].get('url')

            if not stream_url:
                print(f"[WARN] Attempt {attempt}: could not fetch stream URL from yt-dlp info. Retrying...")
                continue

            instance = vlc.Instance("--no-xlib --quiet --intf dummy")
            player = instance.media_player_new()
            media = instance.media_new(stream_url)
            player.set_media(media)
            player.play()

            # small delay to allow VLC to buffer and start
            started = False
            start_wait = time.time()
            start_timeout = 6.0
            while time.time() - start_wait < start_timeout:
                st = player.get_state()
                # Playing or buffering states considered success
                if st in [vlc.State.Playing, vlc.State.Buffering]:
                    started = True
                    break
                if st in [vlc.State.Error, vlc.State.Ended, vlc.State.Stopped]:
                    break
                time.sleep(0.2)

            if not started:
                try:
                    player.stop()
                except Exception:
                    pass
                print(f"[WARN] Playback did not start (attempt {attempt}). Retrying...")
                continue

            # set sensible default volume
            try:
                v = player.audio_get_volume() or 80
                player.audio_set_volume(v)
            except Exception:
                pass

            # --- Display Preview Window with Controls ---
            stop_event = threading.Event()

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
                    img = np.zeros((150, 600, 3), dtype=np.uint8)
                    cv2.putText(img, f"Now Playing:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.putText(img, f"{title[:50]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.putText(img, f"by {artist[:50]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

                    vol = 0
                    try:
                        vol = player.audio_get_volume()
                    except Exception:
                        pass
                    cv2.putText(img, f"[Space]Play/Pause  [n]Next  [+/-]Vol ({vol})  [q]Close", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

                    if duration > 0:
                        bar_x = int(progress * 560)
                    else:
                        bar_x = int(((time.time() - start_time) % 1.0) * 560)
                    cv2.rectangle(img, (20,130), (580,140), (60,60,60), -1)
                    cv2.rectangle(img, (20,130), (20+bar_x,140), (0,255,0), -1)
                    cv2.imshow("Now Playing 🎵", img)

                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        stop_event.set()
                        break
                    elif key == ord('n'):
                        stop_event.set()
                        break
                    elif key == 32:  # space -> toggle pause/play
                        try:
                            if player.get_state() == vlc.State.Playing:
                                player.pause()
                            else:
                                player.play()
                        except Exception:
                            pass
                    elif key in (ord('+'), ord('=')):
                        try:
                            newv = min(100, (player.audio_get_volume() or 50) + 5)
                            player.audio_set_volume(newv)
                        except Exception:
                            pass
                    elif key == ord('-'):
                        try:
                            newv = max(0, (player.audio_get_volume() or 50) - 5)
                            player.audio_set_volume(newv)
                        except Exception:
                            pass

                cv2.destroyWindow("Now Playing 🎵")

            threading.Thread(target=show_preview, daemon=True).start()

            # Wait until ~95% of song done or user requested stop.
            if duration > 0:
                end_time = duration * 0.95
                start = time.time()
                while time.time() - start < end_time:
                    if stop_event.is_set():
                        break
                    time.sleep(0.2)
            else:
                while True:
                    if stop_event.is_set():
                        break
                    st = player.get_state()
                    if st in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                        break
                    time.sleep(0.5)

            try:
                player.stop()
            except Exception:
                pass
            print("[INFO] Song finished — resuming detection...")
            return

        except Exception as e:
            print(f"[ERROR] Playback attempt {attempt} failed: {e}")
            # slight backoff before retry
            time.sleep(0.8)
            continue

    print(f"[ERROR] All {max_retries} playback attempts failed — skipping music this cycle.")

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
