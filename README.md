# Ambiance Detection System 🎵

## Project Concept

The **Ambiance Detection System** is an intelligent, real-time music recommendation system that automatically selects and plays age-appropriate music based on live camera face detection and age estimation. The system continuously monitors a camera feed, detects faces, predicts the age of detected individuals, and dynamically curates music from YouTube Music tailored to the detected demographic.

### Core Idea
Create an ambient music experience that adapts to the audience in real-time—perfect for retail environments, waiting rooms, cafes, or smart home entertainment systems where the music should match the age profile of people present.

---

## System Architecture

```
┌─────────────────┐
│  Camera Feed    │
│   (OpenCV)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Face Detection  │
│   (DNN Model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Age Prediction  │
│ (Caffe Model)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Age Grouping    │
│ (4 Brackets)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Music Search    │
│  (YTMusic API)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Stream Extract  │
│   (yt-dlp)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Audio Playback  │
│  (VLC Player)   │
└─────────────────┘
```

---

## Project Components

### 1. **Face Detection Module**

#### Model Files
- **`deploy.prototxt`**: Network architecture definition (prototxt format)
- **`res10_300x300_ssd_iter_140000.caffemodel`**: Pre-trained Caffe model weights
- **`opencv_face_detector_uint8.pb`**: TensorFlow protobuf model (alternative format)

#### Function: `detect_faces_dnn(net, frame, conf_threshold=0.5)`
**Purpose**: Detects human faces in a video frame using a deep neural network.

**How it works**:
1. **Preprocessing**: Resizes input frame to 300×300 pixels
2. **Blob Creation**: Converts frame to a blob with mean subtraction `(104.0, 177.0, 123.0)` for BGR channels
3. **Forward Pass**: Runs the DNN model to get detection results
4. **Post-processing**: 
   - Filters detections by confidence threshold (default: 50%)
   - Converts normalized coordinates to pixel coordinates
   - Clamps coordinates to frame boundaries
5. **Output**: Returns list of bounding boxes `(x1, y1, x2, y2, confidence)`

**Technical Details**:
- Uses OpenCV's DNN module for hardware-accelerated inference
- Model: SSD (Single Shot MultiBox Detector) with ResNet-10 backbone
- Input size: 300×300×3
- Detection threshold: Configurable (default 0.5)

---

### 2. **Age Prediction Module**

#### Model Files
- **`deploy_age.prototxt`**: Age prediction network architecture
- **`age_net.caffemodel`**: Pre-trained age classification weights

#### Age Brackets & Midpoints
```python
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MIDPOINTS = [1, 5, 10, 18, 28, 40, 50, 70]
```

#### Function: `predict_age(age_net, face_img)`
**Purpose**: Estimates the age of a person from their face image.

**How it works**:
1. **Preprocessing**:
   - Resizes face image to 227×227 pixels
   - Creates blob with mean subtraction `(78.43, 87.77, 114.90)`
2. **Inference**: Runs the age classification model
3. **Softmax Normalization**: Stabilizes probability distribution across 8 age buckets
4. **Expected Value Calculation**: Computes weighted average using age midpoints
5. **Correction Bias**: Applies an interpolated correction factor (1.2 for young, 1.05 for elderly) to improve accuracy
6. **Output**: Returns predicted age as a float (0-100)

**Technical Details**:
- Architecture: Based on VGG-like CNN
- Input: 227×227×3 RGB image
- Output: 8-class probability distribution
- Post-processing: Softmax → expectation → bias correction → clipping

---

### 3. **Age-to-Group Mapping**

#### Function: `age_to_group(age)`
**Purpose**: Maps continuous age values to discrete demographic groups.

**Age Brackets**:
| Age Range | Group Label | Music Preference |
|-----------|-------------|------------------|
| 0-12 years | `kids` | Kids Music |
| 13-25 years | `young` | Pop Music |
| 26-60 years | `adult` | Hindi Music |
| 61+ years | `senior` | Hindi Classic |

**Logic**:
```python
if age <= 12:    return "kids"
elif age <= 25:  return "young"
elif age <= 60:  return "adult"
else:            return "senior"
```

---

### 4. **Music Fetching System**

#### Function: `fetch_song_url_for_group(group)`
**Purpose**: Searches and retrieves a music video URL appropriate for the detected age group.

**Two-Stage Strategy**:

##### Stage 1: Song Search (Primary)
1. **Query Mapping**: Maps age groups to search keywords
   - `kids` → "Kids Music"
   - `young` → "Pop Music"
   - `adult` → "Hindi Music"
   - `senior` → "Hindi Classic"
2. **YTMusic Search**: Uses `ytmusicapi` to search YouTube Music with `filter="songs"`
3. **Random Selection**: Picks a random song from top 10 results
4. **Metadata Extraction**: Returns `(video_url, title, artist)`

##### Stage 2: Playlist Search (Fallback)
*Only triggered if song search fails and `YTDLP_ALLOW_PLAYLISTS=1`*

1. **Playlist Search**: Searches for playlists matching the query
2. **Featured Priority**: Prefers playlists with "featured" in the title
3. **Limited Attempts**: Tries up to 3 playlist candidates
4. **Extraction**: Uses `yt-dlp` to extract playlist entries
5. **Random Selection**: Shuffles entries and picks one
6. **Cookie Support**: Uses cookies file if `YTDLP_COOKIES` is set

**Error Handling**:
- Graceful fallback from playlists to songs
- Logs each attempted playlist with debug messages
- Returns `(None, None, None)` if all methods fail

**Environment Variables**:
- `YTDLP_COOKIES`: Path to cookies file (improves access to region-restricted content)
- `YTDLP_FORCE_SONG`: Skip playlist extraction (set to `'1'`)
- `YTDLP_ALLOW_PLAYLISTS`: Enable playlist extraction (set to `'1'`)

---

### 5. **Music Playback System**

#### Function: `play_music_with_preview(avg_age, group)`
**Purpose**: Streams and plays music with a visual preview and interactive controls, featuring automatic retry on failure.

**Retry Mechanism**:
- Configurable via `YTDLP_PLAYBACK_RETRIES` (default: 3 attempts)
- Re-fetches song on each attempt (gets different candidates)
- Validates VLC player actually starts (6-second timeout)
- Implements exponential backoff (0.8s delay between retries)

**Playback Pipeline**:

##### Step 1: Stream URL Extraction
1. **yt-dlp Configuration**: 
   - Format: `bestaudio/best`
   - Player client: `default` (reduces extractor warnings)
   - Quiet mode enabled
2. **URL Priority**:
   - First: Try direct stream URL from `info['url']`
   - Fallback: Select best audio format from `info['formats']`
   - Sorting: By audio bitrate (abr) + total bitrate (tbr)
3. **Validation**: Retries if no stream URL found

##### Step 2: VLC Player Setup
1. **Instance Creation**: `vlc.Instance("--no-xlib --quiet --intf dummy")`
2. **Media Loading**: Creates media from stream URL
3. **Playback Start**: Calls `player.play()`
4. **Start Validation**: 
   - Waits up to 6 seconds
   - Checks for `Playing` or `Buffering` state
   - Retries if player enters `Error`, `Ended`, or `Stopped` state
5. **Volume Setup**: Sets default volume to 80

##### Step 3: Visual Preview Window (Threading)

**Thread: `show_preview()`**

Runs in a daemon thread to display a real-time music player interface:

**Window Layout**:
```
┌──────────────────────────────────────┐
│ Now Playing: 🎵                      │
│ Song Title (truncated to 50 chars)   │
│ by Artist Name                        │
│ [Space]Play/Pause [n]Next [+/-]Vol   │
│ ████████░░░░░░░░░░░░░░░░░░░░░░ (75%) │
└──────────────────────────────────────┘
```

**Interactive Controls**:
| Key | Action |
|-----|--------|
| `Space` | Toggle Play/Pause |
| `n` | Skip to next (triggers re-detection) |
| `+` or `=` | Increase volume by 5 |
| `-` | Decrease volume by 5 |
| `q` | Close preview and stop playback |

**Technical Implementation**:
- Window: OpenCV `WINDOW_NORMAL` positioned at (1000, 40)
- Canvas: 600×150 black background
- Progress Bar: 
  - Known duration: Shows actual progress (0-100%)
  - Unknown duration: Animated cycling bar
- Stop Event: Thread-safe signaling between preview and main wait loop

##### Step 4: Playback Wait Logic

**Duration-Based Wait** (if duration known):
- Plays until 95% of song duration
- Polls `stop_event` every 0.2s
- Allows early termination via user controls

**State-Based Wait** (if duration unknown):
- Polls player state every 0.5s
- Exits on `Ended`, `Stopped`, or `Error`
- Respects `stop_event` for user-triggered stop

##### Step 5: Cleanup
- Stops VLC player gracefully
- Logs completion message
- Returns to face detection loop

**Error Recovery**:
- Catches all exceptions per attempt
- Logs failure reason
- Implements 0.8s backoff before retry
- Falls back to re-fetching different song
- Skips music if all retries exhausted

---

### 6. **Main Control Loop**

#### Function: `main()`
**Purpose**: Orchestrates the entire system lifecycle.

**Initialization Phase**:
1. **Model Validation**: Checks all required model files exist
2. **Model Loading**: Loads face detection and age prediction networks
3. **Camera Setup**: Opens default camera (index 0)
4. **Camera Validation**: Exits if camera unavailable

**Main Event Loop**:

```python
while True:
    1. Read frame from camera
    2. Display live preview with face boxes
    3. Every CAPTURE_INTERVAL (5 seconds):
       a. Detect faces
       b. Save face crops to temp_faces/
       c. Predict age for each face
       d. Calculate average age
       e. Map to age group
       f. Fetch and play music
    4. Check for 'q' key to quit
```

**Timing Details**:
- **Frame Read**: Continuous (as fast as camera allows)
- **Face Detection**: Every frame (for live preview boxes)
- **Age Prediction**: Every 5 seconds
- **Music Playback**: Blocks until 95% of song plays (or user skips)

**Resource Management**:
- Temporary face images stored in `temp_faces/` directory
- Cleanup after each detection cycle
- Proper release of camera and OpenCV windows on exit

---

## Configuration & Environment Variables

### Camera Settings
```python
CAMERA_SOURCE = 0          # Camera index (0 = default)
CAPTURE_INTERVAL = 5.0     # Seconds between age detections
```

### File Paths
```python
TEMP_DIR = "temp_faces"    # Directory for face crops
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "deploy_age.prototxt"
AGE_MODEL = "age_net.caffemodel"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YTDLP_COOKIES` | `cookies.txt` | Path to browser cookies for yt-dlp (improves access) |
| `YTDLP_FORCE_SONG` | `0` | Set to `1` to skip playlist extraction |
| `YTDLP_ALLOW_PLAYLISTS` | `0` | Set to `1` to enable playlist extraction |
| `YTDLP_PLAYBACK_RETRIES` | `3` | Number of playback retry attempts |

### Example Usage
```powershell
# Windows PowerShell
$env:YTDLP_COOKIES='C:\path\to\cookies.txt'
$env:YTDLP_FORCE_SONG='1'
$env:YTDLP_PLAYBACK_RETRIES='5'
python ambi.py
```

```bash
# Linux/Mac
export YTDLP_COOKIES='/path/to/cookies.txt'
export YTDLP_FORCE_SONG='1'
export YTDLP_PLAYBACK_RETRIES='5'
python ambi.py
```

---

## Installation & Setup

### Prerequisites

1. **System Requirements**:
   - Python 3.8 or higher
   - Webcam or video capture device
   - VLC Media Player installed on system

2. **Install VLC**:
   - **Windows**: Download from [videolan.org](https://www.videolan.org/)
   - **Linux**: `sudo apt install vlc`
   - **Mac**: `brew install vlc`

### Installation Steps

1. **Clone Repository**:
```bash
git clone https://github.com/namanbhute01-cloud/ambiance_detection_system.git
cd ambiance_detection_system
```

2. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Model Files**:
   - Face detection models: `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`
   - Age prediction models: `deploy_age.prototxt`, `age_net.caffemodel`
   
   *(Models should be in the project root directory)*

4. **Optional: Export Browser Cookies**:
   - Install browser extension (e.g., "Get cookies.txt LOCALLY")
   - Export cookies from YouTube.com
   - Save as `cookies.txt` in project directory

### Running the System

**Basic Usage**:
```bash
python ambi.py
```

**With Custom Configuration**:
```powershell
# Force song-only mode (quieter, more reliable)
$env:YTDLP_FORCE_SONG='1'; python ambi.py

# Enable playlists with more retries
$env:YTDLP_ALLOW_PLAYLISTS='1'; $env:YTDLP_PLAYBACK_RETRIES='5'; python ambi.py

# Use cookies for better extraction
$env:YTDLP_COOKIES='C:\path\to\cookies.txt'; python ambi.py
```

**Controls**:
- **Camera Preview**: Press `q` to quit
- **Music Preview**: 
  - `Space`: Play/Pause
  - `n`: Next song
  - `+`/`-`: Volume control
  - `q`: Close and resume detection

---

## Dependencies

### Python Packages (`requirements.txt`)

```
opencv-python>=4.7.0        # Computer vision and GUI
numpy>=1.24.0               # Numerical computations
yt-dlp>=2023.12.1           # YouTube stream extraction
python-vlc>=3.0.0           # VLC Python bindings
ytmusicapi>=0.19.0          # YouTube Music API
```

### System Dependencies
- **VLC Media Player**: Required for audio playback
- **Webcam**: Any OpenCV-compatible camera

### Model Files (Not in Git)
The following large binary files must be downloaded separately:
- `age_net.caffemodel` (~23 MB) - Listed in `.gitignore`
- `res10_300x300_ssd_iter_140000.caffemodel` (~10 MB)
- `deploy.prototxt`, `deploy_age.prototxt` (text files)
- `opencv_face_detector_uint8.pb` (optional TensorFlow format)

---

## Technical Deep Dive

### Face Detection Pipeline

**Input**: BGR image frame (any resolution)  
**Output**: List of `(x1, y1, x2, y2, confidence)` tuples

**Process**:
1. **Resize**: Scale to 300×300 (network input size)
2. **Mean Subtraction**: BGR values `(104, 177, 123)` - ImageNet normalization
3. **Blob Format**: `[1, 3, 300, 300]` (batch, channels, height, width)
4. **DNN Forward Pass**: SSD network inference (~30ms on CPU)
5. **NMS (Non-Maximum Suppression)**: Implicit in SSD model
6. **Coordinate Scaling**: Denormalize from [0,1] to pixel coordinates
7. **Boundary Clipping**: Ensure boxes stay within frame

**Performance Optimization**:
- Uses OpenCV's optimized DNN backend
- Can leverage OpenCL/CUDA if available
- Single-pass detection (no sliding windows)

### Age Prediction Pipeline

**Input**: Face crop image (any size)  
**Output**: Age estimate (float, 0-100)

**Process**:
1. **Alignment**: Assumes frontal face (no explicit alignment)
2. **Resize**: Scale to 227×227 (network input size)
3. **Mean Subtraction**: BGR values `(78.43, 87.77, 114.90)` - VGG normalization
4. **Blob Format**: `[1, 3, 227, 227]`
5. **CNN Forward Pass**: VGG-like architecture (~50ms on CPU)
6. **Softmax**: Convert logits to probabilities
7. **Expectation**: `age = Σ(prob[i] × midpoint[i])`
8. **Bias Correction**: Linear interpolation from 1.2 (young) to 1.05 (old)

**Accuracy Improvements**:
- Softmax stabilization (subtract max before exp)
- Correction bias compensates for model's tendency to underestimate
- Clipping ensures output stays in [0, 100]

**Limitations**:
- Works best on frontal faces
- Lighting variations affect accuracy
- Cultural/ethnic biases in training data

### Music Selection Algorithm

**Strategy**: Multi-tiered fallback with configurable behavior

**Tier 1: Song Search** (Always tried first)
- **Pros**: Fast, quiet, reliable
- **Cons**: Less variety, no curated playlists
- **Implementation**: YTMusic API → random from top 10

**Tier 2: Playlist Extraction** (Optional, if Tier 1 fails)
- **Pros**: Curated playlists, better variety
- **Cons**: Slower, prone to extraction errors
- **Implementation**: 
  1. Search playlists on YouTube Music
  2. Prefer "featured" playlists
  3. Extract up to 3 candidates with yt-dlp
  4. Randomize entries for variety

**Extraction Robustness**:
- Cookie support for private/regional content
- Player client set to `default` (reduces API changes impact)
- No warnings mode to reduce stderr noise
- Limited attempts (3) to avoid flooding errors

### Stream Playback Architecture

**Components**:
1. **yt-dlp Extractor**: Fetches stream URL without downloading
2. **VLC Backend**: Handles actual audio decoding and playback
3. **OpenCV Frontend**: Visual preview and controls
4. **Threading Model**: Preview runs in daemon thread

**Why VLC Instead of Direct Audio Library?**
- Handles complex stream protocols (HLS, DASH)
- Built-in codec support
- Network buffering and error recovery
- Cross-platform consistency

**Threading Considerations**:
- Preview thread: Daemon (auto-terminates on main exit)
- Communication: `threading.Event` for stop signaling
- No shared state mutation (thread-safe)

**Retry Logic Design**:
- **Assumption**: Failures often transient (network issues, expired URLs)
- **Strategy**: Re-fetch gets different candidates (randomized selection)
- **Validation**: Explicit VLC state check (not just `.play()` call)
- **Backoff**: 0.8s delay prevents hammering APIs

---

## Workflow Example

### Scenario: Adult Walks Into Room

1. **t=0.0s**: Camera captures frame
2. **t=0.1s**: Face detected with 87% confidence
3. **t=0.2s**: Face crop saved to `temp_faces/face_1699900000_0.jpg`
4. **t=0.3s**: Age prediction runs → output: 34.2 years
5. **t=0.4s**: Age mapped to group: `"adult"`
6. **t=0.5s**: YTMusic search: `"Hindi Music"` filter="songs"
7. **t=1.2s**: Random song selected: `"Tum Hi Ho"` by Arijit Singh
8. **t=1.5s**: yt-dlp extracts stream URL
9. **t=2.0s**: VLC starts buffering
10. **t=2.5s**: Audio playback begins
11. **t=2.6s**: Preview window appears at (1000, 40)
12. **t=3.0s**: User hears music through speakers
13. **t=180s**: Song plays to 95% (171s of 180s)
14. **t=180.5s**: Player stops, window closes
15. **t=181s**: System returns to detection loop

### Scenario: Playback Failure with Retry

1. **Attempt 1**: Stream URL extraction fails (HTTP 400)
   - Log: `[WARN] Attempt 1: could not fetch stream URL. Retrying...`
   - Wait: 0.8s backoff
2. **Attempt 2**: VLC player doesn't start (timeout)
   - Log: `[WARN] Playback did not start (attempt 2). Retrying...`
   - Re-fetch: New song selected from search results
   - Wait: 0.8s backoff
3. **Attempt 3**: Success - playback starts normally
   - Log: `[INFO] Song finished — resuming detection...`

---

## Troubleshooting

### Common Issues

#### 1. **No audio playing**
**Symptoms**: Preview window appears but no sound  
**Causes**:
- VLC not installed or not in PATH
- Stream URL extraction failed
- Region-restricted content

**Solutions**:
```powershell
# Use cookies for better access
$env:YTDLP_COOKIES='path\to\cookies.txt'; python ambi.py

# Force song-only mode (more reliable)
$env:YTDLP_FORCE_SONG='1'; python ambi.py

# Increase retry attempts
$env:YTDLP_PLAYBACK_RETRIES='5'; python ambi.py
```

#### 2. **Many yt-dlp warnings**
**Symptoms**: Console flooded with HTTP 400/404 errors  
**Causes**:
- Playlist extraction attempts failing
- YouTube API changes
- No JavaScript runtime

**Solutions**:
```powershell
# Disable playlists
$env:YTDLP_ALLOW_PLAYLISTS='0'; python ambi.py

# Or force song-only
$env:YTDLP_FORCE_SONG='1'; python ambi.py

# Install Node.js for JS runtime
# Download from nodejs.org
```

#### 3. **Camera not opening**
**Symptoms**: `[ERROR] Unable to open camera`  
**Causes**:
- Wrong camera index
- Camera in use by another app
- Permission denied

**Solutions**:
```python
# Try different camera index in ambi.py
CAMERA_SOURCE = 1  # or 2, 3, etc.

# Check camera access
# Windows: Settings → Privacy → Camera
# Linux: Check /dev/video* permissions
```

#### 4. **Inaccurate age predictions**
**Symptoms**: Wrong age group, poor music selection  
**Causes**:
- Poor lighting
- Partial face occlusion
- Low-quality camera

**Solutions**:
- Improve lighting conditions
- Ensure frontal face view
- Increase `CAPTURE_INTERVAL` for more stable detection
- Use higher-quality camera

#### 5. **High CPU usage**
**Symptoms**: System lag, fan noise  
**Causes**:
- Continuous DNN inference
- High-resolution camera

**Solutions**:
```python
# Reduce frame processing in ambi.py
time.sleep(0.1)  # Increase from 0.05 to reduce FPS

# Lower camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

## Future Enhancements

### Potential Features
- [ ] Multi-person age averaging with weighted confidence
- [ ] Gender detection for refined music selection
- [ ] Emotion detection for mood-based playlists
- [ ] Spotify/Apple Music integration
- [ ] Web dashboard for remote monitoring
- [ ] Historical analytics and listening trends
- [ ] Custom music preference profiles
- [ ] Voice command integration
- [ ] Multiple camera support
- [ ] Cloud deployment (edge computing)

### Technical Improvements
- [ ] GPU acceleration for faster inference
- [ ] Face tracking for reduced re-detection
- [ ] Async music fetching (pre-fetch next song)
- [ ] Local music cache to reduce API calls
- [ ] Facial recognition for personalized playlists
- [ ] Docker containerization
- [ ] REST API for integration
- [ ] Unit tests and CI/CD pipeline

---

## Performance Metrics

### Typical Latency Breakdown
| Operation | Time (CPU) | Notes |
|-----------|------------|-------|
| Frame capture | 10-30ms | Depends on camera |
| Face detection | 20-50ms | OpenCV DNN |
| Age prediction | 30-80ms | Per face |
| Music search | 500-1500ms | Network-dependent |
| Stream extraction | 800-2000ms | yt-dlp overhead |
| VLC startup | 500-1000ms | Buffering time |
| **Total (cold start)** | **2-5 seconds** | First music play |

### Resource Usage
- **RAM**: ~200-400 MB (models + OpenCV + VLC)
- **CPU**: 10-30% on modern quad-core
- **Network**: ~3-5 MB per song metadata fetch
- **Disk**: Minimal (temp face images < 1 MB)

---

## License & Credits

### Models
- **Face Detection**: SSD ResNet-10 (OpenCV Model Zoo)
- **Age Prediction**: Caffe CNN model (Adience dataset)

### Libraries
- OpenCV: BSD License
- NumPy: BSD License
- yt-dlp: Unlicense
- python-vlc: LGPL
- ytmusicapi: MIT License

### References
- [OpenCV DNN Module](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)
- [Age and Gender Classification Paper](https://talhassner.github.io/home/publication/2015_CVPR)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)
- [YouTube Music API](https://github.com/sigma67/ytmusicapi)

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Author

**Naman Bhute**  
GitHub: [@namanbhute01-cloud](https://github.com/namanbhute01-cloud)

---

## Changelog

### v1.0.0 (Current)
- ✅ Real-time face detection and age prediction
- ✅ Four age group classification
- ✅ YouTube Music integration
- ✅ Interactive music preview with controls
- ✅ Automatic playback retry mechanism
- ✅ Environment-based configuration
- ✅ Cookie support for better extraction
- ✅ Robust error handling and fallbacks

---

**Project Status**: 🟢 Active Development

For issues, suggestions, or questions, please open an issue on GitHub.
