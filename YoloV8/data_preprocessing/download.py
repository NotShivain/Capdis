import argparse
import os
import sys
import cv2
import json
from pathlib import Path
from data_preprocessing.labels import CLASS_TO_IDX, load_game_labels
from SoccerNet.Downloader import SoccerNetDownloader

def _get_downloader(local_dir: str, password: str = None):
    downloader = SoccerNetDownloader(local_dir)
    if password:
        downloader.password = password
    return downloader

#---------------------------------------------------------
# Dataset 1: Labels + pre-extracted Resnet-152 features  |
#---------------------------------------------------------

def download_labels_and_features(local_dir: str, splits: list[str]):
    dl = _get_downloader(local_dir)
    print("Downloading: Labels-v2.json (all 500 games)")
    dl.downloadGames(files=["Labels-v2.json"], split=splits, timeout=300)

    print("Downloading: ResNet PCA-512 features")
    dl.downloadGames(
        files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
        split=splits,
    )

    print("Downloading: MaskRCNN player bboxess")
    dl.downloadGames(
        files=[
            "1_player_boundingbox_maskrcnn.json",
            "2_player_boundingbox_maskrcnn.json",
        ],
        split=splits,
    )

#-------------------------------------------------------------------
# Dataset 2: 224p matches for YOLO (download -> extract -> delete) |
#-------------------------------------------------------------------
    

# 15 selected matches with varying leagues, seasons, stadium types, kits and lighting conditions
YOLO_FINETUNING_GAMES = [
    "england_epl/2015-2016/2016-01-13 - 19-45 Chelsea 2 - 0 Newcastle United",
    "england_epl/2015-2016/2016-03-02 - 19-45 Arsenal 2 - 1 Tottenham Hotspur",
    "england_epl/2014-2015/2015-05-24 - 16-00 Manchester United 1 - 1 Arsenal",
    "france_ligue-1/2015-2016/2016-02-07 - 14-00 Paris Saint-Germain 4 - 1 Montpellier",
    "france_ligue-1/2014-2015/2015-04-18 - 21-05 Marseille 2 - 3 Paris Saint-Germain",
    "germany_bundesliga/2015-2016/2016-04-02 - 15-30 Bayern Munich 0 - 2 Borussia Dortmund",
    "germany_bundesliga/2014-2015/2015-04-25 - 18-30 Borussia Dortmund 0 - 1 Wolfsburg",
    "italy_serie-a/2015-2016/2016-01-06 - 20-45 Juventus 3 - 1 Bologna",
    "italy_serie-a/2014-2015/2015-05-18 - 20-45 Inter 1 - 4 Juventus",
    "spain_laliga/2015-2016/2016-02-13 - 20-00 Atletico Madrid 1 - 0 Celta Vigo",
    "spain_laliga/2015-2016/2016-04-02 - 15-15 Real Madrid 3 - 0 Atletico Madrid",
    "spain_laliga/2014-2015/2015-05-17 - 18-00 Barcelona 2 - 0 Athletic Club",
    "england_epl/2015-2016/2016-04-17 - 16-00 Manchester City 2 - 1 West Bromwich",
    "france_ligue-1/2015-2016/2016-04-30 - 21-00 Paris Saint-Germain 1 - 0 Caen",
    "germany_bundesliga/2015-2016/2016-05-07 - 15-30 Bayern Munich 2 - 1 Augsburg",
]

def download_and_extract_yolo(
        local_dir: str,
        password: str,
        frames_out_dir: str,
        n_games: int = 15,
        sample_fps: float = 5.0,
        delete_after: bool = True
):
    sys.path.insert(0, str(Path(__file__).parent.parent))

    dl = _get_downloader(local_dir, password)
    games = YOLO_FINETUNING_GAMES[:n_games]
    os.makedirs(frames_out_dir, exist_ok=True)

    for i, game_path in enumerate(games):
        game_dir = os.path.join(local_dir, game_path)
        os.makedirs(game_dir, exist_ok=True)
        print(f"\n  [{i+1}/{n_games}] {Path(game_path).name}")

        try:
            dl.downloadSingleGame(
                game=game_path,
                files=["1_224p.mkv", "2_224p.mkv"],
            )
        except Exception as e:
            print(f"    WARN: download failed — {e}")
            continue

        half_events = load_game_labels(game_dir)

        for half in (1,2):
            video_path = os.path.join(game_dir, f"{half}_224p.mkv")
            if not os.path.exists(video_path):
                continue
            n_saved = _extract_frames_around_events(
                video_path=video_path,
                events=half_events.get(half, []),
                out_dir=frames_out_dir,
                sample_fps=sample_fps,
                context_secs=3.0,
                game_tag=f"g{i:02d}_h{half}",
            )

            if delete_after:
                os.remove(video_path)

            print(f"Half {half}: {n_saved} frames saved"
                  + (" (video deleted)" if delete_after else ""))


def _extract_frames_around_events(
        video_path: str,
        events: list[dict],
        out_dir: str,
        sample_fps: float,
        context_secs: float,
        game_tag: str,
) -> int:
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every_n      = max(1, int(src_fps / sample_fps))

    target_frames = set()
    for ev in events:
        if ev.get("our_label") is None:
            continue
        center = int(ev["timestamp"] * src_fps)
        span   = int(context_secs * src_fps)
        for f in range(center - span, center + span, every_n):
            if 0 <= f < total_frames:
                target_frames.add(f)

    if not target_frames:
        cap.release()
        return 0
    frame_idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_frames:
            fname = os.path.join(out_dir, f"{game_tag}_{frame_idx:08d}.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


#-------------------------------------------------------------------
# Dataset 3: Ball action spotting (download -> extract -> delete)  |
#-------------------------------------------------------------------

def download_and_extract_ball_action(
        local_dir: str,
        password: str,
        frames_out_dir: str,
        delete_after: bool = True,
):
    dl = _get_downloader(local_dir, password)
    os.makedirs(frames_out_dir, exist_ok=True)

    dl.downloadDataTask(
        task="spotting-ball-2024",
        split=["train", "valid", "test"],
        password=password,
    )
    ball_action_root = os.path.join(local_dir, "spotting-ball-2024")

    game_dirs = sorted(set(
        v.parent for v in Path(ball_action_root).rglob("*.mkv")
    ))

    for i, game_dir in enumerate(game_dirs):
        print(f"\n  [{i+1}/{len(game_dirs)}] {game_dir.name}")
        for video_path in sorted(game_dir.glob("*.mkv")):
            half = 1 if video_path.name.startswith("1") else 2
            label_file = game_dir / f"{half}_labels.json"

            events = []
            if label_file.exists():
                with open(label_file) as f:
                    raw = json.load(f)
                for ann in raw.get("annotations", []):
                    sn_label  = ann.get("label", "")
                    ts = ann.get("position", 0) / 1000.0  # ms -> seconds
                    events.append({
                        "timestamp":  ts,
                        "our_label":  sn_label,
                        "class_idx":  CLASS_TO_IDX.get(sn_label) if sn_label else None,
                    })

            n_saved = _extract_frames_around_events(
                video_path=str(video_path),
                events=events,
                out_dir=frames_out_dir,
                sample_fps=5.0,
                context_secs=2.0,
                game_tag=f"ball_g{i:02d}_h{half}",
            )

            if delete_after:
                video_path.unlink()

            print(f"    Half {half}: {n_saved} frames"
                  + (" (video deleted)" if delete_after else ""))

#-------------------------------------------------------------------
# Helper Functions                                                 |
#-------------------------------------------------------------------

def _dir_size_gb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9


def print_storage_summary(local_dir: str):
    print("\n═══ Storage Summary ══════════════════════════")

    sections = [
        ("Labels-v2.json",    "*Labels-v2.json"),
        ("ResNet features",   "*_ResNET_TF2_PCA512.npy"),
        ("MaskRCNN bboxes",   "*_player_boundingbox_maskrcnn.json"),
        ("YOLO frames",       "yolo_frames/*.jpg"),
        ("Ball Action frames","ball_action_frames/*.jpg"),
    ]

    total = 0.0
    for label, pattern in sections:
        if "/*.jpg" in pattern:
            d = os.path.join(local_dir, pattern.split("/")[0])
            gb = _dir_size_gb(d)
        else:
            gb = sum(
                f.stat().st_size for f in Path(local_dir).rglob(pattern.lstrip("*"))
                if f.is_file()
            ) / 1e9
        total += gb
        print(f"  {label:26s}  {gb:.2f} GB")

    print(f"  {'TOTAL':26s}  {total:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SoccerNet data download for YoloV8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full recommended download
  python data/download_soccernet.py --dir /data/soccernet --password YOUR_PW

  # Labels + features only (trains spotter, skips YOLO finetuning)
  python data/download_soccernet.py --dir /data/soccernet --no-video

  # More YOLO games if you have storage headroom
  python data/download_soccernet.py --dir /data/soccernet --password YOUR_PW --yolo-games 25
        """
    )
    parser.add_argument("--dir", required = True, help = "Local storage root")
    parser.add_argument("--password", default = None, help = "SoccerNet NDA Password")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--yolo-games", type=int, default=15)
    parser.add_argument("--no-video", action="store_true", help="Skip video download — spotter training only")
    parser.add_argument("--keep-video", action="store_true", help="Do not delete video after frame extraction")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)    

    download_labels_and_features(args.dir, args.splits)

    if not args.no_video:
        if not args.password:
            print("ERROR: --password required for video. Use --no-video to skip.")
            sys.exit(1)
        download_and_extract_yolo(
            local_dir=args.dir,
            password=args.password,
            frames_out_dir=os.path.join(args.dir, "yolo_frames"),
            n_games=args.yolo_games,
            delete_after=not args.keep_video,
        )

        download_and_extract_ball_action(
            local_dir=args.dir,
            password=args.password,
            frames_out_dir=os.path.join(args.dir, "ball_action_frames"),
            delete_after=not args.keep_video,
        )
    print_storage_summary(args.dir)
    