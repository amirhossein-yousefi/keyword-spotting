from __future__ import annotations
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Union

import numpy as np
import soundfile as sf
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Value
from transformers import AutoFeatureExtractor, DataCollatorWithPadding
from tqdm import tqdm
from urllib.request import urlretrieve

from .augment import apply_training_augs
from .utils import seconds_to_samples

SC_V02_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

def _download_with_tqdm(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest

    class _TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with _TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=f"Downloading {dest.name}") as t:
        urlretrieve(url, filename=str(dest), reporthook=t.update_to)
    return dest

def _ensure_extracted(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir
    if root.exists() and (root / "validation_list.txt").exists():
        return root
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(path=extract_dir)
    return root

def _read_rel_list(p: Path) -> Set[str]:
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}

def _gather_files(root: Path) -> List[Path]:
    wavs: List[Path] = []
    for sub in root.iterdir():
        if sub.is_dir() and sub.name != "_background_noise_":
            wavs.extend(sub.glob("*.wav"))
    return wavs

def _build_split_datasets(root: Path, sample_rate: int) -> DatasetDict:
    # Use official split lists from the archive
    val_rel = _read_rel_list(root / "validation_list.txt")
    test_rel = _read_rel_list(root / "testing_list.txt")

    all_wavs = _gather_files(root)
    def rel(p: Path) -> str: return p.relative_to(root).as_posix()
    def label_of(p: Path) -> str: return p.parent.name

    splits = {"train": [], "validation": [], "test": []}
    for p in all_wavs:
        r = rel(p)
        if r in val_rel:
            splits["validation"].append((str(p), label_of(p)))
        elif r in test_rel:
            splits["test"].append((str(p), label_of(p)))
        else:
            splits["train"].append((str(p), label_of(p)))

    label_names = sorted({lbl for lst in splits.values() for _, lbl in lst})
    label_to_id = {n: i for i, n in enumerate(label_names)}

    feats = Features({
        # IMPORTANT: store paths as plain strings — no Audio() feature → no torchcodec needed
        "audio": Value("string"),
        "label": ClassLabel(names=label_names),
    })

    def to_hfds(items: List[Tuple[str, str]]) -> Dataset:
        if not items:
            return Dataset.from_dict({"audio": [], "label": []}, features=feats)
        paths, labels = zip(*items)
        label_ids = [label_to_id[l] for l in labels]
        return Dataset.from_dict({"audio": list(paths), "label": label_ids}, features=feats)

    return DatasetDict({k: to_hfds(v) for k, v in splits.items()})

def _load_speech_commands_noscript(sample_rate: int = 16000, data_cache_dir: str | None = None) -> DatasetDict:
    cache_dir = Path(data_cache_dir or (Path.cwd() / "data"))
    archive = cache_dir / "speech_commands_v0.02.tar.gz"
    root = cache_dir / "speech_commands"

    _download_with_tqdm(SC_V02_URL, archive)
    extracted = _ensure_extracted(archive, root)
    return _build_split_datasets(extracted, sample_rate=sample_rate)

def load_speech_commands(
    dataset_name: str = "speech_commands",
    dataset_config: str = "v0.02",
    sample_rate: int = 16000,
    subset_fraction: float = 1.0,
) -> DatasetDict:
    """
    Try the Hub dataset first. If disabled scripts or other issues occur,
    fall back to a script-less local loader (paths as strings).
    """
    try:
        ds = load_dataset(dataset_name, dataset_config)
        # If this succeeds, the dataset uses the Hub's features (may be Audio(...))
        # We intentionally don't cast to Audio() here to avoid forcing torchcodec.
    except Exception as e:
        msg = str(e)
        if "Dataset scripts are no longer supported" in msg or "speech_commands.py" in msg:
            print("[data_utils] Falling back to no-script loader for Speech Commands…")
            ds = _load_speech_commands_noscript(sample_rate=sample_rate)
        else:
            raise

    if subset_fraction < 1.0:
        def take_fraction(split):
            n = max(1, int(len(ds[split]) * subset_fraction))
            return ds[split].shuffle(seed=1234).select(range(n))
        ds = DatasetDict({k: take_fraction(k) for k in ds.keys()})

    return ds

def build_feature_extractor(model_name_or_path: str, sample_rate: int):
    fe = AutoFeatureExtractor.from_pretrained(model_name_or_path)
    return fe

def _linear_resample(wave: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wave.astype(np.float32)
    duration = len(wave) / float(sr_in)
    new_len = int(round(duration * sr_out))
    x_old = np.arange(len(wave), dtype=np.float64)
    x_new = np.linspace(0, len(wave) - 1, new_len, dtype=np.float64)
    return np.interp(x_new, x_old, wave).astype(np.float32)

def make_preprocess_fn(
    feature_extractor,
    sample_rate: int,
    max_duration_seconds: float,
    is_training: bool,
    augment_cfg: Dict[str, Any],
):
    max_len_samples = seconds_to_samples(max_duration_seconds, sample_rate)

    def _trim_or_pad(wave: np.ndarray) -> np.ndarray:
        if len(wave) > max_len_samples:
            return wave[:max_len_samples]
        if len(wave) < max_len_samples:
            pad = max_len_samples - len(wave)
            return np.pad(wave, (0, pad), mode="constant")
        return wave

    def _load_any(item: Union[dict, str]) -> np.ndarray:
        # Case 1: Hugging Face Audio feature decoded item
        if isinstance(item, dict) and "array" in item:
            wave = item["array"].astype(np.float32)
            sr_in = int(item.get("sampling_rate", sample_rate))
            if sr_in != sample_rate:
                wave = _linear_resample(wave, sr_in, sample_rate)
            return wave
        # Case 2: Our fallback stores a path string
        if isinstance(item, str):
            wave, sr_in = sf.read(item, dtype="float32", always_2d=False)
            if wave.ndim > 1:
                wave = wave.mean(axis=1)  # mono
            if sr_in != sample_rate:
                wave = _linear_resample(wave, sr_in, sample_rate)
            return wave.astype(np.float32)
        raise TypeError(f"Unsupported audio item type: {type(item)}")

    def preprocess(batch):
        waves = []
        for audio in batch["audio"]:
            wave = _load_any(audio)
            if is_training and augment_cfg.get("enabled", False):
                wave = apply_training_augs(
                    wave,
                    sample_rate=sample_rate,
                    time_shift_ms=int(augment_cfg.get("time_shift_ms", 100)),
                    noise_snr_db_min=augment_cfg.get("noise_snr_db_min", 10),
                    noise_snr_db_max=augment_cfg.get("noise_snr_db_max", 30),
                    random_gain_db=augment_cfg.get("random_gain_db", 6),
                )
            wave = _trim_or_pad(wave)
            waves.append(wave)
        inputs = feature_extractor(waves, sampling_rate=sample_rate)
        return {"input_values": inputs["input_values"], "labels": batch["label"]}

    return preprocess

def label_maps(ds: DatasetDict):
    labels: List[str] = ds["train"].features["label"].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}
    return labels, id2label, label2id

# at top of file, update imports
from transformers import AutoFeatureExtractor, DataCollatorWithPadding, default_data_collator

def build_data_collator(processor):
    """
    Works across transformers versions:
    - Newer versions: DataCollatorWithPadding(tokenizer=...)
    - Older versions: DataCollatorWithPadding(feature_extractor=...)
    - Fallback: default_data_collator (we already pad/trim to fixed length)
    """
    # Try the current API first
    try:
        return DataCollatorWithPadding(tokenizer=processor, padding=True)
    except TypeError:
        # Back-compat path
        try:
            return DataCollatorWithPadding(feature_extractor=processor, padding=True)
        except TypeError:
            # Our inputs are fixed-length already, so default stacking works
            return default_data_collator

