"""
Real CommonCrawl WARC parsing - direct S3 access, language detection, domain scoring.
Covers: Large-scale web datasets (real depth, not just load_dataset)
"""
import gzip
import hashlib
import io
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

import boto3
import langdetect
import pandas as pd
import requests
from warcio.archiveiterator import ArchiveIterator

logger = logging.getLogger(__name__)

CC_INDEX_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/wet.paths.gz"
CC_S3_BUCKET = "commoncrawl"
TARGET_LANGUAGES = {"en", "fr", "de", "zh", "es", "ja", "ko"}

# Domain quality tiers - mirrors production data pipeline standards
HIGH_QUALITY_DOMAINS = {
    "wikipedia.org", "arxiv.org", "github.com",
    "stackoverflow.com", "nature.com", "pubmed.ncbi.nlm.nih.gov",
}
LOW_QUALITY_PATTERNS = [
    r"spam", r"casino", r"click-?here", r"buy-?now",
    r"\d{4}-\d{2}-\d{2}\.html$",
]


@dataclass
class WARCDocument:
    url: str
    text: str
    language: str
    domain: str
    quality_tier: str
    content_length: int
    warc_path: str
    doc_hash: str = field(default="")

    def __post_init__(self):
        self.doc_hash = hashlib.sha256(self.text.encode()).hexdigest()


class WARCIngestionPipeline:
    """
    Parses CommonCrawl WARC files directly from S3.
    Handles language detection, domain scoring, deduplication at scale.
    """

    # Maximum number of hashes to keep in memory per pipeline instance.
    # Each SHA-256 hex digest is 64 bytes; 500k entries ≈ 32 MB.
    _MAX_SEEN_HASHES = 500_000

    def __init__(self, target_languages: set = TARGET_LANGUAGES, n_workers: int = 4):
        self.target_languages = target_languages
        self.n_workers = n_workers
        self.s3 = boto3.client("s3", region_name="us-east-1")
        self.seen_hashes: set = set()
        self._hash_lock = threading.Lock()
        self._low_quality_re = re.compile("|".join(LOW_QUALITY_PATTERNS), re.IGNORECASE)

    def get_warc_paths(self, limit: int = 10) -> List[str]:
        """Fetch WET file paths from CommonCrawl index."""
        response = requests.get(CC_INDEX_URL, stream=True, timeout=30)
        response.raise_for_status()
        with gzip.open(io.BytesIO(response.content)) as f:
            paths = [line.decode().strip() for line in f]
        return paths[:limit]

    def _detect_language(self, text: str) -> Optional[str]:
        try:
            return langdetect.detect(text[:500])
        except Exception:
            return None

    def _score_domain(self, url: str) -> str:
        """Score domain quality: high / medium / low."""
        try:
            domain = urlparse(url).netloc.lower()
            domain = re.sub(r"^www\.", "", domain)
        except Exception:
            return "low"

        if any(domain == hq or domain.endswith("." + hq) for hq in HIGH_QUALITY_DOMAINS):
            return "high"
        if self._low_quality_re.search(url):
            return "low"
        return "medium"

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace, remove boilerplate."""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"http\S+", "[URL]", text)
        text = re.sub(r"[^\w\s.,;:!?\"'\-\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def stream_warc_from_s3(self, warc_path: str) -> Iterator[WARCDocument]:
        """
        Stream a single WARC file directly from CommonCrawl S3.
        No full download - processes records on the fly.
        """
        response = self.s3.get_object(Bucket=CC_S3_BUCKET, Key=warc_path)
        stream = response["Body"]

        for record in ArchiveIterator(stream):
            if record.rec_type != "conversion":
                continue

            url = record.rec_headers.get_header("WARC-Target-URI", "")
            if not url:
                continue

            try:
                raw_text = record.content_stream().read().decode("utf-8", errors="ignore")
            except Exception:
                continue

            text = self._clean_text(raw_text)

            words = text.split()
            if len(words) < 100 or len(words) > 50_000:
                continue

            lang = self._detect_language(text)
            if lang not in self.target_languages:
                continue

            doc_hash = hashlib.sha256(text.encode()).hexdigest()
            with self._hash_lock:
                if doc_hash in self.seen_hashes:
                    continue
                # Evict oldest entries when the cap is reached to bound memory.
                if len(self.seen_hashes) >= self._MAX_SEEN_HASHES:
                    self.seen_hashes.clear()
                    logger.warning(
                        "seen_hashes cap (%d) reached; cleared to prevent OOM. "
                        "Consider using a shared Redis set for cross-worker deduplication.",
                        self._MAX_SEEN_HASHES,
                    )
                self.seen_hashes.add(doc_hash)

            quality_tier = self._score_domain(url)
            domain = urlparse(url).netloc

            yield WARCDocument(
                url=url,
                text=text,
                language=lang,
                domain=domain,
                quality_tier=quality_tier,
                content_length=len(text),
                warc_path=warc_path,
                doc_hash=doc_hash,
            )

    def process_warc_paths_parallel(
        self,
        warc_paths: List[str],
        max_docs: int = 100_000,
    ) -> pd.DataFrame:
        """
        Process multiple WARC files in parallel using ThreadPoolExecutor.
        Returns DataFrame with quality metrics per document.
        """
        all_docs: List[Dict] = []

        def process_one(path: str) -> List[Dict]:
            docs = []
            for doc in self.stream_warc_from_s3(path):
                docs.append(
                    {
                        "url": doc.url,
                        "text": doc.text,
                        "language": doc.language,
                        "domain": doc.domain,
                        "quality_tier": doc.quality_tier,
                        "content_length": doc.content_length,
                        "doc_hash": doc.doc_hash,
                        "warc_path": doc.warc_path,
                    }
                )
                if len(docs) >= max_docs // len(warc_paths):
                    break
            return docs

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_one, p): p for p in warc_paths}
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    print(f"Processed {futures[future]}: {len(docs)} docs")
                except Exception as e:
                    print(f"Failed {futures[future]}: {e}")

                if len(all_docs) >= max_docs:
                    break

        df = pd.DataFrame(all_docs[:max_docs])
        self._log_dataset_stats(df)
        return df

    def _log_dataset_stats(self, df: pd.DataFrame):
        """Log corpus statistics."""
        if df.empty:
            print("No documents collected — corpus is empty.")
            return
        print("\n=== Corpus Statistics ===")
        print(f"Total documents: {len(df):,}")
        print(f"Languages:\n{df['language'].value_counts()}")
        print(f"Quality tiers:\n{df['quality_tier'].value_counts()}")
        print(f"Avg content length: {df['content_length'].mean():.0f} chars")
        print(f"Top domains:\n{df['domain'].value_counts().head(10)}")
        print(f"Unique domains: {df['domain'].nunique():,}")
