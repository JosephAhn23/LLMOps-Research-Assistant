"""
Large-scale web dataset ingestion - CommonCrawl, Wikipedia, multilingual.
Covers: Large-scale datasets, HuggingFace Datasets, multilingual corpora
"""
import hashlib
from urllib.request import urlopen
from typing import Dict, Iterator

import pandas as pd
from datasets import interleave_datasets, load_dataset

LANGUAGES = ["en", "fr", "de", "zh", "es"]
BUFFER_SIZE = 1000


class LargeScaleIngestionPipeline:
    """
    Streams large-scale web corpora without full download.
    Covers: CommonCrawl, Wikipedia multilingual, data quality filtering.
    """

    def stream_wikipedia(self, language: str = "en") -> Iterator[Dict]:
        """Stream Wikipedia articles for a given language."""
        dataset = load_dataset(
            "wikimedia/wikipedia",
            f"20231101.{language}",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        for article in dataset:
            yield {"id": article["id"], "text": article["text"], "source": f"wikipedia-{language}"}

    def stream_common_crawl(self) -> Iterator[Dict]:
        """Stream CC-News (CommonCrawl subset) - no full download needed."""
        dataset = load_dataset("cc_news", split="train", streaming=True)
        for article in dataset:
            yield {
                "id": hashlib.md5(article["url"].encode()).hexdigest(),
                "text": article["text"],
                "source": "cc_news",
                "url": article["url"],
            }

    def stream_warc_directly(
        self,
        warc_url: str,
        target_languages: list[str] | None = None,
        min_length: int = 500,
    ) -> Iterator[Dict]:
        """
        Parse raw WARC files directly from CommonCrawl storage.
        Adds language detection + content filtering for practical depth.
        """
        target_languages = target_languages or LANGUAGES
        try:
            from warcio.archiveiterator import ArchiveIterator
            import langdetect
        except Exception as exc:  # pragma: no cover - optional runtime deps
            raise RuntimeError("Install `warcio` and `langdetect` to use direct WARC ingestion.") from exc

        with urlopen(warc_url) as stream:
            for record in ArchiveIterator(stream):
                if record.rec_type != "response":
                    continue
                raw = record.content_stream().read()
                text = raw.decode("utf-8", errors="ignore")
                if len(text) < min_length:
                    continue
                try:
                    lang = langdetect.detect(text)
                except Exception:
                    continue
                if lang not in target_languages:
                    continue

                url = record.rec_headers.get_header("WARC-Target-URI") or ""
                doc_id = hashlib.md5(f"{url}:{len(text)}".encode()).hexdigest()
                yield {
                    "id": doc_id,
                    "text": text,
                    "source": "commoncrawl-warc",
                    "url": url,
                    "lang": lang,
                }

    def stream_multilingual(self) -> Iterator[Dict]:
        """Interleave multilingual Wikipedia streams."""
        streams = [
            load_dataset(
                "wikimedia/wikipedia",
                f"20231101.{lang}",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            for lang in LANGUAGES
        ]
        interleaved = interleave_datasets(streams)
        for article in interleaved:
            yield {"id": article["id"], "text": article["text"], "source": "wikipedia-multilingual"}

    def buffer_and_process(
        self,
        stream: Iterator[Dict],
        quality_filter,
        deduplicator,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Buffer streamed docs, apply quality filter + dedup, return DataFrame."""
        records = []
        seen_hashes = set()
        count = 0

        for doc in stream:
            if count >= limit:
                break
            if not quality_filter.is_quality(doc["text"]):
                continue
            doc_hash = deduplicator.hash(doc["text"])
            if doc_hash in seen_hashes:
                continue
            seen_hashes.add(doc_hash)
            records.append(doc)
            count += 1

            if len(records) % 500 == 0:
                print(f"Processed {len(records)} documents...")

        return pd.DataFrame(records)
