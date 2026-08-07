import time
from pathlib import Path
from toolz import pipe, curry
from datetime import datetime
from operator import itemgetter

first  = itemgetter(0)
second = itemgetter(1)
third  = itemgetter(2)

cmap = curry(map)
cfilter = curry(filter)

@curry
def pick(keys, d):
    return {k: d[k] for k in keys if k in d}

@curry
def omit(keys, d):
    return {k: v for k, v in d.items() if k not in keys}

@curry
def with_delay(seconds, f, x):
    time.sleep(seconds)
    return f(x)

@curry
def write_file(ext: str, outdir: Path, k: str, v: str):
    path = outdir / f"{k}.{ext}"
    path.write_text(v)
    return path

@curry
def tap(f, x):
    f(x)
    return x

@curry
def flatten(xss):
    for xs in xss:
        yield from xs

@curry
def starts_with(prefixes, s: str) -> bool:
    return s.startswith(tuple(prefixes))

@curry
def cdistinct(xs):
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            yield x

@curry
def prepend(prefix: str, value: str) -> str:
    return f"{prefix}{value}"

@curry
def append(suffix: str, value: str) -> str:
    return f"{value}{suffix}"

@curry
def is_cached(cache_dir: Path, key: str) -> bool:
    return any(cache_dir.glob(f"{key}.*"))

def iso_datetime_to_date(ts: str) -> str:
    return (datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat())

def contains_any(content: str, keywords: list[str]) -> bool:
    return any(k in content for k in keywords)


from urllib.parse import urlparse, parse_qsl, unquote, urlunparse
import re

DROP_QUERY_KEYS = {
    "contenttype",
    "content-type",
}

def url_to_slug(url: str) -> str:
    p = urlparse(url)

    host = p.netloc.lower()
    path = unquote(p.path).lower()

    # filter query keys
    query_pairs = [
        (k.lower(), v.lower())
        for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in DROP_QUERY_KEYS
    ]

    query = "-".join(
        f"{k}-{v}" if v else k
        for k, v in sorted(query_pairs)
    )

    raw = "-".join(filter(None, [host, path, query]))

    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)

    return slug

def clean_url(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
