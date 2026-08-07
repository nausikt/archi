#! /usr/bin/env python3
from functional_utils import cmap, cfilter, flatten, starts_with, with_delay, cdistinct, prepend, append, url_to_slug, first, write_file, is_cached, clean_url, tap
from typing import Callable
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from pathlib import Path
from bs4 import BeautifulSoup
from toolz import curry, pipe

def _norm_url(base: str, href: str) -> str | None:
    if not href:
        return None
    abs_url = urljoin(base, href)
    abs_url, _ = urldefrag(abs_url)
    scheme = urlparse(abs_url).scheme.lower()
    if scheme not in ("http", "https"):
        return None
    return abs_url

def extract_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return pipe(
        soup.select("a[href]"),
        cmap(lambda a: _norm_url(BASE_URL, a.get("href", ""))),
        cfilter(lambda u: u is not None),
        list,
        lambda xs: list(dict.fromkeys(xs)),  # distinct, keep order
    )

def fetch(pairs: (str, str), *, timeout_s: int = 20) -> list[str]:
    slug, url = pairs
    session = requests.Session()
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    return slug, r.text

def crawl(url: str, *, timeout_s: int = 20) -> list[str]:
    session = requests.Session()
    r = session.get(url, timeout=(10, 600))
    r.raise_for_status()
    links = extract_links(r.text)
    return [*links, url]

if __name__ == "__main__":
    BASE_URL="https://twiki.cern.ch/twiki/bin/view/CMSPublic/"
    output_dir=Path("/Users/aqua/archi/data/twiki/cache")
    cached_twiki_weblist=Path("/Users/aqua/archi/data/twiki/weblinks.list")
    ROOT_PREFIXES = ["SWGuideCrab", "WorkBook"]
    ALLOWED_PREFIXES = ["CRAB3", "SWGuide", "WorkBook", "Crab", "Crab3"]
    ALLOWED_PREFIXES_URLS = list(map(prepend(BASE_URL), ALLOWED_PREFIXES))
    ### Start Crawling ###
    links = pipe(
        ROOT_PREFIXES,
        cmap(prepend(BASE_URL)),
        ### Crawl: Depth=1 ###
        cmap(with_delay(1)(crawl)), 
        flatten,
        cfilter(starts_with(ALLOWED_PREFIXES_URLS)),
        cdistinct,
        ### Crawl: Depth=2 ###
        cmap(with_delay(1)(crawl)),
        flatten,
        cfilter(starts_with(ALLOWED_PREFIXES_URLS)),
        cdistinct,
        cmap(clean_url),
        cdistinct,
        list,
    )
    ### Write URLs weblist ###
    cached_twiki_weblist.write_text("\n".join(links) + "\n")
    links = cached_twiki_weblist.read_text().splitlines()
    ### Write cache/local Twikis ###
    files = pipe(
        links,
        cmap(clean_url),
        cdistinct,
        cmap(append("?contenttype=text/plain")),
        cmap(lambda url: (url_to_slug(url), url)), # cache by slug
        cfilter(lambda xs: not(is_cached(output_dir, xs[0]))),
        cmap(with_delay(1)(fetch)),
        cmap(lambda xs: write_file('txt', output_dir, *xs)),
        list
    )
    print(links[:5])
    print(len(links))
    print(len(files))
