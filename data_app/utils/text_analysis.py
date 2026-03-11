import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd

try:
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover - optional dependency
    WordCloud = None


STOPWORDS = {
    "the", "and", "for", "you", "that", "this", "with", "from", "have", "are", "was",
    "但是", "然后", "因为", "所以", "我们", "你们", "他们", "可以", "一个", "一些", "进行",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_']+|[\u4e00-\u9fff]{2,}")


def has_wordcloud_support() -> bool:
    return WordCloud is not None


def tokenize_text_series(series: pd.Series) -> List[str]:
    tokens = []
    for value in series.dropna().astype(str):
        for token in TOKEN_PATTERN.findall(value.lower()):
            if token not in STOPWORDS and len(token.strip()) > 1:
                tokens.append(token)
    return tokens


def compute_text_frequency(series: pd.Series, top_n: int = 30) -> pd.DataFrame:
    counter = Counter(tokenize_text_series(series))
    most_common: List[Tuple[str, int]] = counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=["token", "count"])


def generate_wordcloud_image(
    series: pd.Series,
    max_words: int = 200,
    background_color: str = "white",
):
    if WordCloud is None:
        return None

    frequency_df = compute_text_frequency(series, top_n=max_words * 2)
    if frequency_df.empty:
        return None

    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    kwargs = {
        "width": 1200,
        "height": 600,
        "background_color": background_color,
        "max_words": max_words,
        "collocations": False,
    }
    if font_path.exists():
        kwargs["font_path"] = str(font_path)

    wordcloud = WordCloud(**kwargs)
    frequencies = dict(zip(frequency_df["token"], frequency_df["count"]))
    return wordcloud.generate_from_frequencies(frequencies).to_array()
