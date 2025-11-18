import re

class Cleaner:
    def __init__(self, text):
        self.text = text


    def clean_hebrew_text(self, text):
        if not isinstance(text, str):
            return ""

        # 1. Normalize whitespace
        text = text.replace("\xa0", " ")
        text = text.replace("\t", " ")
        text = text.replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()

        # 2. Normalize punctuation
        text = text.replace("...", "…")
        text = text.replace("..", "…")
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = text.replace(" - ", " — ")

        # 3. Remove noise + invisible chars
        text = text.replace("\\/", "/")
        text = text.replace("\u200f", "")
        text = text.replace("\u200e", "")
        text = text.replace("\ufeff", "")
        text = re.sub(r"[\x00-\x1f]", "", text)

        # 4. Normalize Hebrew-specific chars
        text = text.replace("״", '"')
        text = text.replace("׳", "'")
        text = text.replace("־", "-")

        # 5. Clean HTML remnants
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&nbsp;", " ")
        text = text.replace("&quot;", '"')
        text = text.replace("&amp;", "&")

        # Final whitespace pass
        text = re.sub(r"\s+", " ", text).strip()

        return text