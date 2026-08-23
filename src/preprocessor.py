import re

class ArabicTextPreprocessor:
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Removes diacritics and normalizes Arabic letter variants."""
        text = re.sub(r"[\u064B-\u0652]", "", text)  # Remove Tashkeel
        text = re.sub(r"[إأآا]", "ا", text)         # Normalize Alef
        text = re.sub(r"ى", "ي", text)              # Normalize Alef Maqsura
        text = re.sub(r"ؤ", "ء", text)              # Normalize Waw with Hamza
        text = re.sub(r"ئ", "ء", text)              # Normalize Yeh with Hamza
        text = re.sub(r"ة", "ه", text)              # Normalize Teh Marbuta
        return text.strip()

    @staticmethod
    def expand_dialect_query(query: str) -> str:
        """Maps common Egyptian/Gulf dialect terms to search tokens."""
        dialect_map = {
            "ازاي": "كيف طريقة",
            "ليه": "لماذا سبب",
            "عايز": "اريد طلب",
            "علشان": "لانه بسبب",
            "دلوقتي": "الان في الوقت الحالي",
            "فين": "اين المكان"
        }
        words = query.split()
        expanded = [dialect_map.get(w, w) for w in words]
        return " ".join(expanded)