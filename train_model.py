import pickle
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "website_urls_dataset.csv"
MODEL_PATH = BASE_DIR / "model" / "website_phishing_model.pkl"
PROBABILITY_THRESHOLD = 0.65

if __name__ == "__main__":
    sys.modules["train_model"] = sys.modules[__name__]

SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "update",
    "confirm",
    "bank",
    "account",
    "password",
    "payment",
    "claim",
    "win",
    "free",
    "gift",
    "urgent",
    "alert",
    "security",
    "click",
    "auth",
    "signin",
    "recover",
    "reset",
    "access",
    "session",
]

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "dot_count",
    "hyphen_count",
    "digit_count",
    "special_char_count",
    "subdomain_count",
    "slash_count",
    "query_count",
    "equal_count",
    "at_sign_count",
    "has_ip",
    "has_https",
    "has_http",
    "suspicious_word_count",
    "login_keyword_count",
    "verify_keyword_count",
    "account_keyword_count",
    "auth_keyword_count",
    "shortener_used",
]


def is_ip_address(domain):
    if not domain:
        return False
    parts = domain.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def is_valid_domain(hostname):
    if not hostname:
        return False
    hostname = hostname.rstrip(".")
    if not hostname or hostname.startswith(".") or hostname.endswith("."):
        return False
    if is_ip_address(hostname):
        return True
    labels = hostname.split(".")
    if len(labels) < 2:
        return False
    if any(not label for label in labels):
        return False
    tld = labels[-1]
    if len(tld) < 2 or not re.fullmatch(r"[a-zA-Z]{2,63}", tld):
        return False
    for label in labels[:-1]:
        if len(label) > 63 or not re.fullmatch(r"[a-zA-Z0-9-]+", label):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
    return True


def normalize_url(url):
    if pd.isna(url) or not isinstance(url, str):
        raise ValueError("Invalid URL: Please enter a valid website URL.")

    candidate = url.strip().lower()
    if not candidate:
        raise ValueError("Invalid URL: Please enter a valid website URL.")
    if " " in candidate:
        raise ValueError("Invalid URL: Please enter a valid website URL.")

    if not candidate.startswith(("http://", "https://")):
        if "." not in candidate or ("/" in candidate and "." not in candidate.split("/")[0]):
            raise ValueError("Invalid URL: Please enter a valid website URL.")
        candidate = "https://" + candidate

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid URL: Please enter a valid website URL.")
    if not parsed.netloc:
        raise ValueError("Invalid URL: Please enter a valid website URL.")

    hostname = parsed.hostname
    if not hostname or not is_valid_domain(hostname):
        raise ValueError("Invalid URL: Please enter a valid website URL.")

    return candidate


def safe_keyword_count(text):
    if not text:
        return 0
    lowered = text.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lowered)


def extract_url_features(url):
    clean_url = normalize_url(url)
    parsed = urlparse(clean_url)
    hostname = (parsed.hostname or "").lower()
    domain = hostname.split(":")[0]
    path = parsed.path.lower()
    query = parsed.query.lower()
    full_text = f"{hostname} {path} {query}"
    domain_parts = [part for part in domain.split(".") if part]
    subdomain_count = max(0, len(domain_parts) - 2) if not is_ip_address(domain) and "." in domain else 0

    return {
        "url_length": len(clean_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "dot_count": clean_url.count("."),
        "hyphen_count": clean_url.count("-"),
        "digit_count": sum(ch.isdigit() for ch in clean_url),
        "special_char_count": sum(ch in clean_url for ch in ["?", "=", "&", "%", "#", "@", "_", ";"]),
        "subdomain_count": subdomain_count,
        "slash_count": clean_url.count("/"),
        "query_count": clean_url.count("?"),
        "equal_count": clean_url.count("="),
        "at_sign_count": clean_url.count("@"),
        "has_ip": 1 if is_ip_address(domain) else 0,
        "has_https": 1 if clean_url.startswith("https://") else 0,
        "has_http": 1 if clean_url.startswith("http://") else 0,
        "suspicious_word_count": safe_keyword_count(full_text),
        "login_keyword_count": sum(1 for kw in ["login", "signin", "sign-in", "sign_in"] if kw in full_text),
        "verify_keyword_count": sum(1 for kw in ["verify", "verification", "confirm", "confirmation"] if kw in full_text),
        "account_keyword_count": sum(1 for kw in ["account", "bank", "payment", "wallet", "profile"] if kw in full_text),
        "auth_keyword_count": sum(1 for kw in ["auth", "authentication", "access", "recover", "reset"] if kw in full_text),
        "shortener_used": 1 if any(token in clean_url for token in ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"]) else 0,
    }


extract_url_features.__module__ = "train_model"


def build_feature_explanation(feature_dict):
    """Return only explanations supported by the extracted URL features."""
    positive = []
    concerns = []

    if feature_dict["has_ip"]:
        concerns.append("The URL uses an IP address instead of a normal domain name.")
    if feature_dict["url_length"] > 100:
        concerns.append("The URL is unusually long, which can be associated with obfuscation.")
    if feature_dict["hostname_length"] > 50:
        concerns.append("The hostname is unusually long.")
    if feature_dict["subdomain_count"] > 2:
        concerns.append("The URL contains several subdomains, which can make the destination harder to verify.")
    if feature_dict["special_char_count"] > 4:
        concerns.append("The URL contains an unusual number of special characters or encoded parameters.")
    if feature_dict["shortener_used"]:
        concerns.append("A URL-shortening service hides the final destination.")
    if feature_dict["suspicious_word_count"]:
        concerns.append("The domain or path contains security-sensitive or urgency-related keywords.")
    if feature_dict["at_sign_count"]:
        concerns.append("The URL contains an @ symbol, which can obscure the actual hostname.")

    if feature_dict["url_length"] <= 75:
        positive.append("The URL length is within a typical range.")
    if not feature_dict["has_ip"]:
        positive.append("No IP address was detected in place of a domain.")
    if feature_dict["subdomain_count"] <= 2:
        positive.append("The number of subdomains is within a typical range.")
    if feature_dict["at_sign_count"] == 0:
        positive.append("No @ symbol was detected in the URL.")

    return {"concerns": concerns, "positive": positive}


def prepare_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")

    required = {"url", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["url", "label"]].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["safe", "phishing"])].copy()

    modern_phishing_patterns = [
        "https://secure-login-auth.web.app/login-ui.html",
        "https://account-verify.web.app/secure-login",
        "https://signin-recover.web.app/account",
        "https://bank-login-update.firebaseapp.com/verify",
        "https://secure.access.web.app/confirm",
        "https://login.account.web.app/verify",
        "https://auth-login.web.app/secure",
        "https://payment-update.web.app/signin",
        "https://address-update.web.app/recover",
        "https://verify-login.web.app/account",
    ]
    modern_phishing_df = pd.DataFrame({"url": modern_phishing_patterns, "label": ["phishing"] * len(modern_phishing_patterns)})
    df = pd.concat([df, modern_phishing_df], ignore_index=True)

    feature_rows = df["url"].apply(extract_url_features)
    feature_df = pd.DataFrame(list(feature_rows), columns=FEATURE_COLUMNS, dtype=float)
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    return df


def train_and_save_model():
    dataset = prepare_dataset()
    feature_columns = FEATURE_COLUMNS
    X = dataset[feature_columns].astype(float)
    y = dataset["label"].map({"safe": 0, "phishing": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample")),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["safe", "phishing"])

    sys.modules.setdefault("train_model", sys.modules[__name__])
    extract_url_features.__module__ = "train_model"
    payload = {
        "model": pipeline,
        "feature_columns": feature_columns,
        "feature_extractor": extract_url_features,
        "label_map": {0: "safe", 1: "phishing"},
        "decision_threshold": PROBABILITY_THRESHOLD,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(payload, file)

    return payload


if __name__ == "__main__":
    result = train_and_save_model()
    print(f"Model trained successfully. Accuracy: {result['accuracy']:.4f}")
    print(result["report"])
    print("Confusion matrix:\n", result["confusion_matrix"])
