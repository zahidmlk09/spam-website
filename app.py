import os
import pickle
import re
import secrets
import sqlite3
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from train_model import FEATURE_COLUMNS, build_feature_explanation, normalize_url

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "spam-phishing-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "website_phishing_model.pkl"
DEFAULT_DATABASE_PATH = BASE_DIR / "spam_phishing.db"
app.config["PROFILE_UPLOAD_FOLDER"] = str(BASE_DIR / "static" / "uploads")


def get_db():
    db_path = app.config.get("DATABASE", str(DEFAULT_DATABASE_PATH))
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT DEFAULT '',
            username TEXT UNIQUE NOT NULL,
            email TEXT DEFAULT '',
            mobile TEXT DEFAULT '',
            password_hash TEXT NOT NULL,
            profile_image TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            verdict TEXT NOT NULL,
            risk_level TEXT DEFAULT 'low',
            scan_type TEXT DEFAULT 'URL',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    user_columns = {row[1] for row in db.execute("PRAGMA table_info(users)").fetchall()}
    for column, definition in {
        "full_name": "TEXT DEFAULT ''",
        "email": "TEXT DEFAULT ''",
        "mobile": "TEXT DEFAULT ''",
        "profile_image": "TEXT DEFAULT ''",
    }.items():
        if column not in user_columns:
            db.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
    scan_columns = {row[1] for row in db.execute("PRAGMA table_info(scan_history)").fetchall()}
    for column, definition in {"risk_level": "TEXT DEFAULT 'low'", "scan_type": "TEXT DEFAULT 'URL'"}.items():
        if column not in scan_columns:
            db.execute(f"ALTER TABLE scan_history ADD COLUMN {column} {definition}")
    db.commit()
    db.close()


def get_user_by_username(username):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    db.close()
    return user


def get_user_by_identifier(identifier):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE lower(username) = lower(?) OR lower(email) = lower(?) OR mobile = ?",
        (identifier, identifier, identifier),
    ).fetchone()
    db.close()
    return user


def get_user_by_id(user_id):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    db.close()
    return user


def _field_used_by_other_user(field, value, user_id):
    if field not in {"email", "mobile"}:
        raise ValueError("Unsupported user field")
    db = get_db()
    user = db.execute(
        f"SELECT id FROM users WHERE {field} = ? AND id != ? AND {field} != ''",
        (value, user_id),
    ).fetchone()
    db.close()
    return user is not None


def get_recent_scans(user_id, limit=10):
    db = get_db()
    scans = db.execute(
        """
        SELECT id, url, label, confidence, verdict, risk_level, scan_type, created_at
        FROM scan_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    db.close()
    return scans


def save_scan(user_id, url, label, confidence, verdict, risk_level="low", scan_type="URL"):
    db = get_db()
    db.execute(
        """
        INSERT INTO scan_history (user_id, url, label, confidence, verdict, risk_level, scan_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, url, label, confidence, verdict, risk_level, scan_type),
    )
    db.commit()
    db.close()


def get_scan_stats(user_id):
    db = get_db()
    stats = db.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN label = 'safe' THEN 1 ELSE 0 END) AS safe,
               SUM(CASE WHEN label = 'phishing' THEN 1 ELSE 0 END) AS phishing,
               SUM(CASE WHEN scan_type = 'QR' THEN 1 ELSE 0 END) AS qr
        FROM scan_history WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    db.close()
    return stats


@app.before_request
def load_user():
    user_id = session.get("user_id")
    g.user = get_user_by_id(user_id) if user_id else None


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please log in to access your dashboard.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.errorhandler(413)
def request_entity_too_large(error):
    if g.user is not None:
        flash("The uploaded profile image is too large. Please choose an image under 8 MB.", "error")
        return redirect(url_for("profile"))
    flash("The submitted file is too large. Please try a smaller image.", "error")
    return redirect(url_for("login"))


def redirect_authenticated_user():
    if g.user is None:
        flash("Please log in or create an account to continue.", "warning")
        return redirect(url_for("login"))
    return None


def load_model():
    if not MODEL_PATH.exists():
        from train_model import train_and_save_model

        return train_and_save_model()

    try:
        with MODEL_PATH.open("rb") as file:
            payload = pickle.load(file)
    except Exception:
        app.logger.exception("Saved model is unreadable; retraining a fresh model.")
        from train_model import train_and_save_model

        return train_and_save_model()

    required_keys = {"model", "feature_columns", "feature_extractor", "label_map", "decision_threshold"}
    if not isinstance(payload, dict) or not required_keys.issubset(payload):
        app.logger.warning("Saved model is missing required fields; retraining a fresh model.")
        from train_model import train_and_save_model

        return train_and_save_model()

    return payload


def validate_url(url):
    if url is None:
        raise ValueError("Invalid URL: Please enter a valid website URL.")
    return normalize_url(url)


def build_feature_vector(feature_extractor, valid_url, feature_columns):
    feature_dict = feature_extractor(valid_url)
    feature_values = pd.DataFrame([feature_dict], columns=feature_columns)
    feature_values = feature_values.reindex(columns=feature_columns, fill_value=0.0).astype(float)
    if feature_values.shape[1] != len(feature_columns):
        raise ValueError("Feature extraction produced an unexpected vector shape.")
    return feature_values, feature_dict


def analyze_url(url):
    valid_url = validate_url(url)
    payload = load_model()
    feature_columns = list(payload["feature_columns"])
    model = payload["model"]
    class_order = list(model.classes_)
    decision_threshold = float(payload.get("decision_threshold", 0.65))

    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Feature mismatch: the saved model was trained with a different feature schema. "
            f"Expected {FEATURE_COLUMNS}, got {feature_columns}"
        )

    feature_values, feature_dict = build_feature_vector(payload["feature_extractor"], valid_url, feature_columns)
    probabilities = model.predict_proba(feature_values)[0]
    predicted_value = model.predict(feature_values)[0]
    probability_map = {int(cls): float(prob) for cls, prob in zip(class_order, probabilities)}
    predicted_index = int(list(class_order).index(predicted_value))
    safe_probability = probability_map.get(0, 0.0)
    phishing_probability = probability_map.get(1, 0.0)
    label = payload["label_map"].get(int(predicted_value), "safe")
    confidence = round(float(probabilities[predicted_index] * 100), 2)
    parsed = urlparse(valid_url)
    explanation = build_feature_explanation(feature_dict)
    risk_level = "high" if label == "phishing" and confidence >= 80 else "medium" if label == "phishing" else "low"

    app.logger.info("entered_url=%s", url)
    app.logger.info("validation_result=VALID")
    app.logger.info("parsed_hostname=%s", parsed.hostname)
    app.logger.info("parsed_path=%s", parsed.path)
    app.logger.info("extracted_features=%s", feature_dict)
    app.logger.info("feature_vector_shape=%s", feature_values.shape)
    app.logger.info("model_class_order=%s", class_order)
    app.logger.info("model_prediction=%s", predicted_value)
    app.logger.info("prediction_probabilities=%s", probability_map)
    app.logger.info("final_classification=%s", label)

    if max(probabilities) < decision_threshold:
        app.logger.warning("Low-confidence ML decision for URL %s. Model confidence: %s", valid_url, max(probabilities))

    return label, confidence, valid_url, {
        "safe_probability": round(safe_probability * 100, 2),
        "phishing_probability": round(phishing_probability * 100, 2),
        "class_order": [int(c) for c in class_order],
        "feature_vector": feature_values.iloc[0].to_dict(),
        "feature_names": feature_columns,
        "features": {**feature_dict, "hostname": parsed.hostname, "path": parsed.path},
        "explanation": explanation,
        "risk_level": risk_level,
        "hostname": parsed.hostname,
        "path": parsed.path,
    }


@app.route("/", methods=["GET"])
def index():
    if g.user is None:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = (request.form.get("username") or request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""

        user = get_user_by_identifier(username)
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")
        return render_template("login.html")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("index"))

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        mobile = (request.form.get("mobile") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")

        if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Please enter a valid email address.", "error")
            return render_template("register.html")

        if mobile and not re.fullmatch(r"\+?[0-9 ()-]{7,20}", mobile):
            flash("Please enter a valid mobile number.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if get_user_by_username(username):
            flash("This username is already taken.", "error")
            return render_template("register.html")

        db = get_db()
        db.execute(
            "INSERT INTO users (full_name, username, email, mobile, password_hash) VALUES (?, ?, ?, ?, ?)",
            (full_name, username, email, mobile, generate_password_hash(password)),
        )
        db.commit()
        db.close()
        flash("Account created successfully. Please sign in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    recent_scans = get_recent_scans(g.user["id"], limit=5)
    return render_template("dashboard.html", recent_scans=recent_scans, stats=get_scan_stats(g.user["id"]))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    url = (request.form.get("url") or "").strip()
    recent_scans = get_recent_scans(g.user["id"], limit=5)

    try:
        label, confidence, valid_url, debug_info = analyze_url(url)
    except ValueError as exc:
        return render_template("dashboard.html", error=str(exc), recent_scans=recent_scans, stats=get_scan_stats(g.user["id"]))
    except Exception as exc:
        app.logger.exception("Unexpected prediction failure for URL: %s", url)
        error_message = (
            str(exc)
            if app.debug
            else "Unable to analyze this URL. Please try a valid website URL."
        )
        return render_template("dashboard.html", error=error_message, recent_scans=recent_scans, stats=get_scan_stats(g.user["id"]))

    if label == "phishing":
        verdict = "Suspicious or phishing website detected."
    else:
        verdict = "Legitimate and safe website detected."

    save_scan(g.user["id"], valid_url, label, confidence, verdict, debug_info["risk_level"])

    return render_template(
        "result.html",
        label=label.upper(),
        confidence=confidence,
        verdict=verdict,
        risk_level=debug_info["risk_level"],
        explanation=debug_info["explanation"],
        features=debug_info["features"],
        url=valid_url,
        recent_scans=recent_scans,
    )


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    payload = request.get_json(silent=True) or request.form
    raw_url = (payload.get("url") or "").strip()
    try:
        label, confidence, valid_url, details = analyze_url(raw_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}, 400
    except Exception:
        app.logger.exception("API prediction failure")
        return {"ok": False, "error": "The model could not analyze this URL right now."}, 500

    result = {
        "ok": True,
        "url": valid_url,
        "label": label,
        "confidence": confidence,
        "risk_level": details["risk_level"],
        "explanation": details["explanation"],
        "features": details["features"],
        "safe_probability": details["safe_probability"],
        "phishing_probability": details["phishing_probability"],
    }
    if g.user is not None:
        verdict = "Suspicious or phishing website detected." if label == "phishing" else "Legitimate and safe website detected."
        save_scan(g.user["id"], valid_url, label, confidence, verdict, details["risk_level"], "QR" if payload.get("scan_type") == "QR" else "URL")
    return result


@app.route("/history", methods=["GET"])
@login_required
def history():
    scans = get_recent_scans(g.user["id"], limit=100)
    return render_template("history.html", scans=scans)


@app.route("/scanner")
@login_required
def scanner():
    return render_template("scanner.html")


@app.route("/qr-scanner")
@login_required
def qr_scanner():
    return render_template("qr_scanner.html")


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        mobile = (request.form.get("mobile") or "").strip()
        current_password = request.form.get("current_password") or ""
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
            flash("Username must be 3-30 characters using letters, numbers, dots, underscores, or hyphens.", "error")
        elif get_user_by_username(username) and get_user_by_username(username)["id"] != g.user["id"]:
            flash("That username is already in use.", "error")
        elif email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            flash("Please enter a valid email address.", "error")
        elif email and _field_used_by_other_user("email", email, g.user["id"]):
            flash("That email address is already in use.", "error")
        elif mobile and not re.fullmatch(r"\+?[0-9 ()-]{7,20}", mobile):
            flash("Please enter a valid mobile number.", "error")
        elif mobile and _field_used_by_other_user("mobile", mobile, g.user["id"]):
            flash("That mobile number is already in use.", "error")
        elif password and not check_password_hash(g.user["password_hash"], current_password):
            flash("Enter your current password before choosing a new password.", "error")
        elif password and len(password) < 6:
            flash("New password must be at least 6 characters long.", "error")
        elif password != confirm_password:
            flash("New passwords do not match.", "error")
        else:
            profile_image = g.user["profile_image"] or ""
            image = request.files.get("profile_image")
            if image and image.filename:
                safe_name = secure_filename(image.filename)
                extension = Path(safe_name).suffix.lower()
                if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
                    flash("Profile image must be PNG, JPG, JPEG, or WEBP.", "error")
                    return render_template("profile.html")
                upload_folder = Path(app.config["PROFILE_UPLOAD_FOLDER"])
                upload_folder.mkdir(parents=True, exist_ok=True)
                stored_name = f"user-{g.user['id']}-{secrets.token_hex(8)}{extension}"
                image.save(upload_folder / stored_name)
                profile_image = stored_name
            db = get_db()
            db.execute("UPDATE users SET full_name = ?, username = ?, email = ?, mobile = ?, profile_image = ?, password_hash = COALESCE(NULLIF(?, ''), password_hash) WHERE id = ?", (full_name, username, email, mobile, profile_image, generate_password_hash(password) if password else "", g.user["id"]))
            db.commit()
            db.close()
            flash("Profile updated successfully.", "success")
            return redirect(url_for("profile"))
    return render_template("profile.html")


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


init_db()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
