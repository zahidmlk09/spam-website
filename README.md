# PhishGuard

PhishGuard is a Flask application that estimates whether a website URL resembles safe or phishing traffic. It preserves the existing scikit-learn URL feature pipeline and adds feature-grounded explanations, authenticated scan history, a responsive interface, and QR-to-URL analysis.

## Run locally

```powershell
cd spam_phishing_system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

The SQLite database is created automatically on application startup. Existing databases are migrated with the additional account and scan metadata columns.

## Configuration

Set a strong secret in the environment before deployment:

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-value"
```

The development fallback secret in `app.py` is only for local demonstration and must not be used in production. No SMTP or SMS credentials are currently required. The QR camera uses the browser-compatible `html5-qrcode` library from its CDN; camera access requires HTTPS or localhost.

## Current features

- Existing RandomForest model and exact feature schema preserved
- URL validation and model probability output
- Feature-based reasons for concerns and safe characteristics
- Dedicated scanner and JSON `/api/analyze` endpoint
- Camera and uploaded-image QR scanning that reuses the URL model
- Password hashing, login by username/email/mobile, and SQLite scan history
- Profile fields and password change
- Responsive shared navigation, result details, and dashboard statistics

## Testing

```powershell
python -m py_compile app.py train_model.py
python -m unittest discover -v
```

The detector is an estimate, not a guarantee. A safe result should never be treated as permission to enter sensitive information on an unfamiliar website.

## Known limitations

Email/mobile OTP delivery, profile image storage, CSRF protection, rate limiting, and production-grade secret management still require a deployment decision and external service/configuration. The current profile route supports safe text fields and password hashing, but does not claim those additional production controls are implemented.
