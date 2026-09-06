import os
import tempfile
import unittest

from app import app


class SpamPhishingAppTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SECRET_KEY'] = 'test-secret'
        db_fd, app.config['DATABASE'] = tempfile.mkstemp(prefix='spam_test_', suffix='.db')
        os.close(db_fd)
        with app.app_context():
            from app import init_db
            init_db()
        self.client = app.test_client()

    def tearDown(self):
        if app.config.get('DATABASE') and os.path.exists(app.config['DATABASE']):
            os.remove(app.config['DATABASE'])

    def test_register_login_and_history(self):
        response = self.client.post('/register', data={
            'username': 'alice',
            'password': 'StrongPass123',
            'confirm_password': 'StrongPass123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Login', response.data)

        response = self.client.post('/login', data={
            'username': 'alice',
            'password': 'StrongPass123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome', response.data)

        response = self.client.post('/predict', data={'url': 'https://example.com'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'example.com', response.data)

        response = self.client.get('/history', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'example.com', response.data)

    def test_predict_accepts_bare_domain_urls(self):
        self.client.post('/register', data={
            'username': 'bob',
            'password': 'StrongPass123',
            'confirm_password': 'StrongPass123'
        }, follow_redirects=True)
        self.client.post('/login', data={
            'username': 'bob',
            'password': 'StrongPass123'
        }, follow_redirects=True)

        response = self.client.post('/predict', data={'url': 'google.com'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'google.com', response.data)
        self.assertNotIn(b'Unable to analyze this URL', response.data)

    def test_predict_accepts_suspicious_url(self):
        self.client.post('/register', data={
            'username': 'charlie',
            'password': 'StrongPass123',
            'confirm_password': 'StrongPass123'
        }, follow_redirects=True)
        self.client.post('/login', data={
            'username': 'charlie',
            'password': 'StrongPass123'
        }, follow_redirects=True)

        response = self.client.post('/predict', data={'url': 'https://verify-bank-account-security.net'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'verify-bank-account-security.net', response.data)
        self.assertNotIn(b'Unable to analyze this URL', response.data)

    def test_predict_rejects_invalid_urls(self):
        self.client.post('/register', data={
            'username': 'dana',
            'password': 'StrongPass123',
            'confirm_password': 'StrongPass123'
        }, follow_redirects=True)
        self.client.post('/login', data={
            'username': 'dana',
            'password': 'StrongPass123'
        }, follow_redirects=True)

        for invalid_url in ['asahga', 'hello', 'abc', 'test', '12345', '']:
            response = self.client.post('/predict', data={'url': invalid_url}, follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Invalid URL: Please enter a valid website URL.', response.data)

    def test_predict_flags_login_style_webapp_phishing(self):
        self.client.post('/register', data={
            'username': 'erin',
            'password': 'StrongPass123',
            'confirm_password': 'StrongPass123'
        }, follow_redirects=True)
        self.client.post('/login', data={
            'username': 'erin',
            'password': 'StrongPass123'
        }, follow_redirects=True)

        response = self.client.post('/predict', data={'url': 'https://docombike.web.app/login-ui.html'}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'SAFE', response.data)
        self.assertNotIn(b'96%', response.data)
        self.assertIn(b'docombike.web.app', response.data)


if __name__ == '__main__':
    unittest.main()
