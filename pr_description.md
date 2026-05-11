🎯 **What:** Removed hardcoded default values for `admin_password` and `secret_key` in `smartwaste/settings.py`.

⚠️ **Risk:** Hardcoded credentials (`password123` and a static secret key) could be extracted from the codebase, allowing unauthorized access to the admin functionality and the ability to forge sessions.

🛡️ **Solution:** Removed defaults in `Pydantic` settings to force configuration exclusively via environment variables. Updated `tests/conftest.py` to provide test values so that the test suite runs correctly.
