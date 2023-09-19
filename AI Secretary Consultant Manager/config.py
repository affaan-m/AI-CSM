import os

CONFIG = {
    "azure_redis": {
        "host": "dcube.redis.cache.windows.net",
        "port": "6380",
        "db": 0,
        "password": "moht0ktXdXVjFHzh9zvSbhieYJ7CKKzdYAzCaCkoaTg="
    },
    "database": {
        "name": "your-database-name",
        "user": "your-database-user",
        "password": "your-database-password",
        "host": "your-database-host",
        "port": "your-database-port"
    },
    "openai": {
        "api_key": "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"
    },
    "email": {
        "generic_email_credentials": {
            "username": "generic-email-username",
            "password": "generic-email-password",
        }
    },
    "google_api": {
        "calendar": {
            "credentials_path": "path/to/google/calendar/credentials.json",
        },
        "drive": {
            "credentials_path": "path/to/google/drive/credentials.json",
        }
    },
    "microsoft_api": {
        "calendar": {
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
            "redirect_uri": "your-redirect-uri",
            "refresh_token": "your-refresh-token",
        },
        "onedrive": {
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
            "redirect_uri": "your-redirect-uri",
            "refresh_token": "your-refresh-token",
        }
    },
    "apple_api": {
        "calendar": {
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
        },
        "icloud_drive": {
            "client_id": "your-client-id",
            "client_secret": "your-client-secret",
        }
    },
}
