import os

CONFIG = {
    "azure_redis": {
        "host": "dcube.redis.cache.windows.net",
        "port": "6380",
        "db": 0,
        "password": "moht0ktXdXVjFHzh9zvSbhieYJ7CKKzdYAzCaCkoaTg="
    },
    "database": {
        "name": "dcube",
        "user": "dcube",
        "password": "D33Cub3d!",
        "host": "dcube.database.windows.net",
        "port": "1433"
    },
    "openai": {
        "api_key": "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"
    },
    "google_api": {
        "client_id": "35133980476-l14hfrntg9f96n2omvme09f58r98b0tn.apps.googleusercontent.com",
        "client_secret": "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",
        "scopes": [
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/contacts.readonly"
        ],
        "token_file": "token.json"
    },
    "microsoft_api": {
        "client_id": "4ld8Q~KT5XMRFhzV2N~2WEAVNDF1~MZbCBdcHbNL",
        "client_secret": "7018f836-4f2d-41e1-9a26-19f381ee9521",
        "redirect_uri": "https://dcube.ai",
        "tenant": "common",
        "scopes": [
            "https://graph.microsoft.com/Mail.ReadWrite",
            "https://graph.microsoft.com/Mail.Send",
            "https://graph.microsoft.com/Calendars.ReadWrite",
            "https://graph.microsoft.com/Files.ReadWrite",
            "https://graph.microsoft.com/Contacts.ReadWrite",
            "https://graph.microsoft.com/User.Read",
            "https://graph.microsoft.com/offline_access",
            "https://graph.microsoft.com/openid"
        ]
    }
}
