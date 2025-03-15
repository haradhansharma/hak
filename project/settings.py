

from pathlib import Path
import environ
import os

BASE_DIR = Path(__file__).resolve().parent.parent

environ.Env.read_env(os.path.join(BASE_DIR, ".env"))
env = environ.Env()

SECRET_KEY = env("HAK_SECRET_KEY")

if not SECRET_KEY:
    raise ValueError("No  HAK_SECRET_KEY set for production please set it in .env file!")

DEBUG = env.bool("HAK_DEBUG", default=False)

if not DEBUG:
    PREPEND_WWW = True

if DEBUG:
    ALLOWED_HOSTS = ["*"]
    CSRF_TRUSTED_ORIGINS = env("HAK_CSRF_TRUSTED_ORIGINS_LOCAL").split(",")
else:
    ALLOWED_HOSTS = env("HAK_ALLOWED_HOSTS").split(",")
    CSRF_TRUSTED_ORIGINS = env("HAK_CSRF_TRUSTED_ORIGINS_LIVE").split(",")
    
SITE_ID = 1

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    "whitenoise.runserver_nostatic",
    'django.contrib.staticfiles',
    "django.contrib.sites",
    'django.contrib.sitemaps',
    "django.contrib.humanize",
    
    "account",
    "common"
]

AUTH_USER_MODEL = "account.User"
AUTHENTICATION_BACKENDS = ["django.contrib.auth.backends.ModelBackend"]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    "whitenoise.middleware.WhiteNoiseMiddleware",
    'django.contrib.sessions.middleware.SessionMiddleware',
    "django.contrib.sites.middleware.CurrentSiteMiddleware",
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, "templates")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                "common.context_processor.hak_common_context"
            ],
        },
    },
]

WSGI_APPLICATION = 'project.wsgi.application'

if DEBUG:

    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": env("HAK_DB_NAME"),
            "USER": env("HAK_DB_USER"),
            "PASSWORD": env("HAK_DB_PASSWORD"),
            "HOST": env("HAK_DB_HOST"),
            "PORT": env("HAK_DB_PORT"),      
            
            "OPTIONS": {
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
    
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": env("HAK_DB_NAME_PRO"),
            "USER": env("HAK_DB_USER_PRO"),
            "PASSWORD": env("HAK_DB_PASSWORD_PRO"),
            "HOST": env("HAK_DB_HOST_PRO"),
            "PORT": env("HAK_DB_PORT_PRO"),      
            
            "OPTIONS": {
                "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
            },
        }
    }
    


AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': os.path.join(BASE_DIR,"cache"),
        'TIMEOUT': 3600,
        'OPTIONS': {
            'MAX_ENTRIES': 1000
        }
    }
}

STATIC_URL = '/static/'

if DEBUG:
    STATICFILES_DIRS = [
        os.path.join(BASE_DIR, "static")
    ]
else:
    STATIC_ROOT = os.path.join(BASE_DIR, "static")
    
    
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
X_FRAME_OPTIONS = "SAMEORIGIN"
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE=True
FILE_UPLOAD_DIRECTORY_PERMISSIONS = 0o775
FILE_UPLOAD_PERMISSIONS = 0o644

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_HOST = True
    SECURE_SSL_REDIRECT = True
    
    


FORMATTERS = (
    {
        "verbose": {
            "format": "{levelname} {asctime} {name} {threadName} {thread} {pathname} {lineno} {funcName} {process} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {asctime} {pathname} {lineno} {message}",
            "style": "{",
        },
    },
)

HANDLERS = {
    "console_handler": {
        "class": "logging.StreamHandler",
        "formatter": "simple",
        "level": "DEBUG"
    },
    "info_handler": {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": os.path.join(BASE_DIR, 'logs/info.log' ),
        "mode": "a",
        "encoding": "utf-8",
        "formatter": "verbose",
        "level": "INFO",
        "backupCount": 5,
        "maxBytes": 1024 * 1024 * 5,  # 5 MB
    },
    "error_handler": {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": os.path.join(BASE_DIR, 'logs/error.log' ),
        "mode": "a",
        "formatter": "verbose",
        "level": "WARNING",
        "backupCount": 5,
        "maxBytes": 1024 * 1024 * 5,  # 5 MB
    },
    'hak_handler': {
        "filename": os.path.join(BASE_DIR, 'logs/debug.log' ),
        "mode": "a",
        "maxBytes": 1024 * 1024 * 5,  # 5 MB            
        'class': 'logging.handlers.RotatingFileHandler',
        "formatter": "simple",
        "encoding": "utf-8",
        "level": "DEBUG",
        "backupCount": 5,
        
    }
}

LOGGERS = (
    {
        "django": {
            "handlers": ["console_handler", "info_handler"],
            "level": "INFO",
           
        },
        "django.request": {
            "handlers": ["error_handler"],
            'level': 'INFO',             
            "propagate": True,
        },
        "django.template": {
            "handlers": ["error_handler"],
            'level': 'INFO',             
            "propagate": False,
        },
        "django.server": {
            "handlers": ["error_handler"],
            'level': 'INFO',             
            "propagate": True,
        },
        'log': {
            'handlers': ['console_handler', 'hak_handler'],
            'level': 'INFO', 
            'level': 'DEBUG',   
            "propagate": True,    
            
        },
    },
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": FORMATTERS[0],
    "handlers": HANDLERS,
    "loggers": LOGGERS[0],
}
