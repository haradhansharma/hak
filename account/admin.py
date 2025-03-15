from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from account.models import User
"""
Registers the User model with the Django admin interface.

This allows the User model to be managed through the Django admin site.
"""

class UserAdmin(UserAdmin):
    pass
admin.site.register(User, UserAdmin)
