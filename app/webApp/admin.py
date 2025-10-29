from django.contrib import admin
from .models import Operations, Conference, Paper, Review, MetaReview


@admin.register(Operations)
class OperationsAdmin(admin.ModelAdmin):
    list_display = ["name"]
    search_fields = ["name"]


@admin.register(Conference)
class ConferenceAdmin(admin.ModelAdmin):
    list_display = ["name", "year", "last_update"]
    list_filter = ["year"]
    search_fields = ["name"]
    readonly_fields = ["last_update"]


class ReviewInline(admin.TabularInline):
    model = Review
    extra = 1
    fields = ["text", "last_update"]
    readonly_fields = ["last_update"]


class MetaReviewInline(admin.TabularInline):
    model = MetaReview
    extra = 1
    fields = ["text", "last_update"]
    readonly_fields = ["last_update"]


@admin.register(Paper)
class PaperAdmin(admin.ModelAdmin):
    list_display = ["title", "doi", "last_update"]
    search_fields = ["title", "doi", "authors", "abstract"]
    readonly_fields = ["last_update"]
    inlines = [ReviewInline, MetaReviewInline]


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ["paper", "last_update", "text_preview"]
    list_filter = ["last_update"]
    search_fields = ["text", "paper__title"]
    readonly_fields = ["last_update"]

    def text_preview(self, obj):
        return obj.text[:100] + "..." if len(obj.text) > 100 else obj.text

    text_preview.short_description = "Text preview"


@admin.register(MetaReview)
class MetaReviewAdmin(admin.ModelAdmin):
    list_display = ["paper", "last_update", "text_preview"]
    list_filter = ["last_update"]
    search_fields = ["text", "paper__title"]
    readonly_fields = ["last_update"]

    def text_preview(self, obj):
        return obj.text[:100] + "..." if len(obj.text) > 100 else obj.text

    text_preview.short_description = "Text preview"
