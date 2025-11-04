from django.db import models
from django.utils import timezone


class Operations(models.Model):
    """Different operations of analizer"""

    name = models.CharField(max_length=100, unique=True, verbose_name="Operation type")

    class Meta:
        verbose_name = "Operations"
        verbose_name_plural = "Operations"

    def __str__(self):
        return self.name


class Conference(models.Model):

    name = models.CharField(max_length=300, verbose_name="Conference name")
    year = models.IntegerField(verbose_name="Year", blank=True, null=True)
    website = models.URLField(verbose_name="Website", blank=True, max_length=500)
    last_update = models.DateTimeField(auto_now=True, verbose_name="Last update")

    class Meta:
        ordering = ["name"]
        verbose_name = "Conference"
        verbose_name_plural = "Conferences"

    def __str__(self):
        return f"{self.name}{self.year}"


class Paper(models.Model):

    title = models.CharField(max_length=500, verbose_name="Title")
    doi = models.CharField(
        max_length=255, unique=True, verbose_name="DOI", blank=True, null=True
    )
    abstract = models.TextField(verbose_name="Abstract")
    datasets = models.TextField(
        verbose_name="Datasets used",
        blank=True,
        null=True,
        help_text="Comma-separated dataset names",
    )
    supp_materials = models.TextField(
        verbose_name="Supplementary materials", blank=True, null=True
    )
    paper_url = models.URLField(verbose_name="Paper URL", blank=True, max_length=500)
    code_url = models.URLField(verbose_name="code URL", blank=True, max_length=500)
    conference = models.ForeignKey(
        Conference,
        on_delete=models.PROTECT,
        related_name="papers",
        verbose_name="Conference",
        blank=True,
        null=True,
    )
    pdf_file = models.FileField(
        upload_to="pdf",
        blank=True,
        null=True,
    )

    text = models.TextField(verbose_name="Full paper text", blank=True, null=True)
    reviews = models.TextField(verbose_name="All reviews text", blank=True, null=True)
    author_feedback = models.TextField(
        verbose_name="Author feedback", blank=True, null=True
    )
    meta_review = models.TextField(
        verbose_name="All Meta-reviews text", blank=True, null=True
    )
    last_update = models.DateTimeField(
        auto_now=True, verbose_name="Last update", blank=True, null=True
    )

    class Meta:
        verbose_name = "Paper"
        verbose_name_plural = "Papers"

    def __str__(self):
        return self.title


class Author(models.Model):

    papers = models.ManyToManyField(Paper, related_name="authors", verbose_name="Paper")
    name = models.CharField(max_length=500, verbose_name="Author name")
    flag = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Author"
        verbose_name_plural = "Authors"

    def __str__(self):
        return self.name
