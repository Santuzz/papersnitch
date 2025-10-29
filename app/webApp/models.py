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
    text = models.TextField(verbose_name="Full paper text")
    datasets = models.TextField(
        verbose_name="Datasets used",
        blank=True,
        help_text="Comma-separated dataset names",
    )
    suppl_materials = models.TextField(
        verbose_name="Supplementary materials", blank=True
    )
    pdf_url = models.URLField(verbose_name="PDF URL", blank=True, max_length=500)
    conference = models.ForeignKey(
        Conference,
        on_delete=models.PROTECT,
        related_name="papers",
        verbose_name="Conference",
        null=True,
    )
    last_update = models.DateTimeField(auto_now=True, verbose_name="Last update")

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


class Review(models.Model):

    paper = models.ForeignKey(
        Paper, on_delete=models.CASCADE, related_name="reviews", verbose_name="Paper"
    )
    text = models.TextField(verbose_name="Review text")
    last_update = models.DateTimeField(auto_now=True, verbose_name="Last update")

    class Meta:
        verbose_name = "Review"
        verbose_name_plural = "Reviews"

    def __str__(self):
        return f"Review for {self.paper.title}"


class MetaReview(models.Model):

    paper = models.ForeignKey(
        Paper,
        on_delete=models.CASCADE,
        related_name="metareviews",
        verbose_name="Paper",
    )
    text = models.TextField(verbose_name="Meta-review text")
    last_update = models.DateTimeField(auto_now=True, verbose_name="Last update")

    class Meta:
        verbose_name = "Meta-Review"
        verbose_name_plural = "Meta-Reviews"

    def __str__(self):
        return f"MetaReview for {self.paper.title}"
