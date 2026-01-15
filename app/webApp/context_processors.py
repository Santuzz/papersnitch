# webApp/context_processors.py
from .models import AnalysisTask


def unread_analyses(request):
    """
    Context processor to provide unread analysis count for the navbar badge.
    """
    if request.user.is_authenticated:
        count = AnalysisTask.objects.filter(
            user=request.user, status="completed", is_read=False
        ).count()
        return {"unread_analyses_count": count}
    return {"unread_analyses_count": 0}
