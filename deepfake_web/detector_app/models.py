from django.db import models
import uuid

class InferenceJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.FileField(upload_to='videos/')
    status = models.CharField(max_length=20, default='PENDING') # PENDING, PROCESSING, COMPLETED, FAILED
    progress = models.FloatField(default=0.0)
    result_label = models.CharField(max_length=10, blank=True, null=True) # FAKE, REAL
    confidence = models.JSONField(blank=True, null=True) # Per-modality confidence
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    report_pdf = models.FileField(upload_to='reports/', blank=True, null=True)

    def __str__(self):
        return f"{self.id} - {self.status}"
