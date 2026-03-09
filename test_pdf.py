from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
import re

html = markdown2.markdown("**Hello** World!")
print("HTML:", html)
