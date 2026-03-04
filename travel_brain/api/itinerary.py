"""
itinerary.py — Endpoint to convert AI-generated itineraries (Markdown) into downloadable PDFs.
"""

import io
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
import markdown2
from xhtml2pdf import pisa
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/itinerary", tags=["Itinerary"])

class PDFRequest(BaseModel):
    title: str = "Travel Brain Itinerary"
    markdown_content: str

def generate_pdf_from_html(html_content: str) -> io.BytesIO:
    """Uses xhtml2pdf to generate a PDF from an HTML string."""
    pdf_buffer = io.BytesIO()
    # Create PDF
    pisa_status = pisa.CreatePDF(
        io.StringIO(html_content),
        dest=pdf_buffer
    )
    if pisa_status.err:
        raise Exception("PDF generation failed")
    
    pdf_buffer.seek(0)
    return pdf_buffer

@router.post("/pdf")
async def generate_pdf(req: PDFRequest):
    """
    Converts a Markdown itinerary into a beautifully formatted PDF.
    """
    try:
        # Convert Markdown to HTML
        html_body = markdown2.markdown(req.markdown_content)
        
        # Wrap in minimal CSS styling for xhtml2pdf
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: a4 portrait;
                    margin: 2cm;
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 14px;
                    line-height: 1.6;
                    color: #333333;
                }}
                h1 {{
                    color: #1e3a8a;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                h2 {{
                    color: #2563eb;
                    border-bottom: 1px solid #e5e7eb;
                    padding-bottom: 5px;
                    margin-top: 25px;
                }}
                h3 {{
                    color: #3b82f6;
                }}
                a {{
                    color: #2563eb;
                    text-decoration: none;
                }}
                ul, ol {{
                    margin-bottom: 15px;
                }}
                li {{
                    margin-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>{req.title}</h1>
            {html_body}
        </body>
        </html>
        """
        
        pdf_buffer = generate_pdf_from_html(html_content)
        
        # Return PDF as a downloadable file stream
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{req.title.replace(" ", "_").lower()}.pdf"'
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF: {{e}}")
        return Response(content="Failed to generate PDF", status_code=500)
