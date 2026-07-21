import io
import os
import time

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, SimpleDocTemplate

from charts import make_confusion_matrix_figure, make_roc_figure


def generate_pdf_report(metrics, model_path):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=colors.HexColor('#0F52BA'), spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13,
        textColor=colors.HexColor('#64748b'), spaceAfter=22
    )

    h1_style = ParagraphStyle(
        'SectionHeader', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=13, leading=16,
        textColor=colors.HexColor('#0F52BA'), spaceBefore=12, spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=colors.HexColor('#0f172a'), spaceAfter=8
    )

    story = []

    story.append(Paragraph("Leukemia Detection Diagnostic Report", title_style))
    story.append(Paragraph(f"AI-Powered Blood Smear Analysis &bull; Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Model & Database Metadata", h1_style))
    meta_data = [
        [Paragraph("<b>Model Family:</b>", body_style), Paragraph("DenseNet121 Deep Neural Network", body_style)],
        [Paragraph("<b>Model Location:</b>", body_style), Paragraph(os.path.basename(model_path), body_style)],
        [Paragraph("<b>Primary Target:</b>", body_style), Paragraph("Acute Lymphoblastic Leukemia (ALL) Detection", body_style)],
        [Paragraph("<b>Test Benchmark:</b>", body_style), Paragraph(f"Random Validation Sample Split ({metrics['support']} samples)", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[140, 360])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Diagnostic Metric Diagnostics", h1_style))
    summary_data = [
        ["Classification Metric", "Representative Score"],
        ["Model General Accuracy", f"{metrics['accuracy'] * 100:.2f}%"],
        ["Inference Precision (Leukemia)", f"{metrics['precision'] * 100:.2f}%"],
        ["Inference Recall (Leukemia)", f"{metrics['recall'] * 100:.2f}%"],
        ["Inference F1-Score (Leukemia)", f"{metrics['f1'] * 100:.2f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[240, 260])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F52BA')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,1), (-1,-1), 5),
        ('TOPPADDING', (0,1), (-1,-1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Visual Performance Metrics", h1_style))

    cm_fig = make_confusion_matrix_figure(metrics["confusion_matrix"], metrics["class_names"])
    cm_buf = io.BytesIO()
    cm_fig.savefig(cm_buf, format='png', dpi=220, bbox_inches='tight')
    cm_buf.seek(0)

    roc_fig = make_roc_figure(*metrics["roc_curve"], metrics["roc_auc"])
    roc_buf = io.BytesIO()
    roc_fig.savefig(roc_buf, format='png', dpi=220, bbox_inches='tight')
    roc_buf.seek(0)

    rl_cm_img = RLImage(cm_buf, width=230, height=200)
    rl_roc_img = RLImage(roc_buf, width=230, height=200)

    plot_table = Table([[rl_cm_img, rl_roc_img]], colWidths=[250, 250])
    plot_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(plot_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
