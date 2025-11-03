#!/usr/bin/env python3
"""Convertir le fichier texte en PDF pour l'indexation dans Kibali"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def convert_txt_to_pdf(txt_path, pdf_path):
    """Convertir un fichier texte en PDF avec formatage"""
    
    # Lire le contenu
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Créer le PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor='#2C3E50',
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='#34495E',
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        spaceAfter=6
    )
    
    # Construire le document
    story = []
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        if not line:
            story.append(Spacer(1, 0.1*inch))
            continue
            
        if line.startswith('# '):
            # Titre principal
            text = line[2:].strip()
            story.append(Paragraph(text, title_style))
        elif line.startswith('## '):
            # Sous-titre
            text = line[3:].strip()
            story.append(Paragraph(text, heading_style))
        elif line.startswith('### '):
            # Sous-sous-titre
            text = line[4:].strip()
            p = Paragraph(f"<b>{text}</b>", body_style)
            story.append(p)
        elif line.startswith('**') and line.endswith('**'):
            # Texte en gras
            text = line.replace('**', '')
            if ':' in text:
                key, value = text.split(':', 1)
                p = Paragraph(f"<b>{key}:</b>{value}", body_style)
            else:
                p = Paragraph(f"<b>{text}</b>", body_style)
            story.append(p)
        elif line.startswith('- '):
            # Liste à puces
            text = line[2:].strip()
            p = Paragraph(f"• {text}", body_style)
            story.append(p)
        elif line.startswith(('1. ', '2. ', '3. ')):
            # Liste numérotée
            text = line[3:].strip()
            num = line[0]
            p = Paragraph(f"{num}. {text}", body_style)
            story.append(p)
        else:
            # Texte normal
            if line:
                story.append(Paragraph(line, body_style))
    
    # Générer le PDF
    doc.build(story)
    print(f"✅ PDF créé: {pdf_path}")

if __name__ == '__main__':
    txt_file = '/root/RAG_ChatBot/pdfs/informations_kibali.txt'
    pdf_file = '/root/RAG_ChatBot/pdfs/informations_kibali.pdf'
    
    convert_txt_to_pdf(txt_file, pdf_file)
