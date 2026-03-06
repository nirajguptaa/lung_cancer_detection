from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
import os


def generate_report(data, output_path):

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Lung Cancer Detection Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # -------------------------
    # Patient Information
    # -------------------------

    elements.append(Paragraph("Patient Information", styles["Heading2"]))

    patient_table = Table([
        ["Age", data["age"]],
        ["Smoking History", data["smoking"]],
        ["Family History", data["family_history"]],
        ["Symptoms", data["symptoms"]],
    ])

    patient_table.setStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("BACKGROUND",(0,0),(0,-1),colors.lightgrey)
    ])

    elements.append(patient_table)
    elements.append(Spacer(1,20))

    # -------------------------
    # Prediction Result
    # -------------------------

    elements.append(Paragraph("Prediction Result", styles["Heading2"]))

    prediction_table = Table([
        ["Prediction", data["result"]],
        ["Confidence", f"{data['confidence']:.2f}%"],
        ["Risk Level", data["risk"]],
    ])

    prediction_table.setStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.grey),
        ("BACKGROUND",(0,0),(0,-1),colors.lightgrey)
    ])

    elements.append(prediction_table)
    elements.append(Spacer(1,20))

    # -------------------------
    # Model Probabilities
    # -------------------------

    elements.append(Paragraph("Model Probabilities", styles["Heading2"]))

    classes = ["Benign", "Malignant", "Normal"]

    prob_table_data = [["Class", "Probability"]]

    for c,p in zip(classes,data["probs"]):
        prob_table_data.append([c,f"{p*100:.2f}%"])

    prob_table = Table(prob_table_data)

    prob_table.setStyle([
        ("GRID",(0,0),(-1,-1),1,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)
    ])

    elements.append(prob_table)
    elements.append(Spacer(1,20))

    # -------------------------
    # CT Scan Image
    # -------------------------

    if data.get("scan_path") and os.path.exists(data["scan_path"]):
        elements.append(Paragraph("Original CT Scan", styles["Heading2"]))
        elements.append(Image(data["scan_path"], width=4*inch, height=4*inch))
        elements.append(Spacer(1,20))

    # -------------------------
    # GradCAM
    # -------------------------

    if data.get("heatmap_path") and os.path.exists(data["heatmap_path"]):
        elements.append(Paragraph("Grad-CAM Visualization", styles["Heading2"]))
        elements.append(Image(data["heatmap_path"], width=4*inch, height=4*inch))
        elements.append(Spacer(1,20))

    # -------------------------
    # Recommendations
    # -------------------------

    elements.append(Paragraph("Recommended Next Steps", styles["Heading2"]))

    if data["result"] == "Malignant":
        rec = """
        • Consult an oncologist or pulmonologist immediately.<br/>
        • Further diagnostic tests such as biopsy or PET scan may be recommended.<br/>
        • Follow-up imaging may be required.<br/>
        """
    elif data["result"] == "Benign":
        rec = """
        • Monitor the growth with periodic CT scans.<br/>
        • Follow doctor recommendations.<br/>
        """
    else:
        rec = """
        • Maintain healthy lifestyle.<br/>
        • Regular medical checkups.<br/>
        """

    elements.append(Paragraph(rec, styles["Normal"]))
    elements.append(Spacer(1,20))

    # -------------------------
    # Disclaimer
    # -------------------------

    elements.append(Paragraph("Medical Disclaimer", styles["Heading2"]))
    elements.append(Paragraph(
        "This AI system is developed for academic and research purposes only. "
        "It does not replace professional medical diagnosis. "
        "Always consult a qualified healthcare professional.",
        styles["Normal"]
    ))

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    doc.build(elements)