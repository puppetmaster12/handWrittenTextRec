from reportlab.pdfgen import canvas

import datetime
import os

def test_savepdf():
    text = "This is a test document"
    t = datetime.datetime.now()
    current = t.strftime('%m-%d-%y')
    save_name = "./test_doc/test_pdf_" + current + ".pdf"
    c = canvas.Canvas(save_name)
    c.drawString(40, 800, text)
    c.save()

    assert os.path.exists(save_name) == 1
