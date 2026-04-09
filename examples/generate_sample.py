#!/usr/bin/env python3
"""Generate sample mill certificate PDFs for demo/testing purposes.

All data is fictional. No real companies or products.
"""

import random
from pathlib import Path

from fpdf import FPDF


class MillCertPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, self.company_name, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, self.company_address, align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(
            0,
            5,
            f"Tel: {self.company_tel}  |  Fax: {self.company_fax}",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(3)
        self.set_font("Helvetica", "B", 14)
        self.cell(
            0, 10, "INSPECTION CERTIFICATE EN 10204 - 3.1", align="C", new_x="LMARGIN", new_y="NEXT"
        )
        self.ln(2)
        self.set_draw_color(0)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-25)
        self.set_font("Helvetica", "I", 7)
        self.cell(
            0,
            4,
            "This document is generated for demonstration purposes only.",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.cell(
            0,
            4,
            "All company names, data, and values are entirely fictional.",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.set_font("Helvetica", "", 8)
        self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="C")


def _info_row(pdf, label, value):
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, 6, label, border=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(135, 6, str(value), border=1, new_x="LMARGIN", new_y="NEXT")


def _table_header(pdf, headers, widths):
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 220, 220)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()


def _table_row(pdf, values, widths):
    pdf.set_font("Helvetica", "", 8)
    for v, w in zip(values, widths):
        pdf.cell(w, 6, str(v), border=1, align="C")
    pdf.ln()


def generate_cert(output_path: Path, seed: int = None):
    if seed is not None:
        random.seed(seed)

    companies = [
        (
            "Acero Steelworks S.p.A.",
            "Via dell'Industria 42, 25100 Brescia, Italy",
            "+39 030 555 1234",
            "+39 030 555 5678",
        ),
        (
            "Nordic Steel AB",
            "Stalgatan 15, SE-811 30 Sandviken, Sweden",
            "+46 26 555 100",
            "+46 26 555 200",
        ),
        (
            "Rheinland Stahlwerke GmbH",
            "Stahlstrasse 8, 47053 Duisburg, Germany",
            "+49 203 555 300",
            "+49 203 555 400",
        ),
        (
            "Britannia Metals Ltd",
            "Forge Road, Sheffield S9 1XX, United Kingdom",
            "+44 114 555 6000",
            "+44 114 555 6001",
        ),
        (
            "Atlantis Heavy Industries S.A.",
            "Zona Industrial Norte, 4520 Santa Maria da Feira, Portugal",
            "+351 256 555 700",
            "+351 256 555 800",
        ),
    ]

    grades = ["S235JR", "S275JR", "S355J2", "S355JR", "S460ML", "S500MC", "S690QL"]
    products = [
        ("Hot Rolled Steel Plate", "mm"),
        ("Steel Rebar", "mm"),
        ("Structural Steel Beam", "mm"),
        ("Cold Rolled Coil", "mm"),
        ("Wire Rod", "mm"),
    ]
    standards = ["EN 10025-2", "EN 10027-1", "EN 10204:2004", "EN 10002-1"]

    company = random.choice(companies)
    grade = random.choice(grades)
    product_name, unit = random.choice(products)
    heat_no = f"H{random.randint(10000, 99999)}"
    cert_no = f"MC-{random.randint(100000, 999999)}"
    order_no = f"PO-{random.randint(10000, 99999)}"
    batch_no = f"B{random.randint(1000, 9999)}"
    date = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2026"
    thickness = round(random.uniform(4.0, 50.0), 1)
    width = random.randint(800, 2500)
    length = random.randint(2000, 12000)
    weight = round(random.uniform(0.5, 25.0), 2)
    n_items = random.randint(1, 8)

    pdf = MillCertPDF()
    pdf.company_name = company[0]
    pdf.company_address = company[1]
    pdf.company_tel = company[2]
    pdf.company_fax = company[3]
    pdf.alias_nb_pages()
    pdf.add_page()

    # Certificate info
    _info_row(pdf, "Certificate No:", cert_no)
    _info_row(pdf, "Date:", date)
    _info_row(pdf, "Customer Order No:", order_no)
    _info_row(pdf, "Heat No:", heat_no)
    _info_row(pdf, "Batch No:", batch_no)
    _info_row(pdf, "Product:", product_name)
    _info_row(pdf, "Steel Grade:", grade)
    _info_row(pdf, "Standard:", " / ".join(random.sample(standards, 2)))
    _info_row(pdf, "Dimensions:", f"{thickness} x {width} x {length} {unit}")
    _info_row(pdf, "No. of Items:", str(n_items))
    _info_row(pdf, "Total Weight (tonnes):", str(weight))
    pdf.ln(8)

    # Chemical composition
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "CHEMICAL COMPOSITION (LADLE ANALYSIS) - %", new_x="LMARGIN", new_y="NEXT")

    elements = ["C", "Si", "Mn", "P", "S", "Cr", "Ni", "Mo", "Cu", "V", "N", "CEV"]
    widths_chem = [190 // len(elements)] * len(elements)
    widths_chem[-1] += 190 - sum(widths_chem)

    _table_header(pdf, elements, widths_chem)

    # Specified max
    maxvals = [0.22, 0.55, 1.60, 0.035, 0.035, 0.30, 0.30, 0.08, 0.55, 0.12, 0.012, 0.47]
    _table_row(pdf, [f"{v:.3f}" for v in maxvals], widths_chem)

    # Actual values (slightly below max)
    actuals = [round(v * random.uniform(0.3, 0.85), 3) for v in maxvals]
    _table_row(pdf, [f"{v:.3f}" for v in actuals], widths_chem)
    pdf.ln(8)

    # Mechanical properties
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "MECHANICAL PROPERTIES", new_x="LMARGIN", new_y="NEXT")

    mech_headers = ["Test", "Direction", "Temp (C)", "Specified", "Actual", "Result"]
    mech_widths = [35, 25, 20, 35, 35, 40]

    _table_header(pdf, mech_headers, mech_widths)

    yield_min = random.choice([235, 275, 355, 460])
    tensile_min = yield_min + random.randint(80, 200)
    yield_actual = yield_min + random.randint(10, 80)
    tensile_actual = tensile_min + random.randint(10, 60)
    elong_min = random.randint(17, 26)
    elong_actual = elong_min + random.randint(2, 10)
    charpy_min = random.randint(20, 40)
    charpy_actual = charpy_min + random.randint(5, 30)

    _table_row(
        pdf,
        ["Yield Strength (MPa)", "L", "20", f"min {yield_min}", str(yield_actual), "PASS"],
        mech_widths,
    )
    _table_row(
        pdf,
        [
            "Tensile Strength (MPa)",
            "L",
            "20",
            f"{tensile_min}-{tensile_min + 200}",
            str(tensile_actual),
            "PASS",
        ],
        mech_widths,
    )
    _table_row(
        pdf,
        ["Elongation (%)", "L", "20", f"min {elong_min}", str(elong_actual), "PASS"],
        mech_widths,
    )
    _table_row(
        pdf,
        [
            "Charpy V (J)",
            "T",
            f"-{random.choice([20, 40])}",
            f"min {charpy_min}",
            str(charpy_actual),
            "PASS",
        ],
        mech_widths,
    )
    pdf.ln(10)

    # Stamp area
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "CERTIFICATION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0,
        5,
        "We hereby certify that the above material has been manufactured, tested and inspected "
        "in accordance with the requirements of the above specification(s) and found to be in "
        "conformity therewith.",
    )
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(95, 6, "Quality Department", border="T")
    pdf.cell(95, 6, "Authorised Signatory", border="T", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    names = ["A. Rossi", "M. Lindqvist", "K. Schmidt", "J. Thompson", "P. Silva"]
    pdf.cell(95, 6, random.choice(names))
    pdf.cell(95, 6, random.choice(names), align="R")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return output_path


def main():
    output_dir = Path(__file__).parent
    certs = []
    for i in range(3):
        path = output_dir / f"sample_cert_{i + 1:03d}.pdf"
        generate_cert(path, seed=42 + i)
        certs.append(path)
        print(f"Generated: {path}")
    print(f"\n{len(certs)} sample certificates created in {output_dir}")


if __name__ == "__main__":
    main()
