"""Generate examples/employee-handbook.pdf, a small fictional handbook for demos and tests."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Employee Handbook of Example Corp", [
        "This handbook explains the most important rules for all employees of Example Corp, a fictional company "
        "used for this demo. It applies from 1 January 2026.",
        "Vacation. Full-time employees receive 28 vacation days per calendar year. Part-time employees receive "
        "vacation days in proportion to their working days. Vacation requests must be submitted in the HR portal "
        "at least two weeks in advance. Up to five unused vacation days can be carried over until 31 March of the "
        "following year.",
        "Sick leave. If you are sick, inform your team lead before 9:00 on the first day. A doctor's note is "
        "required from the third day of illness.",
    ]),
    ("Remote Work", [
        "Employees may work remotely up to three days per week after their probation period of six months. "
        "Remote work from abroad is allowed for up to 20 working days per year and must be approved by HR.",
        "Core hours are from 10:00 to 15:00 CET. During core hours employees should be reachable by chat. "
        "The company provides a laptop and pays a monthly home office allowance of 30 euros.",
    ]),
    ("Expenses and Travel", [
        "Business travel must be booked through the travel tool. Train travel is preferred for trips under "
        "four hours. Second class is standard; first class is allowed for trips longer than four hours.",
        "Hotel costs are reimbursed up to 120 euros per night, or up to 160 euros per night in Munich, Paris and "
        "London. Meals during business travel are covered by a daily allowance of 28 euros.",
        "Expense reports must be submitted within 30 days with all receipts attached. Reports submitted later "
        "than 90 days cannot be reimbursed.",
    ]),
]


def main() -> None:
    out = Path(__file__).with_name("employee-handbook.pdf")
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 8)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4, title="Example Corp Employee Handbook").build(story)
    print(out)


if __name__ == "__main__":
    main()
