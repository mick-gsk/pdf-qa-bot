"""Generate the example PDFs used in the README, the tests and the evaluation.

- employee-handbook.pdf: a short HR handbook (3 pages)
- espresso-manual.pdf: a user manual with error codes and specifications (7 pages)

Both companies and products are fictional.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

HANDBOOK = [
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

MANUAL = [
    ("Brava One Espresso Machine: User Manual", [
        "Thank you for choosing the Brava One. This manual covers setup, daily use, cleaning and "
        "troubleshooting. Model number BR-100, firmware 2.4. Keep this manual for later reference.",
        "Safety. Connect the machine only to a grounded outlet with 220 to 240 volts. Never immerse the "
        "machine, the power cord or the plug in water. Children under 8 years must not use or clean the "
        "machine. The steam wand and the brew group get hot during use; do not touch them until they have "
        "cooled for at least 10 minutes.",
    ]),
    ("Setup", [
        "Place the machine on a flat, dry surface at least 10 cm away from the wall. Remove the water tank, "
        "rinse it and fill it with fresh, cold water up to the MAX mark. The tank holds 1.8 liters.",
        "Before the first use, insert the water filter cartridge. Soak it in cold water for 5 minutes, then "
        "screw it into the holder inside the tank. Set the water hardness in the menu under Settings, Water "
        "hardness. The factory setting is level 3 (medium).",
        "Press the power button. The machine heats up in about 45 seconds and rinses the brew group "
        "automatically. The coffee button lights up white when the machine is ready.",
    ]),
    ("Making Coffee", [
        "Fill the bean container with up to 250 grams of whole beans. Do not use oily, caramelized or "
        "flavored beans, as they can block the grinder.",
        "The grinder has 12 settings. Setting 1 is the finest and setting 12 the coarsest. Change the grind "
        "setting only while the grinder is running. For a classic espresso we recommend settings 4 to 6.",
        "Press the espresso button once for one cup (40 ml) or twice for two cups. Hold the button for "
        "3 seconds to save a new cup volume between 20 and 220 ml. The brew temperature can be set to "
        "low (90 degrees Celsius), medium (92 degrees Celsius) or high (94 degrees Celsius).",
        "Milk. Immerse the steam wand in cold milk and turn the steam knob counterclockwise. Wipe the steam "
        "wand with a damp cloth and release steam for 2 seconds after every use.",
    ]),
    ("Cleaning and Maintenance", [
        "Empty the drip tray and the coffee grounds container daily. The grounds container holds up to "
        "12 portions; the display shows a message when it is full.",
        "Rinse the brew group under running water once a week. Do not use detergent and do not put the brew "
        "group in the dishwasher. Let it air-dry before inserting it again.",
        "Replace the water filter cartridge every 2 months or after 50 liters of water, whichever comes "
        "first. Only use Brava CF-2 filter cartridges.",
        "Descaling. The machine asks for descaling with the message Descale. Use one bottle of Brava "
        "descaler (250 ml) mixed with water up to the 0.5 liter mark. Descaling takes about 25 minutes and "
        "must not be interrupted. Remove the water filter before descaling.",
    ]),
    ("Troubleshooting", [
        "E01: The brew group is missing or not inserted correctly. Insert the brew group and close the "
        "service door.",
        "E03: The grinder is blocked. Switch off the machine, empty the bean container and remove foreign "
        "objects with the cleaning brush. If the error remains, contact customer service.",
        "E05: The water tank is empty or the float is stuck. Fill the tank and check that the float moves "
        "freely.",
        "E09: The water circuit is blocked by limescale. Descale the machine. If E09 appears again within "
        "one week, contact customer service.",
        "Coffee is too watery: choose a finer grind setting and use fresh beans. Coffee is not hot enough: "
        "set the brew temperature to high and preheat the cups.",
    ]),
    ("Technical Specifications", [
        "Power: 1450 watts. Voltage: 220 to 240 volts, 50 Hz. Pump pressure: 15 bar. Water tank: 1.8 liters. "
        "Bean container: 250 grams. Grounds container: 12 portions. Weight: 8.2 kg. Dimensions: "
        "24 cm wide, 44 cm deep, 36 cm high. Power consumption in standby: under 0.5 watts.",
        "The machine switches off automatically after 30 minutes without use. The auto-off time can be set "
        "to 15, 30, 60 or 120 minutes.",
    ]),
    ("Warranty and Service", [
        "Brava gives a warranty of 2 years from the date of purchase. Registering the machine at the Brava "
        "website within 30 days extends the warranty to 3 years.",
        "The warranty does not cover damage caused by missing descaling, by using unsuitable beans, or by "
        "repairs from unauthorized service partners.",
        "Customer service is available Monday to Friday from 8:00 to 18:00 at +49 30 1234 5678. Please have "
        "the serial number ready; it is printed on the label behind the drip tray.",
    ]),
]


def build(out: Path, title: str, pages: list[tuple[str, list[str]]]) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (heading, paragraphs) in enumerate(pages):
        story.append(Paragraph(heading, styles["Heading1"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 8)]
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(out), pagesize=A4, title=title).build(story)
    print(out)


def main() -> None:
    here = Path(__file__).parent
    build(here / "employee-handbook.pdf", "Example Corp Employee Handbook", HANDBOOK)
    build(here / "espresso-manual.pdf", "Brava One User Manual", MANUAL)


if __name__ == "__main__":
    main()
