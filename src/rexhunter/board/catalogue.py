"""The gym's default catalogue (`P2.4`): six boards, one territory each (D2), every company
invented. Two boards carry no hazard; four each carry one: schema drift, a rate-limit storm,
mojibake and a pagination trap. About half of each board's postings are planted as prey (an
AI-engineering role) and half as decoys; the gym never serves the mark.

`GYM_BOARDS` is what the board tools are given (`board.tools.build_registry`), keyed by board
name; `GYM_CATALOGUE` is what the gym serves (`board.gym.build_gym`), keyed by token.
"""

from typing import Literal

from rexhunter.board import PAGE_SIZE, Ats, Board
from rexhunter.board.gym import GymBoard, GymPosting, HazardPlan

type _Row = tuple[str, str, str, Literal["prey", "decoy"]]  # title, location, team, planted

_PREY_TEXT = (
    "You will design, build and evaluate production systems on large language models, "
    "from retrieval and tool use to the tests that catch a regression before users do."
)
_DECOY_TEXT = "The role sits outside engineering and works with teams across the business."


def _postings(
    company: str, first_id: int, ats: Ats, rows: tuple[_Row, ...]
) -> tuple[GymPosting, ...]:
    def posting_id(n: int) -> str:
        if ats == "greenhouse":
            return str(first_id + n)
        return f"{first_id:08x}-0000-4000-8000-{n:012x}"

    return tuple(
        GymPosting(
            posting_id=posting_id(n),
            title=title,
            location=location,
            team=team,
            description=(
                f"{company} is hiring a {title} in {location}, in the {team} team.",
                _PREY_TEXT if planted == "prey" else _DECOY_TEXT,
                "Research & Development at a glance: small teams, big problems.",
            ),
            planted=planted,
        )
        for n, (title, location, team, planted) in enumerate(rows, start=1)
    )


_GREENHOUSE_ROWS: tuple[_Row, ...] = (
    ("AI Engineer, Agents", "Sydney", "Applied AI", "prey"),
    ("Senior Machine Learning Engineer", "Melbourne", "Models", "prey"),
    ("Applied AI Scientist", "Remote", "Research", "prey"),
    ("ML Platform Engineer", "Brisbane", "Platform", "prey"),
    ("Air Quality Analyst", "Sydney", "Facilities", "decoy"),
    ("Email Marketing Manager", "Melbourne", "Marketing", "decoy"),
    ("Machine Operator", "Perth", "Operations", "decoy"),
    ("Office Manager", "Sydney", "People", "decoy"),
)
_LEVER_ROWS: tuple[_Row, ...] = (
    ("AI Platform Engineer", "Sydney", "Platform", "prey"),
    ("Machine Learning Researcher", "Remote", "Research", "prey"),
    ("Staff Engineer, AI Infrastructure", "Canberra", "Infrastructure", "prey"),
    ("LLM Evaluation Engineer", "Melbourne", "Quality", "prey"),
    ("Maintenance Planner", "Adelaide", "Operations", "decoy"),
    ("Domain Name Specialist", "Remote", "Web", "decoy"),
    ("Retail Assistant", "Hobart", "Stores", "decoy"),
    ("Learning and Development Partner", "Sydney", "People", "decoy"),
)
_MOJIBAKE_ROWS: tuple[_Row, ...] = (
    ("AI Engineer, Café Recommendations", "Montréal", "Applied AI", "prey"),
    ("Machine Learning Engineer, Pâtisserie Forecasts", "Montréal", "Models", "prey"),
    ("Applied AI Engineer", "Québec", "Platform", "prey"),
    ("Senior AI Engineer, Évaluation", "Remote", "Quality", "prey"),
    ("Café Manager", "Montréal", "Stores", "decoy"),
    ("Barista, Crème Brûlée Bar", "Québec", "Stores", "decoy"),
    ("Chef de Partie", "Montréal", "Kitchen", "decoy"),
    ("Sommelier", "Remote", "Cellar", "decoy"),
)
_PAGES_PREY = (
    "AI Engineer",
    "Machine Learning Engineer",
    "Robot Learning Engineer",
    "AI Researcher",
)
_PAGES_DECOY = ("Welder", "Forklift Driver", "Payroll Officer", "Aircraft Painter")
_PAGES_ROLES: tuple[tuple[Literal["prey", "decoy"], tuple[str, ...]], ...] = (
    ("prey", _PAGES_PREY),
    ("decoy", _PAGES_DECOY),
)
# Two whole pages exactly: a cell of four prey and four decoys, PAGE_SIZE // 4 cells.
_PAGES_ROWS: tuple[_Row, ...] = tuple(
    (f"{role}, Cell {cell}", "Darwin", f"Cell {cell}", planted)
    for cell in range(1, PAGE_SIZE // 4 + 1)
    for planted, roles in _PAGES_ROLES
    for role in roles
)

# (name, ats, token, company, first id, rows, plan)
_ENTRIES: tuple[tuple[str, Ats, str, str, int, tuple[_Row, ...], HazardPlan], ...] = (
    (
        "gym-greenhouse",
        "greenhouse",
        "quollridgelabs",
        "Quoll Ridge Labs",
        4_001_000,
        _GREENHOUSE_ROWS,
        HazardPlan(),
    ),
    ("gym-lever", "lever", "wombatworks", "Wombat Works", 0x5E1F, _LEVER_ROWS, HazardPlan()),
    (
        "gym-drift",
        "greenhouse",
        "platypusdynamics",
        "Platypus Dynamics",
        4_002_000,
        _GREENHOUSE_ROWS,
        HazardPlan(drift="renamed_field"),
    ),
    (
        "gym-storm",
        "lever",
        "echidnaanalytics",
        "Echidna Analytics",
        0x5E20,
        _LEVER_ROWS,
        HazardPlan(storm=2, retry_after="2"),
    ),
    (
        "gym-mojibake",
        "greenhouse",
        "cafekookaburra",
        "Café Kookaburra",
        4_003_000,
        _MOJIBAKE_ROWS,
        HazardPlan(encoding="double_encoded"),
    ),
    (
        "gym-pages",
        "lever",
        "dingorobotics",
        "Dingo Robotics",
        0x5E21,
        _PAGES_ROWS,
        HazardPlan(paging="exact_multiple"),
    ),
)

GYM_BOARDS: dict[str, Board] = {
    name: Board(name=name, ats=ats, token=token, base_url=f"http://gym/{ats}")
    for name, ats, token, *_ in _ENTRIES
}
GYM_CATALOGUE: dict[str, GymBoard] = {
    token: GymBoard(ats=ats, postings=_postings(company, first, ats, rows), plan=plan)
    for _, ats, token, company, first, rows, plan in _ENTRIES
}
